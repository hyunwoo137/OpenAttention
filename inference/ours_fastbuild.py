#!/usr/bin/env python3
"""V70: the per-request build of ours (prefill side) made fast; the decode step is ours_graph OursG2 unchanged.

V69 measured the build of the deployed configuration at 11.1 ms per layer (1%; 13.5 ms at 3%), i.e. 356 / 431 ms
per request, 91% of the E2E gap to dense at batch 8.  Its stages were eager PyTorch:
  a  metric M: the closed-form RoPE window average recomputed per layer, although its coefficients depend only on
     the decode position T (not on the layer)                                           -> computed once per T
  b,d  two eigh calls per layer (KV-head M for QGS, query-head M for QGC)               -> ONE batched call for all
     32 layers x 40 matrices at the end of the prefill (bitwise identical to the separate calls; the build does not
     feed the prefill's own attention, so deferring it to the last layer changes nothing)
  f  4 Lloyd iterations: a (T x C) bf16 distance tensor, argmax, and scatter_add of 16-d points onto C addresses
     (heavy atomic contention)                                                          -> Triton: one kernel per
     iteration computes the distances in registers (same bf16 roundings as the bf16 matmul and subtraction), the
     argmax, and per-tile partial sums (fp32 atomics into a tile-private slice); a second sums the partials and
     updates the centroids (empty cluster -> its initial point, as before)
  g  n_b, sum k, sum k^2, sum v by scatter_add over a repeat_interleaved fp32 copy of the cache (2 x 0.5 GB)
                                                                                        -> CSR member lists (built
     anyway for the top-q) + one Triton kernel per (head, cluster) summing its members' rows in fp32
  e  the k-means seeds (CPU randperm per layer) are cached per (T, layer): the 8 rows of a batch share T

Bitwise identical to methods_eager/v66: M, the eigenvectors, U, scale, the int8 codes, W, the column means, the seeds.
Equal up to fp32 summation order (the reference's own scatter_add atomics are not reproducible either): centroids,
sums.  Assignments: equal except near-ties, which the selftest compares with the reference's run-to-run spread.

    python ours_fastbuild.py --selftest --device cuda:0
    python ours_fastbuild.py --profile-build --device cuda:0 --out results_v70_fastbuild/build_profile.json
    python ours_fastbuild.py --method ours3 --rho 0.01 --batch 8 --mode graph --device cuda:0 --out ...
    python ours_fastbuild.py --acc --method ours3 --rho 0.01 --max-requests 40 --device cuda:0 --out ...jsonl
"""
import json
import sys
import time

import torch
import triton
import triton.language as tl

import os as _os
HUB = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # repo root
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HUB)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import methods_eager as bl                                                  # noqa: E402
import ours_fused as F                                                      # noqa: E402
import graph_runner as G5                                                    # noqa: E402
import ours_graph as G6                                                    # noqa: E402
from methods_eager import C, D, G, HKV, HQ                                  # noqa: E402
sys.argv = _argv


# ============================================================ k-means (QGC) kernels
@triton.jit
def _k_km_assign(ZB, ZP, CB, HALF, ASG, PACC, PCNT, T, CN, NTL, ASG_S,
                 RC: tl.constexpr, CNP: tl.constexpr, BLOCK_T: tl.constexpr, NSUB: tl.constexpr):
    """one (query head, tile of NSUB x BLOCK_T tokens): nearest centroid as methods_eager computes it --
    bf16(bf16(z.c) - bf16 half), argmax with ties to the lower cluster -- and the tile's partial sums of the points
    per cluster, by fp32 atomic adds into this program's own (CNP x RC) slice (no contention across programs; a
    one-hot tensor-core product measured 2.5-3.7x slower).  Partial sums are small, so the fp32 error is below the
    reference's single scatter_add (5e-8 vs fp64)."""
    h = tl.program_id(0)
    t = tl.program_id(1)
    offs_r = tl.arange(0, RC)
    offs_c = tl.arange(0, CNP)
    cm = offs_c < CN
    cb = tl.load(CB + h * CN * RC + offs_c[:, None] * RC + offs_r[None, :], mask=cm[:, None], other=0.0)
    half = tl.load(HALF + h * CN + offs_c, mask=cm, other=0.0).to(tl.float32)
    base = (h * NTL + t) * CNP
    tl.store(PACC + (base + offs_c[:, None]) * RC + offs_r[None, :], tl.zeros([CNP, RC], tl.float32))
    tl.store(PCNT + base + offs_c, tl.zeros([CNP], tl.float32))
    tl.debug_barrier()
    for s in range(NSUB):
        offs_t = (t * NSUB + s) * BLOCK_T + tl.arange(0, BLOCK_T)
        tm = offs_t < T
        zb = tl.load(ZB + h * T * RC + offs_t[:, None] * RC + offs_r[None, :], mask=tm[:, None], other=0.0)
        dist = tl.dot(zb, tl.trans(cb)).to(tl.bfloat16).to(tl.float32)              # the bf16 matmul output
        dist = (dist - half[None, :]).to(tl.bfloat16).to(tl.float32)                # the bf16 subtraction
        dist = tl.where(cm[None, :], dist, -float("inf"))
        a = tl.argmax(dist, axis=1, tie_break_left=True)
        tl.store(ASG + h * ASG_S + offs_t, a.to(tl.int32), mask=tm)
        zp = tl.load(ZP + h * T * RC + offs_t[:, None] * RC + offs_r[None, :], mask=tm[:, None], other=0.0)
        tl.atomic_add(PACC + (base + a[:, None]) * RC + offs_r[None, :], zp, mask=tm[:, None], sem="relaxed")
        tl.atomic_add(PCNT + base + a, tl.full([BLOCK_T], 1.0, tl.float32), mask=tm, sem="relaxed")


@triton.jit
def _k_km_update(PACC, PCNT, ZP, INIT, CENT, CB, HALF, NB, T, CN, NTL,
                 RC: tl.constexpr, CNP: tl.constexpr, BC: tl.constexpr):
    """one (query head, block of BC clusters): sum the tile partials, centroid = mean (empty -> its initial point);
    also the bf16 copy and the bf16 half squared norm the next assignment uses, and the counts (n_b after the last
    iteration)."""
    h = tl.program_id(0)
    offs_r = tl.arange(0, RC)
    offs_c = tl.program_id(1) * BC + tl.arange(0, BC)
    cm = offs_c < CN
    acc = tl.zeros([BC, RC], tl.float32)
    cnt = tl.zeros([BC], tl.float32)
    for t in range(0, NTL):
        base = (h * NTL + t) * CNP
        acc += tl.load(PACC + (base + offs_c[:, None]) * RC + offs_r[None, :])
        cnt += tl.load(PCNT + base + offs_c)
    init = tl.load(INIT + offs_c, mask=cm, other=0)
    zi = tl.load(ZP + h * T * RC + init[:, None] * RC + offs_r[None, :], mask=cm[:, None], other=0.0)
    cent = tl.where(cnt[:, None] > 0, acc / tl.maximum(cnt, 1.0)[:, None], zi)
    sp = h * CN * RC + offs_c[:, None] * RC + offs_r[None, :]
    tl.store(CENT + sp, cent, mask=cm[:, None])
    tl.store(CB + sp, cent.to(tl.bfloat16), mask=cm[:, None])
    tl.store(HALF + h * CN + offs_c, (0.5 * tl.sum(cent * cent, axis=1)).to(tl.bfloat16), mask=cm)
    tl.store(NB + h * CN + offs_c, cnt, mask=cm)


@triton.jit
def _k_cluster_sums(K, V, MEM, CSTART, CLEN, SK, SK2, SV, Lmax, CN,
                    G: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, VAR: tl.constexpr):
    """one (query head, cluster): sum k, k^2, v of its members (CSR list) in fp32"""
    h = tl.program_id(0)
    c = tl.program_id(1)
    kvh = h // G
    offs_d = tl.arange(0, D)
    start = tl.load(CSTART + h * CN + c)
    ln = tl.load(CLEN + h * CN + c)
    sk = tl.zeros([D], tl.float32)
    sk2 = tl.zeros([D], tl.float32)
    sv = tl.zeros([D], tl.float32)
    for j0 in range(0, ln, BLOCK):
        oj = j0 + tl.arange(0, BLOCK)
        m = oj < ln
        tok = tl.load(MEM + h * Lmax + start + oj, mask=m, other=0)
        p = kvh * Lmax * D + tok[:, None] * D + offs_d[None, :]
        kt = tl.load(K + p, mask=m[:, None], other=0.0).to(tl.float32)
        vt = tl.load(V + p, mask=m[:, None], other=0.0).to(tl.float32)
        sk += tl.sum(kt, axis=0)
        sv += tl.sum(vt, axis=0)
        if VAR:
            sk2 += tl.sum(kt * kt, axis=0)
    o = (h * CN + c) * D + offs_d
    tl.store(SK + o, sk)
    tl.store(SV + o, sv)
    if VAR:
        tl.store(SK2 + o, sk2)


# ============================================================ the method
class OursG3(G6.OursG2):
    """ours_graph OursG2 (same decode) with the fast per-request build"""
    name = "ours3"
    VAR = True                   # keep sum k^2 (ErrorGate reads the variance)
    BC_KM = 16                   # clusters per k-means update program

    def __init__(self, a, B, Lmax, dev):
        super().__init__(a, B, Lmax, dev)
        self.CNP = triton.next_power_of_2(self.C)
        self.BT_KM, self.NSUB_KM = (64, 8) if self.CNP <= 64 else (128, 4)   # sweep: 65 / 185 us per assignment pass
        self._coef = {}                                  # T -> (cc, ss, cs, sc, J)   (RoPE window average)
        self._seed = {}                                  # (T, layer) -> device seeds
        self._stash = {}                                 # layer -> (M, M_kv) of the row being prefilled
        self._ws = {}                                    # T -> k-means work buffers
        self.timer = None

    def _tick(self, name):
        if self.timer is not None:
            self.timer(name)

    # ---------------------------------------------------------------- metric (bitwise = methods_eager)
    def _coefs(self, p, w):
        key = (p, w)
        if key not in self._coef:                        # methods_eager -> StaticTopKMasker._rot_avg, coefficient part
            inv, scl = self._inv_freq, self._rope_scaling
            dev = self.dev
            th = torch.cat([inv, inv]).double()
            a = th.unsqueeze(1)
            b = th.unsqueeze(0)

            def mean_exp(alpha):
                if w <= 64:
                    t = torch.arange(p, p + w, device=dev, dtype=torch.float64)
                    ph = alpha.unsqueeze(-1) * t
                    return torch.complex(ph.cos().mean(-1), ph.sin().mean(-1))
                z = torch.polar(torch.ones_like(alpha), alpha)
                den = 1 - z
                small = den.abs() < 1e-9
                s = torch.where(small, torch.full_like(z, float(w)),
                                (1 - z ** w) / torch.where(small, torch.ones_like(den), den))
                return s * torch.polar(torch.ones_like(alpha), alpha * p) / w

            Em = mean_exp(a - b)
            Ep = mean_exp(a + b)
            cc = 0.5 * (Em.real + Ep.real)
            ss = 0.5 * (Em.real - Ep.real)
            cs = 0.5 * (Ep.imag - Em.imag)
            sc = 0.5 * (Ep.imag + Em.imag)
            cc, ss, cs, sc = [x.float() * (scl * scl) for x in (cc, ss, cs, sc)]
            h = D // 2
            J = torch.zeros(D, D, device=dev)
            J[:h, h:] = -torch.eye(h, device=dev)
            J[h:, :h] = torch.eye(h, device=dev)
            self._coef = {key: (cc, ss, cs, sc, J)}      # one T at a time
        return self._coef[key]

    def _metric(self, l, b, q, T):
        cc, ss, cs, sc, J = self._coefs(T, self.win)
        M = self.Mpre[l].float()
        Mst = cc * M + cs * (M @ J.T) + sc * (J @ M) + ss * (J @ M @ J.T)
        Qn = C.qlen[b]
        dev = self.dev
        P = min(self.P, max(Qn - 2, 1))
        pi = torch.linspace(0, max(Qn - 3, 0), P, device=dev).long().unique()
        qsel = q[0, :, T - Qn:, :].index_select(1, pi).float()
        cos, sin = self.STK._rope_cs(self._inv_freq, self._rope_scaling, (Qn - pi).to(dev))
        qrot = self.STK._rope_apply(qsel, cos, sin)
        Mq = torch.matmul(qrot.transpose(-1, -2), qrot) / qrot.shape[1]
        M = 0.5 * self._nt(Mst) + 0.5 * self._nt(Mq)
        return M, M.view(HKV, G, D, D).mean(1)

    def _seeds(self, T, l):
        key = (T, l)
        if key not in self._seed:
            if len(self._seed) >= 4 * bl.NL:
                self._seed.clear()
            g = torch.Generator(device="cpu").manual_seed(31 * (l + 1))
            self._seed[key] = torch.randperm(T, generator=g)[:self.C].to(self.dev)
        return self._seed[key]

    def _work(self, T):
        if T not in self._ws:
            NTL = triton.cdiv(T, self.BT_KM * self.NSUB_KM)
            dev = self.dev
            self._ws = {T: dict(
                NTL=NTL,
                pacc=torch.empty(HQ, NTL, self.CNP, self.rc, device=dev),
                pcnt=torch.empty(HQ, NTL, self.CNP, device=dev),
                cb=torch.empty(HQ, self.C, self.rc, device=dev, dtype=torch.bfloat16),
                half=torch.empty(HQ, self.C, device=dev, dtype=torch.bfloat16))}
        return self._ws[T]

    # ---------------------------------------------------------------- build
    def build(self, l, b, q, T):
        """per layer: the metric (needs this layer's queries); at the last layer: everything else for all layers"""
        if l == 0:
            self._stash = {}
            self._tick(None)
        self._stash[l] = self._metric(l, b, q, T)
        if l == bl.NL - 1:
            self._tick("a metric M (all layers)")
            Ls = sorted(self._stash)
            stack = torch.cat([torch.cat([self._stash[i][1], self._stash[i][0]]) for i in Ls])   # per layer: 8 + 32
            ev, vec = torch.linalg.eigh(stack)
            self._tick("b eigh (one call, all layers)")
            per = HKV + HQ
            for j, i in enumerate(Ls):
                self._finish(i, b, T, ev[j * per:(j + 1) * per], vec[j * per:(j + 1) * per])
            self._stash = {}

    def _finish(self, l, b, T, ev, vec):
        dev, Cn, rc = self.dev, self.C, self.rc
        k = C.K[l][b, :, :T]
        # QGS: U (top-r eigenvectors of the KV-head metric), int8 codes with a per-coordinate scale
        U = vec[:HKV, :, -self.rq:].contiguous()
        zc = torch.matmul(k.float(), U)
        scale = (zc.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-12)
        self.codes8[l][b, :, :T] = (zc / scale).round().clamp(-127, 127).to(torch.int8)
        self.U[l][b], self.scale[l][b] = U, scale
        self._tick("c QGS codes")
        # QGC projection, seeds, first centroids (methods_eager)
        evq, Uq = ev[HKV:], vec[HKV:]
        W = (Uq[..., -rc:] * evq[..., -rc:].clamp(min=0).sqrt().unsqueeze(-2)).to(torch.bfloat16)
        krep = k.repeat_interleave(G, 0)
        zP = torch.matmul(krep, W).float()
        colmean = zP.mean(1, keepdim=True)
        zP = zP - colmean
        init = self._seeds(T, l)
        cent0 = zP.index_select(1, init)
        zb = zP.to(torch.bfloat16)
        ws = self._work(T)
        cb, half = ws["cb"], ws["half"]
        cb.copy_(cent0.to(torch.bfloat16))
        half.copy_((0.5 * (cent0 * cent0).sum(-1)).to(torch.bfloat16))
        self.W[l][b], self.colmean[l][b] = W, colmean
        self._tick("e QGC projection + seeds")
        # 4 Lloyd iterations
        asg = self.asg[l][b]
        cent = self.cent[l][b]
        nb = self.nb[l][b]
        NTL = ws["NTL"]
        for _ in range(self.iters):
            _k_km_assign[(HQ, NTL)](zb, zP, cb, half, asg, ws["pacc"], ws["pcnt"], T, Cn, NTL, self.Lmax,
                                    RC=rc, CNP=self.CNP, BLOCK_T=self.BT_KM, NSUB=self.NSUB_KM,
                                    num_warps=4)
            _k_km_update[(HQ, triton.cdiv(Cn, self.BC_KM))](ws["pacc"], ws["pcnt"], zP, init, cent, cb, half, nb,
                                                            T, Cn, NTL, RC=rc, CNP=self.CNP, BC=self.BC_KM,
                                                            num_warps=2)
        self._tick("f k-means (4 Lloyd iterations)")
        # CSR member lists (ours_fused _build_csr) and the cluster sums
        self.mem[l][b, :, :T] = torch.argsort(asg[:, :T], dim=-1, stable=True).to(torch.int32)
        cnt = nb.round().to(torch.int32)
        self.cstart[l][b] = (cnt.cumsum(-1) - cnt).to(torch.int32)
        self.clen[l][b] = cnt
        self._tick("j CSR member lists")
        _k_cluster_sums[(HQ, Cn)](C.K[l][b], C.V[l][b], self.mem[l][b], self.cstart[l][b], self.clen[l][b],
                                  self.Sk[l][b], self.Sk2[l][b], self.Sv[l][b], self.Lmax, Cn,
                                  G=G, D=D, BLOCK=64, VAR=self.VAR, num_warps=4)
        self._tick("g cluster sums (n_b, sum k, sum k^2, sum v)")
        self.n0 = T
        self.BM[l] = max(self.BM_MIN, min(self.BM_MAX, triton.next_power_of_2(-(-T // Cn))))
        self.reccnt[l][b].zero_()


G5.METHODS["ours3"] = OursG3


# ============================================================ self test
STATE = ("U", "scale", "codes8", "W", "colmean", "cent", "asg", "nb", "Sk", "Sk2", "Sv", "mem", "cstart", "clen")


def _snap(m, T):
    s = {nm: getattr(m, nm)[0].clone() for nm in STATE}
    s["asg"], s["mem"] = s["asg"][..., :T], s["mem"][..., :T]
    s["codes8"] = s["codes8"][:, :, :T]
    return s


def selftest(dev):
    print("== OursG3 fast build vs methods_eager/v66 build (two reference builds give the reference's own spread)")
    for rho in (0.01, 0.03):
        for B in (1, 8):
            a, Lmax, qp = F._synthetic(B, rho, dev, extra=160)
            T = qp.shape[2]
            refs, new = [], None
            for rep in range(2):
                m = G6.OursG2(a, B, Lmax, dev)
                for b in range(B):
                    m.build(0, b, qp, T)
                refs.append((m, _snap(m, T)))
            new = OursG3(a, B, Lmax, dev)
            for b in range(B):
                new.build(0, b, qp, T)
            sn = _snap(new, T)
            s1, s2 = refs[0][1], refs[1][1]
            ident = [nm for nm in ("U", "scale", "codes8", "W", "colmean") if torch.equal(s1[nm], sn[nm])]
            agree_rr = (s1["asg"] == s2["asg"]).float().mean().item()
            agree_rn = (s1["asg"] == sn["asg"]).float().mean().item()

            def rel(x, y):
                return ((x - y).norm() / y.norm().clamp(min=1e-30)).item()
            print(f"  rho={rho} B={B} C={new.C}: bitwise identical {ident}; assignments equal "
                  f"ref-vs-ref {100 * agree_rr:.3f}% / ref-vs-fast {100 * agree_rn:.3f}%; centroid rel diff "
                  f"ref-ref {rel(s2['cent'], s1['cent']):.1e} / ref-fast {rel(sn['cent'], s1['cent']):.1e}")
            # the same Lloyd iterations with fp64 accumulation (methods_eager code otherwise): which build is closer?
            mis_r = mis_f = tot = 0
            for b in range(B):
                krep = C.K[0][b, :, :T].repeat_interleave(G, 0)
                zP = torch.matmul(krep, sn["W"][b]).float() - sn["colmean"][b]
                init = new._seeds(T, 0)
                cent = zP.index_select(1, init).clone()
                zb = zP.to(torch.bfloat16)
                asg = torch.empty(HQ, T, device=dev, dtype=torch.long)
                for _ in range(new.iters):
                    half = (0.5 * (cent * cent).sum(-1)).to(torch.bfloat16).unsqueeze(1)
                    cbt = cent.to(torch.bfloat16).transpose(1, 2)
                    for h0 in range(0, HQ, 8):
                        asg[h0:h0 + 8] = (torch.matmul(zb[h0:h0 + 8], cbt[h0:h0 + 8]) - half[h0:h0 + 8]).argmax(-1)
                    acc = torch.zeros(HQ, new.C, new.rc, device=dev, dtype=torch.float64).scatter_add_(
                        1, asg.unsqueeze(-1).expand(-1, -1, new.rc), zP.double()).float()
                    cnt = torch.zeros(HQ, new.C, device=dev).scatter_add_(1, asg, torch.ones_like(zP[..., 0]))
                    cent = torch.where(cnt.unsqueeze(-1) > 0, acc / cnt.clamp(min=1).unsqueeze(-1), zP.index_select(1, init))
                mis_r += int((s1["asg"][b].long() != asg).sum())
                mis_f += int((sn["asg"][b].long() != asg).sum())
                tot += asg.numel()
            print(f"    vs the fp64-accumulated Lloyd iterations: assignments differ ref {100 * mis_r / tot:.3f}% / "
                  f"fast {100 * mis_f / tot:.3f}%")
            # cluster sums on the SAME assignment: copy the reference clusters into a fast instance and re-sum
            chk = OursG3(a, B, Lmax, dev)
            for nm in ("asg", "nb"):
                getattr(chk, nm)[0].copy_(getattr(refs[0][0], nm)[0])
            for b in range(B):
                chk.mem[0][b, :, :T] = torch.argsort(chk.asg[0][b, :, :T], dim=-1, stable=True).to(torch.int32)
                cnt = chk.nb[0][b].round().to(torch.int32)
                chk.cstart[0][b] = (cnt.cumsum(-1) - cnt).to(torch.int32)
                chk.clen[0][b] = cnt
                _k_cluster_sums[(HQ, chk.C)](C.K[0][b], C.V[0][b], chk.mem[0][b], chk.cstart[0][b], chk.clen[0][b],
                                             chk.Sk[0][b], chk.Sk2[0][b], chk.Sv[0][b], Lmax, chk.C,
                                             G=G, D=D, BLOCK=64, VAR=True, num_warps=4)
            print(f"    sums on the reference assignment: rel diff Sk {rel(chk.Sk[0], s1['Sk']):.1e} "
                  f"Sk2 {rel(chk.Sk2[0], s1['Sk2']):.1e} Sv {rel(chk.Sv[0], s1['Sv']):.1e}; "
                  f"CSR equal {torch.equal(chk.mem[0][..., :T], s1['mem'])}")
            # decode outputs from the three states (same queries)
            C.n0_dev = torch.tensor([T], device=dev, dtype=torch.int32)
            n, worst_rr, worst_rn = T, 0.0, 0.0
            ms = [refs[0][0], refs[1][0], new]
            for it in range(6):
                n += 1
                C.n_dev = torch.tensor([n], device=dev, dtype=torch.int32)
                q = torch.randn(B, HQ, 1, D, device=dev, dtype=torch.bfloat16)
                o = [mm.decode(0, q, n).float().view(-1, D) for mm in ms]
                r = lambda x, y: ((x - y).norm(dim=-1) / y.norm(dim=-1)).median().item()   # noqa: E731
                worst_rr, worst_rn = max(worst_rr, r(o[1], o[0])), max(worst_rn, r(o[2], o[0]))
            print(f"    decode output median rel diff: ref-vs-ref {worst_rr:.1e} / ref-vs-fast {worst_rn:.1e}")
            del refs, new, chk, ms
            torch.cuda.empty_cache()


# ============================================================ build stage times
def profile_build(dev, out):
    res = {}
    for rho in (0.01, 0.03):
        for tag in ("ours2", "ours3"):
            a, Lmax, qp = F._synthetic(1, rho, dev, extra=160)
            T = qp.shape[2]
            bl.NL = 32                                   # a whole request: 32 layers on the same synthetic cache
            C.K, C.V = C.K * 32, C.V * 32
            m = (G6.OursG2 if tag == "ours2" else OursG3)(a, 1, Lmax, dev)
            tot, stages = [], {}
            for rep in range(4):
                if tag == "ours3":
                    st = {"prev": None}

                    def tick(name, st=st):
                        torch.cuda.synchronize(dev)
                        now = time.perf_counter()
                        if name is not None and rep > 0:
                            stages[name] = stages.get(name, 0.0) + (now - st["prev"]) * 1e3 / 3
                        st["prev"] = now
                    m.timer = tick
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                for l in range(32):
                    m.build(l, 0, qp, T)
                torch.cuda.synchronize(dev)
                if rep > 0:
                    tot.append((time.perf_counter() - t0) * 1e3)
            m.timer = None
            ms = sorted(tot)[len(tot) // 2]
            res[f"{tag} rho={rho}"] = {"request_ms": ms, "stages_ms": stages, "C": m.C}
            print(f"== {tag} rho={rho} C={m.C}: build {ms:.1f} ms/request (32 layers)")
            for k_, v_ in stages.items():
                print(f"   {k_:44s} {v_:7.2f} ms/request")
            del m
            C.K = C.V = None
            torch.cuda.empty_cache()
    if out:
        json.dump(res, open(out, "w"), indent=2)


def acc_main():
    """eval_accuracy driver with ours3 registered"""
    import eval_accuracy as A

    class OursG3Acc(OursG3):
        graph_key = A.OursQ.graph_key
    A.METHODS["ours3"] = OursG3Acc
    A.main()


if __name__ == "__main__":
    if "--selftest" in sys.argv or "--profile-build" in sys.argv:
        dev = torch.device(sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0")
        torch.cuda.set_device(dev)
        if "--selftest" in sys.argv:
            selftest(dev)
        else:
            profile_build(dev, sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "")
    elif "--acc" in sys.argv:
        sys.argv.remove("--acc")
        acc_main()
    else:
        G5.main()
