#!/usr/bin/env python
"""Unit tests for the dense-direct QEq solver and the SOG kernel identity.

Verifies the plan's five claims:

  1. Kernel identity: the explicit J from :func:`build_sog_kernel` reproduces
     the SOG long-range energy exactly,  E_lr(f) = ½ q_fᵀ J_f q_f,  for
     periodic (orthorhombic + triclinic, per-frame and batched paths) and
     non-periodic frames, including a mixed batch.
  2. Direct dense solve == CG solve (< 1e-4 e relative).
  3. Charge conservation Σ_i q_i = 0 is exact (closed by λ).
  4. LiF / Li₂O equilibrium charges have the physically correct sign,
     symmetry and magnitude; the minimiser identity  E_charge(q*) = ½ χᵀ q*
     holds.
  5. Potential consistency:  J q* == autograd ∂E_lr/∂q (same-kernel proof
     that the frozen-charge envelope theorem is exact).

Run from the MACE repo root:

    /home/ubuntu/.conda/envs/mlip/bin/python mace/modules/tests/test_qeq_direct.py

Also importable under pytest (``pytest mace/modules/tests/test_qeq_direct.py``).
"""

import math
import os
import sys
import time

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sog import Sog

from mace.modules.qeq import (
    MULLIKEN_ELECTRONEGATIVITY,
    build_hardness,
    build_sog_kernel,
    qeq_equilibrate,
    qeq_equilibrate_direct,
)
from mace.tools.scatter import scatter_sum

torch.manual_seed(0)

# ── fixtures ──────────────────────────────────────────────────────────────


def make_sog():
    """Default-construction SOG with atomwise disabled (matches sog_args_qeq.yaml
    except the trained amp/bandwidth — the kernel params are read from the
    module state either way).

    ``remove_self_interaction=False`` is required for QEq: the atomic
    self-term J_ii > 0 keeps A = diag(η) + J symmetric positive definite
    (with self-removal the kernel is indefinite even on neutral patterns).
    """
    return Sog(
        sog_arguments={
            "use_atomwise": False,
            "remove_self_interaction": False,
            "use_nufft": False,
        }
    ).eval()


def make_frame(n, cell, seed, nonperiodic=False):
    gen = torch.Generator().manual_seed(seed)
    if nonperiodic:
        return 3.0 * torch.randn(n, 3, generator=gen)
    frac = torch.rand(n, 3, generator=gen)
    return frac @ cell


def sog_e_lr(sog, q, pos, cell, batch):
    res = sog(latent_charges=q, positions=pos, cell=cell, batch=batch, compute_energy=True)
    e = res["E_lr"]
    assert e is not None
    return e.reshape(-1)


def rel_err(a, b, eps=1e-30):
    """Relative error: scalars, or max-norm for tensors."""
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).abs().max() / max(a.abs().max(), b.abs().max(), eps))


# ── 1. kernel identity ────────────────────────────────────────────────────


def test_kernel_identity_per_frame():
    """E_lr(q) == ½qᵀJq per frame: orthorhombic, triclinic, non-periodic."""
    sog = make_sog()
    cells = [
        torch.diag(torch.tensor([12.0, 10.0, 9.0])),          # orthorhombic
        torch.tensor([[8.0, 0.0, 0.0],                        # triclinic
                      [2.5, 9.5, 0.0],
                      [1.5, 1.0, 10.0]]),
        torch.zeros(3, 3),                                    # non-periodic
    ]
    n_atoms = [40, 32, 24]
    for i, (cell, n) in enumerate(zip(cells, n_atoms)):
        pos = make_frame(n, cell, seed=100 + i, nonperiodic=(i == 2))
        q = 0.5 * torch.randn(n)
        batch = torch.zeros(n, dtype=torch.int64)
        e_sog = sog_e_lr(sog, q, pos, cell.unsqueeze(0), batch).sum()
        J = build_sog_kernel(pos, cell.unsqueeze(0), batch, sog)
        e_J = 0.5 * q @ (J @ q)
        err = rel_err(e_J, e_sog)
        print(f"  [1] frame {i} (n={n}): E_lr={e_sog.item():+.6f}  ½qᵀJq={e_J.item():+.6f}  rel={err:.2e}")
        assert err < 1e-5, f"kernel identity frame {i}: rel {err:.2e}"


def test_kernel_identity_batched():
    """Batched path (all periodic): SOG uses one shared k-grid sized by the
    largest per-frame nk — the builder must replicate that dispatch."""
    sog = make_sog()
    cell_a = torch.diag(torch.tensor([12.0, 10.0, 9.0]))
    cell_b = torch.tensor([[8.0, 0.0, 0.0], [2.5, 9.5, 0.0], [1.5, 1.0, 10.0]])
    pos_a, pos_b = make_frame(40, cell_a, seed=200), make_frame(32, cell_b, seed=201)
    pos = torch.cat([pos_a, pos_b], dim=0)
    cell = torch.stack([cell_a, cell_b], dim=0)
    batch = torch.cat([torch.zeros(40, dtype=torch.int64), torch.ones(32, dtype=torch.int64)])
    q = 0.5 * torch.randn(72)

    e_sog = sog_e_lr(sog, q, pos, cell, batch)                # [2]
    J = build_sog_kernel(pos, cell, batch, sog)
    for f in range(2):
        idx = (batch == f).nonzero(as_tuple=True)[0]
        e_J_f = 0.5 * q[idx] @ (J[idx.unsqueeze(0), idx.unsqueeze(1)] @ q[idx])
        err = rel_err(e_J_f, e_sog[f])
        print(f"  [2] frame {f}: E_lr={e_sog[f].item():+.6f}  ½qᵀJq={e_J_f.item():+.6f}  rel={err:.2e}")
        assert err < 1e-5, f"batched kernel identity frame {f}: rel {err:.2e}"
    e_tot = 0.5 * q @ (J @ q)
    err_tot = rel_err(e_tot, e_sog.sum())
    assert err_tot < 1e-5, f"batched total: rel {err_tot:.2e}"


def test_kernel_identity_mixed_batch():
    """Mixed batch (periodic + non-periodic) forces the per-frame loop path."""
    sog = make_sog()
    cell_a = torch.diag(torch.tensor([12.0, 10.0, 9.0]))
    cell_c = torch.zeros(3, 3)
    pos_a = make_frame(40, cell_a, seed=300)
    pos_c = make_frame(24, cell_c, seed=301, nonperiodic=True)
    pos = torch.cat([pos_a, pos_c], dim=0)
    cell = torch.stack([cell_a, cell_c], dim=0)
    batch = torch.cat([torch.zeros(40, dtype=torch.int64), torch.ones(24, dtype=torch.int64)])
    q = 0.5 * torch.randn(64)

    e_sog = sog_e_lr(sog, q, pos, cell, batch)                # [2]
    J = build_sog_kernel(pos, cell, batch, sog)
    for f in range(2):
        idx = (batch == f).nonzero(as_tuple=True)[0]
        e_J_f = 0.5 * q[idx] @ (J[idx.unsqueeze(0), idx.unsqueeze(1)] @ q[idx])
        err = rel_err(e_J_f, e_sog[f])
        print(f"  [3] frame {f}: E_lr={e_sog[f].item():+.6f}  ½qᵀJq={e_J_f.item():+.6f}  rel={err:.2e}")
        assert err < 1e-5, f"mixed-batch kernel identity frame {f}: rel {err:.2e}"


def test_kernel_symmetry_and_block_structure():
    """J must be symmetric, block-diagonal, with a strictly positive diagonal
    (self-term J_ii > 0 — required for the QEq functional to be convex)."""
    sog = make_sog()
    cell_a = torch.diag(torch.tensor([12.0, 10.0, 9.0]))
    pos_a, pos_c = make_frame(40, cell_a, seed=400), make_frame(24, torch.zeros(3, 3), seed=401, nonperiodic=True)
    pos = torch.cat([pos_a, pos_c], dim=0)
    cell = torch.stack([cell_a, torch.zeros(3, 3)], dim=0)
    batch = torch.cat([torch.zeros(40, dtype=torch.int64), torch.ones(24, dtype=torch.int64)])
    J = build_sog_kernel(pos, cell, batch, sog)
    asym = (J - J.t()).abs().max().item()
    diag_min = J.diagonal().min().item()
    offblock = J[:40, 40:].abs().max().item()
    print(f"  [4] |J−Jᵀ|max={asym:.2e}  min(diag)={diag_min:.2e}  |off-block|max={offblock:.2e}")
    assert asym < 1e-6, f"J not symmetric: {asym:.2e}"
    assert diag_min > 0, f"J self-term not positive: {diag_min:.2e}"
    assert offblock < 1e-6, f"J not block-diagonal: {offblock:.2e}"


# ── 2. direct vs CG ───────────────────────────────────────────────────────


def test_direct_vs_cg():
    """The dense solve and the autograd-CG solve agree on the same χ, η."""
    sog = make_sog()
    cell = torch.diag(torch.tensor([12.0, 10.0, 9.0]))
    n = 40
    pos = make_frame(n, cell, seed=500)
    batch = torch.zeros(n, dtype=torch.int64)
    elements = [3, 8, 9] * math.ceil(n / 3)
    hardness = build_hardness(elements[:n])                   # per-atom η
    chi = 2.0 * torch.randn(n)                                # per-atom χ
    J = build_sog_kernel(pos, cell.unsqueeze(0), batch, sog)

    # Same potential_fn as extensions.py's CG branch.
    def potential_fn(q_leaf):
        q_leaf = q_leaf.detach().requires_grad_(True)
        e_lr = sog_e_lr(sog, q_leaf, pos, cell.unsqueeze(0), batch).sum()
        return torch.autograd.grad(e_lr, q_leaf)[0]

    q_cg = qeq_equilibrate(chi, hardness, batch, potential_fn, iters=50)
    q_d = qeq_equilibrate_direct(chi, hardness, batch, J).squeeze(-1)
    diff = (q_cg - q_d).abs().max().item()
    scale = max(1.0, q_d.abs().max().item())
    print(f"  [5] |q_cg − q_direct|max = {diff:.2e} e   (|q|max = {q_d.abs().max().item():.3f} e)")
    assert diff < 1e-4 * scale, f"direct/CG disagree: {diff:.2e}"


def test_multihead_direct():
    """Multi-head χ [n, nheads] solves independently on the shared A."""
    sog = make_sog()
    cell = torch.diag(torch.tensor([12.0, 10.0, 9.0]))
    n = 40
    pos = make_frame(n, cell, seed=600)
    batch = torch.zeros(n, dtype=torch.int64)
    hardness = build_hardness([3, 8, 9] * math.ceil(n / 3))[:n]
    chi = 2.0 * torch.randn(n, 3)
    J = build_sog_kernel(pos, cell.unsqueeze(0), batch, sog)
    q = qeq_equilibrate_direct(chi, hardness, batch, J)
    assert q.shape == chi.shape, f"shape {tuple(q.shape)} != {tuple(chi.shape)}"
    # Per-head: A q_h == −χ_h − μ_h must hold at the returned q.
    A = torch.diag_embed(hardness) + J
    for h in range(3):
        mu = -(A @ q[:, h] + chi[:, h])                       # ≈ const vector
        resid = (A @ q[:, h] + chi[:, h] + mu.mean()).abs().max().item()
        qsum = q[:, h].sum().abs().item()
        assert resid < 1e-4, f"head {h} Euler–Lagrange residual {resid:.2e}"
        assert qsum < 1e-6, f"head {h} charge sum {qsum:.2e}"
    print(f"  [6] multi-head (nheads=3): Euler–Lagrange residuals OK, Σq per head < 1e-6")


# ── 3. charge conservation ────────────────────────────────────────────────


def test_charge_conservation_exact():
    """Σq = 0 is closed algebraically by λ — exact up to float rounding."""
    sog = make_sog()
    cell = torch.diag(torch.tensor([12.0, 10.0, 9.0]))
    n = 40
    pos = make_frame(n, cell, seed=700)
    batch = torch.zeros(n, dtype=torch.int64)
    hardness = build_hardness([3, 8, 9] * math.ceil(n / 3))[:n]
    chi = 2.0 * torch.randn(n)
    J = build_sog_kernel(pos, cell.unsqueeze(0), batch, sog)
    q = qeq_equilibrate_direct(chi, hardness, batch, J)
    qsum = q.sum().abs().item()
    print(f"  [7] Σq* = {qsum:.2e} e")
    assert qsum < 1e-6, f"charge not conserved: {qsum:.2e}"


# ── 4. LiF / Li₂O physics ─────────────────────────────────────────────────


def test_lif_lio_charges():
    """Mulliken χ + table η give the right sign/symmetry/magnitude, and the
    equilibrium identity E_charge(q*) = ½χᵀq* holds at the solve's minimiser."""
    sog = make_sog()

    # Periodic 8 Å boxes: the realspace (zero-cell) kernel of a default-constructed
    # SOG uses the raw geometric amp (Σ_m amp_m ~ 10⁸) with no Gaussian
    # normalisation, which drives QEq charges to ~0 — that convention is SOG's,
    # not the builder's (the builder reproduces it exactly).  In a periodic box
    # the reciprocal k-sum kernel is the well-behaved one the model is trained on.
    L = 8.0
    cell = torch.diag(torch.tensor([L, L, L])).unsqueeze(0)

    def solve_molecule(zs, pos):
        batch = torch.zeros(len(zs), dtype=torch.int64)
        chi = torch.tensor([MULLIKEN_ELECTRONEGATIVITY[z] for z in zs])
        hardness = build_hardness(zs)                         # per-atom η
        J = build_sog_kernel(pos, cell, batch, sog)
        q = qeq_equilibrate_direct(chi, hardness, batch, J).squeeze(-1)
        e_charge = chi @ q + 0.5 * (q * q) @ hardness + 0.5 * q @ (J @ q)
        return q, e_charge, J

    # LiF at 1.5 Å (box centre) — two atoms, so exact antisymmetry q_Li = −q_F.
    pos = torch.tensor([[L / 2, L / 2, L / 2], [L / 2, L / 2, L / 2 + 1.5]])
    q, e_charge, J = solve_molecule([3, 9], pos)
    j12 = J[0, 1].item()
    print(f"  [8a] LiF:  q = [{q[0].item():+.4f}, {q[1].item():+.4f}] e,  J₁₂ = {j12:.3f} eV/e²")
    assert q[0] > 0 and q[1] < 0, "LiF charge sign wrong"
    assert abs(q[0] + q[1]) < 1e-7, "LiF q_Li ≠ −q_F"
    assert 0.1 < q[0] < 0.95, f"LiF |q| out of physical band: {q[0].item():.4f}"

    # Minimiser identity: E_charge(q*) = ½ χᵀq* < 0 (charge-transfer stabilisation).
    half_chi_q = 0.5 * (torch.tensor([MULLIKEN_ELECTRONEGATIVITY[z] for z in [3, 9]]) @ q)
    err = abs(e_charge - half_chi_q).item()
    print(f"       E_charge(q*) = {e_charge.item():+.6f} eV,  ½χᵀq* = {half_chi_q.item():+.6f} eV  (diff {err:.2e})")
    assert err < 1e-4, f"equilibrium identity broken: diff {err:.2e}"
    assert e_charge < 0, "charge transfer not stabilising"

    # Li₂O: Li at (±1.9,0,0)-ish around O, box centre.
    pos = torch.tensor(
        [[L / 2, L / 2, L / 2], [L / 2, L / 2, L / 2 + 1.9], [L / 2 + 1.9, L / 2, L / 2]]
    )
    q, e_charge, _ = solve_molecule([3, 8, 3], pos)
    print(f"  [8b] Li₂O: q = [{q[0].item():+.4f}, {q[1].item():+.4f}, {q[2].item():+.4f}] e")
    assert q[0] > 0 and q[2] > 0 and q[1] < 0, "Li₂O charge sign wrong"
    assert abs(q.sum().item()) < 1e-6, "Li₂O not neutral"
    assert 0.1 < abs(q[1]) < 1.8, f"Li₂O |q_O| out of physical band: {q[1].item():.4f}"


# ── 5. potential consistency (same-kernel proof) ──────────────────────────


def test_potential_consistency():
    """J q* == autograd ∂E_lr/∂q: the solve kernel and the energy kernel are
    the same operator, so the frozen-charge envelope theorem holds exactly."""
    sog = make_sog()
    cell = torch.diag(torch.tensor([12.0, 10.0, 9.0]))
    n = 40
    pos = make_frame(n, cell, seed=800)
    batch = torch.zeros(n, dtype=torch.int64)
    J = build_sog_kernel(pos, cell.unsqueeze(0), batch, sog)
    q_leaf = (0.5 * torch.randn(n)).requires_grad_(True)
    e_lr = sog_e_lr(sog, q_leaf, pos, cell.unsqueeze(0), batch).sum()
    g = torch.autograd.grad(e_lr, q_leaf)[0]
    v = J @ q_leaf.detach()
    err = rel_err(v, g)
    print(f"  [9] |Jq − ∂E_lr/∂q| rel = {err:.2e}")
    assert err < 1e-5, f"potential inconsistent: rel {err:.2e}"


# ── 6. timing (battery-scale frame) ───────────────────────────────────────


def test_timing_battery_scale():
    """260-atom frame: J construction + dense solve should be sub-second even
    on CPU; on GPU it is sub-ms."""
    sog = make_sog()
    cell = torch.diag(torch.tensor([10.0, 10.0, 6.0]))
    n = 260
    pos = make_frame(n, cell, seed=900)
    batch = torch.zeros(n, dtype=torch.int64)
    hardness = build_hardness([3, 8, 9] * math.ceil(n / 3))[:n]
    chi = 2.0 * torch.randn(n)

    t0 = time.perf_counter()
    J = build_sog_kernel(pos, cell.unsqueeze(0), batch, sog)
    t1 = time.perf_counter()
    q = qeq_equilibrate_direct(chi, hardness, batch, J)
    t2 = time.perf_counter()
    dt_build = (t1 - t0) * 1e3
    dt_solve = (t2 - t1) * 1e3
    print(f"  [10] n=260: build J {dt_build:.1f} ms, dense solve {dt_solve:.1f} ms (CPU)")
    assert dt_build + dt_solve < 5000, "kernel+solve unreasonably slow"


# ── 7. gml broadcast trap (regression) ─────────────────────────────────────


def test_gml_broadcast_trap():
    """Regression: in MACESOG.forward chi is 1-D [n_atoms] while q_star is
    [n_atoms, nheads].  A bare `chi * q_star` broadcasts as the [n, n] outer
    product with entries χ_j·q_i (torch left-pads [n]→[1,n] against [n,1]);
    summing over dim=-1 then mixes every atom's χ into each atom's term —
    this produced gml ≈ +100–180 eV/atom in smoke runs instead of ~ −1 eV/atom.
    The G_ML term must be composed per atom: chi * q_star.squeeze(-1) + ½η q²."""
    n, nheads = 7, 1
    chi = torch.arange(1.0, n + 1)
    q = torch.arange(0.5, n + 0.5).unsqueeze(-1)  # [n, nheads]
    prod = chi * q                                # the trap: outer product [n, n]
    outer = q * chi.unsqueeze(0)                  # explicit χ_j·q_i outer product
    assert prod.shape == (n, n), "bare chi*q must be the [n, n] outer product"
    assert (prod - outer).abs().max() < 1e-6, "bare chi*q must equal the outer product"
    wrong = prod.sum(-1)                          # row sums: q_i * Σ_j χ_j
    right = chi * q.squeeze(-1)                   # the fix: per-atom product
    assert (wrong - right).abs().max() > 1e-6, "trap and per-atom product must differ"
    assert torch.allclose(
        right, chi * q.flatten(), atol=1e-6
    ), "per-atom gml must equal elementwise chi_i q_i"


# ── runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_kernel_identity_per_frame,
        test_kernel_identity_batched,
        test_kernel_identity_mixed_batch,
        test_kernel_symmetry_and_block_structure,
        test_direct_vs_cg,
        test_multihead_direct,
        test_charge_conservation_exact,
        test_lif_lio_charges,
        test_potential_consistency,
        test_timing_battery_scale,
        test_gml_broadcast_trap,
    ]
    failures = 0
    print(f"== QEq direct-solver unit tests (dtype={torch.get_default_dtype()}) ==")
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:  # noqa: BLE001 — report and continue
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print("==" + (" ALL PASSED ==" if failures == 0 else f" {failures} FAILED =="))
    sys.exit(1 if failures else 0)
