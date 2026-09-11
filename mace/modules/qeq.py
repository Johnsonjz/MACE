"""QEq charge equilibration for MACESOG.

Replaces the direct charge readout ``q = MLP(node_feats)`` with a physically
motivated charge equilibration.  This is the Rappé–Goddard QEq specialised to
the *one-body quadratic* energy functional of the review by Baldwin, Batatia,
Vondrák, Margraf & Csányi (2026, "Design Space of Self-Consistent Electrostatic
Machine-Learned Interatomic Potentials"):

    E = E_local({z, r}) + Σ_i [ χ_i q_i + ½ η_i q_i² ] + ½ Σ_ij q_i J_ij q_j

where
  * χ_i  — *electronegativity*, learned from the local environment (a MACE
           readout), drives charge transfer linearly,
  * η_i  — *chemical hardness*, FIXED per element η_z = (IP − EA)/2
           (Parr–Pearson), gives the quadratic penalty that bounds charges and
           makes the functional strictly convex,
  * ½ Σ_ij q_i J_ij q_j — the long-range Hartree/Coulomb energy, evaluated by the
           SOG Gaussian (reciprocal-space Ewald).

The equilibrium charges q* minimise E subject to per-structure charge
conservation Σ_i q_i = Q_g.  The Euler–Lagrange condition is Sanderson's
electronegativity equalisation

    ∂E/∂q_i = χ_i + η_i q_i + v_i − μ = 0,   v_i := ∂E_LR/∂q_i = Σ_j J_ij q_j,

a symmetric positive-definite linear system A q = −χ − μ·1 (A = diag(η) + J).
SPD requires the kernel to keep its self-interaction: J_ii > 0 (the atomic
self-Coulomb, standard in Rappé–Goddard QEq).  With SOG's
``remove_self_interaction`` the kernel trace vanishes and A becomes indefinite
even on neutral charge patterns — the one-body quadratic is then unbounded
below and charges explode.  The SOG config for QEq therefore sets
``remove_self_interaction: false``.

The electrostatic potential v = ∂E_LR/∂q is obtained by autograd from the SOG
Gaussian energy, so the solve and the energy share *exactly* the same Coulomb
kernel (no separate real-space J matrix is ever constructed).

Consistency of forces follows from the envelope theorem: at the minimiser
∂E/∂q|_q* = 0, hence ∂E/∂r = ∂E/∂r|_q*.  The solver therefore returns q*
detached ("frozen charge"), which is exact at SCF convergence.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from mace.tools.scatter import scatter_sum


# ── Element chemical hardness η = (IP − EA)/2  [eV/e²] ────────────────
# First ionisation potential (IP) and electron affinity (EA) in eV.  Missing
# elements fall back to HARDNESS_DEFAULT (a "typical" non-metal hardness).
ELEMENT_HARDNESS: Dict[int, float] = {
    1: 6.42,    # H
    3: 2.39,    # Li
    4: 2.38,    # Be
    5: 4.01,    # B
    6: 5.00,    # C
    7: 7.23,    # N
    8: 6.08,    # O
    9: 7.01,    # F
    11: 2.30,   # Na
    12: 3.82,   # Mg
    13: 2.78,   # Al
    14: 3.38,   # Si
    15: 4.87,   # P
    16: 4.14,   # S
    17: 4.68,   # Cl
    19: 1.92,   # K
    20: 3.04,   # Ca
    21: 3.17,   # Sc
    22: 3.38,   # Ti
    23: 3.35,   # V
    24: 3.32,   # Cr
    25: 3.33,   # Mn
    26: 3.87,   # Fe
    27: 3.79,   # Co
    28: 3.69,   # Ni
    29: 3.24,   # Cu
    30: 4.70,   # Zn
    31: 3.02,   # Ga
    32: 3.33,   # Ge
    33: 3.73,   # As
    34: 4.72,   # Se
    35: 3.75,   # Br
    37: 1.71,   # Rb
    38: 2.83,   # Sr
    39: 3.26,   # Y
    40: 3.29,   # Zr
    41: 3.45,   # Nb
    42: 3.42,   # Mo
    44: 3.49,   # Ru
    45: 3.61,   # Rh
    46: 3.46,   # Pd
    47: 3.15,   # Ag
    48: 4.45,   # Cd
    49: 2.83,   # In
    50: 3.12,   # Sn
    51: 3.80,   # Sb
    52: 4.21,   # Te
    53: 3.70,   # I
    55: 1.62,   # Cs
    56: 2.78,   # Ba
    57: 2.56,   # La
    72: 3.33,   # Hf
    73: 3.83,   # Ta
    74: 3.66,   # W
    75: 3.77,   # Re
    76: 3.63,   # Os
    77: 3.60,   # Ir
    78: 3.83,   # Pt
    79: 3.36,   # Au
    80: 4.56,   # Hg
    81: 2.86,   # Tl
    82: 3.44,   # Pb
    83: 3.32,   # Bi
}

# Mulliken electronegativity χ_M = (IP + EA)/2  [eV].  Optional initialisation
# target for the learned χ readout (mean-centred so initial charges ≈ neutral).
MULLIKEN_ELECTRONEGATIVITY: Dict[int, float] = {
    1: 7.18, 3: 3.01, 4: 2.32, 5: 4.29, 6: 6.26, 7: 7.30, 8: 7.54, 9: 10.41,
    11: 2.84, 12: 3.82, 13: 3.22, 14: 4.77, 15: 5.62, 16: 6.22, 17: 8.29,
    19: 2.42, 20: 3.07, 21: 3.21, 22: 3.46, 23: 3.43, 24: 3.34, 25: 3.33,
    26: 4.03, 27: 3.87, 28: 3.80, 29: 3.68, 30: 4.70, 31: 3.02, 32: 4.57,
    33: 4.73, 34: 5.89, 35: 5.71, 37: 2.41, 38: 2.88, 39: 3.29, 40: 3.31,
    41: 3.70, 42: 3.52, 44: 3.53, 45: 3.62, 46: 3.90, 47: 3.68, 48: 4.45,
    49: 2.90, 50: 4.23, 51: 4.84, 52: 5.20, 53: 5.71, 55: 2.33, 56: 2.78,
    57: 2.86, 72: 3.40, 73: 3.89, 74: 3.74, 75: 3.83, 76: 3.70, 77: 3.67,
    78: 4.33, 79: 4.02, 80: 4.56, 81: 2.91, 82: 3.73, 83: 3.44,
}

HARDNESS_DEFAULT = 5.0  # eV/e² fallback for missing elements


def build_hardness(
    atomic_numbers: List[int],
    hardness_default: float = HARDNESS_DEFAULT,
    hardness_override: Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    """Return a per-element hardness vector η_z = (IP − EA)/2, ordered by
    ``atomic_numbers``.  ``hardness_override`` maps Z → η and takes precedence."""
    vals = []
    for z in atomic_numbers:
        z = int(z)
        if hardness_override is not None and z in hardness_override:
            vals.append(float(hardness_override[z]))
        else:
            vals.append(ELEMENT_HARDNESS.get(z, float(hardness_default)))
    return torch.tensor(vals, dtype=torch.get_default_dtype())


class ElementHardness(nn.Module):
    """Fixed per-element chemical hardness η_z = (IP − EA)/2.

    Non-trainable (the "fixed element J" choice).  Supplies the quadratic
    penalty ½η_i q_i² that keeps charges bounded and the QEq functional convex.
    """

    def __init__(self, hardness: torch.Tensor):
        super().__init__()
        if torch.any(hardness <= 0.0):
            raise ValueError("Element hardness must be strictly positive.")
        self.register_buffer("hardness", hardness)

    def forward(self, node_attrs: torch.Tensor) -> torch.Tensor:
        """node_attrs: one-hot [n_atoms, n_elements] → per-atom η [n_atoms]."""
        return node_attrs @ self.hardness


class PerElementChiBias(nn.Module):
    """Per-element bias added to the χ readout output.

    Initialised from Mulliken electronegativity (mean-centred over the model's
    elements) so that the initial QEq charges are physically sensible
    (electropositive elements positive, electronegative negative).  The bias is
    learnable, letting the environment-dependent part of χ refine on top.
    """

    def __init__(self, atomic_numbers: List[int]):
        super().__init__()
        z_list = [int(z) for z in atomic_numbers]
        chi = torch.tensor(
            [MULLIKEN_ELECTRONEGATIVITY.get(z, 3.0) for z in z_list],
            dtype=torch.get_default_dtype(),
        )
        self.bias = nn.Parameter(chi - chi.mean())

    def forward(self, node_attrs: torch.Tensor) -> torch.Tensor:
        return (node_attrs @ self.bias).unsqueeze(-1)


def _cg_solve(
    a_fn: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor,
    iters: int,
    rtol: float = 1e-8,
) -> torch.Tensor:
    """Conjugate gradient for A x = b (A symmetric positive-definite).

    ``a_fn`` is the matrix-free operator x → A x.  CG converges monotonically
    and, unlike damped Jacobi, does not diverge for the ill-conditioned Coulomb
    operator that arises in charge equilibration.
    """
    x = x0.clone()
    r = b - a_fn(x)
    p = r.clone()
    rsold = (r * r).sum()
    bnorm = (b * b).sum().clamp(min=1e-30)
    for _ in range(int(iters)):
        ap = a_fn(p)
        pap = (p * ap).sum()
        if pap.item() <= 0.0:  # numerically non-SPD — stop rather than blow up
            break
        alpha = rsold / pap
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = (r * r).sum()
        if (rsnew / bnorm).item() < rtol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


def qeq_equilibrate(
    chi: torch.Tensor,
    hardness: torch.Tensor,
    batch: torch.Tensor,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    iters: int = 20,
    mixing: Optional[float] = None,
    total_charge: Optional[torch.Tensor] = None,
    q_init: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve the QEq equilibrium exactly by conjugate gradient.

    The one-body quadratic energy functional

        E(q) = χᵀq + ½ qᵀ diag(η) q + E_LR(q)

    is minimised subject to per-structure charge conservation Σ_i q_i = Q_g.
    The Euler–Lagrange equation is the linear system

        A q = −χ − μ·1,   A := diag(η) + J,   J·x := ∂E_LR/∂x = v(x),

    with μ the chemical potential enforcing the constraint.  Because A is
    symmetric positive-definite (the kernel keeps its self-interaction,
    J_ii > 0), CG solves it robustly — the damped-Jacobi fixed point is
    *unstable* for the ill-conditioned Coulomb operator.  Writing u = A⁻¹(−χ)
    and w = A⁻¹(1), the constrained solution is

        q* = u − λ w,   λ = (Σ_g u − Q_g) / Σ_g w.

    Parameters
    ----------
    chi : [n_atoms] learned electronegativity.
    hardness : [n_atoms] fixed element hardness η_i > 0.
    batch : [n_atoms] graph index per atom.
    potential_fn : q [n_atoms] → v [n_atoms], where v = ∂E_LR/∂q.  The closure
        is responsible for detaching/`requires_grad_` internally; its return is
        treated as a constant (no graph is built through the iterations).
    iters : max CG iterations per solve (typical convergence in ~5–10).
    mixing : ignored (kept for API compatibility with the fixed-point solver).
    total_charge : [n_graphs] target total charge per structure (default 0).
    q_init : optional [n_atoms] starting charges (default 0).

    Returns
    -------
    q : [n_atoms] equilibrium charges.  Returned *detached* (frozen-charge /
        envelope-theorem approximation, exact at the minimiser).
    """
    del mixing  # retained for API compatibility
    device = chi.device
    dtype = chi.dtype
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1

    if total_charge is None:
        total_charge = torch.zeros(num_graphs, device=device, dtype=dtype)
    if q_init is None:
        x0 = torch.zeros_like(chi)
    else:
        x0 = q_init.to(device=device, dtype=dtype).clone()

    def a_fn(x: torch.Tensor) -> torch.Tensor:
        # A x = diag(η) x + J x = η·x + v(x)
        return hardness * x + potential_fn(x)

    u = _cg_solve(a_fn, -chi, x0, int(iters))
    w = _cg_solve(a_fn, torch.ones_like(chi), torch.zeros_like(chi), int(iters))

    u_sum = scatter_sum(u, batch, dim=0, dim_size=num_graphs)
    w_sum = scatter_sum(w, batch, dim=0, dim_size=num_graphs)
    lam = (u_sum - total_charge) / w_sum
    q = u - lam[batch] * w

    return q.detach()


@torch.no_grad()
def build_sog_kernel(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
    sog: torch.nn.Module,
) -> torch.Tensor:
    """Build the dense Coulomb kernel J (block-diagonal over frames) that
    reproduces the SOG long-range energy exactly:

        E_lr(f) = ½ q_fᵀ J_f q_f          (q_f: charges of frame f)

    The construction mirrors ``sog/src/sog/module/gaussian.py`` (direct k-sum
    path) term by term:

      * periodic frame:
          J_ij = (norm/V)·[ Σ_k kfac_k·cos(k·r_ij) + kfac_eff ]    (all i, j)
        with k on the integer grid |k|² ≤ (2π/n_dl)² (k≠0), kfac_k =
        Σ_m amp_m·e^{−½·bw²_m·|k|²}, and kfac_eff the regularised k=0 value
        evaluated at k_min = min(|b_1|,|b_2|,|b_3|) of the reciprocal lattice
        (gaussian.py:1260-1275).  With remove_self_interaction the diagonal
        is subtracted off (k-sum diag, k=0 self); the QEq config keeps the
        self terms so J_ii > 0 (SPD).  self_coeff is subtracted from the
        diagonal unconditionally when nonzero.
      * non-periodic frame (zeroed cell rows):
          J_ij = norm·Σ_m amp_m·e^{−r²_ij/(2·bw²_m)};  J_ii = 0 with
          remove_self_interaction, else J_ii = norm·Σ_m amp_m
        (realspace path, gaussian.py:822-845; no k=0 correction).

    The integer k-grid replicates SOG's own dispatch: when *all* frames are
    periodic the batched SOG path uses one shared grid sized by the largest
    per-frame nk (gaussian.py:455-459) with a per-frame |k| mask; otherwise
    the per-frame loop uses each frame's own nk.  Matching this dispatch keeps
    the solve kernel identical to the energy kernel, so the frozen-charge
    envelope theorem holds exactly.  amp/bandwidth are read from the *trained*
    SOG state, not the construction defaults.

    Parameters
    ----------
    positions : [n_atoms, 3]
    cell : [n_graphs, 3, 3] rows = lattice vectors; rows zeroed for
        non-periodic frames (same convention as MACESOG's ``cell_sog``).
    batch : [n_atoms] graph index per atom.
    sog : the ``Sog`` module whose kernel to reproduce.

    Returns
    -------
    J : [n_atoms, n_atoms] block-diagonal (one block per frame), no grad.
    """
    g = sog.gaussian
    dtype = positions.dtype
    device = positions.device
    amp = g.amp.detach().to(dtype=dtype, device=device)        # [M]
    bw2 = g.bandwidth.detach().to(dtype=dtype, device=device)  # [M] = variances
    norm = float(g.norm_factor)
    remove_self = bool(g.remove_self_interaction)
    self_coeff = float(getattr(g, "self_coeff", 0.0))
    two_pi = 2.0 * math.pi

    # Direct k-sum n_dl (gaussian.py:_resolve_direct_n_dl), constant per batch.
    n_dl = getattr(g, "n_dl", None)
    if n_dl is None:
        bw_min = bw2.min().item()
        eps = 1e-5
        kmax = math.sqrt(2.0 * math.log(1.0 / eps) / max(bw_min, 1e-30))
        n_dl = two_pi / max(kmax, 1e-30)

    num_graphs = int(cell.shape[0])
    num_atoms = positions.shape[0]
    eps_vol = torch.finfo(cell.dtype).eps
    volumes = torch.abs(torch.det(cell))                       # [nf]
    periodic = volumes > eps_vol
    nk_all = (torch.norm(cell, dim=2) / n_dl).to(torch.int64).clamp(min=1)  # [nf,3]

    # Batched regularised k=0 kernel value kfac_eff (gaussian.py:1260-1271).
    a1, a2, a3 = cell[:, 0], cell[:, 1], cell[:, 2]
    vol_c = volumes.clamp(min=eps_vol).unsqueeze(1)
    b1 = two_pi * torch.linalg.cross(a2, a3, dim=1) / vol_c
    b2 = two_pi * torch.linalg.cross(a3, a1, dim=1) / vol_c
    b3 = two_pi * torch.linalg.cross(a1, a2, dim=1) / vol_c
    k_min_sq = torch.stack(
        [(b1 * b1).sum(1), (b2 * b2).sum(1), (b3 * b3).sum(1)], dim=1
    ).min(dim=1).values                                         # [nf]
    kfac_eff = (
        amp.unsqueeze(0) * torch.exp(-0.5 * bw2.unsqueeze(0) * k_min_sq.unsqueeze(1))
    ).sum(dim=1)                                                # [nf]

    k_sq_max = (two_pi / n_dl) ** 2
    # Shared grid for the batched SOG path (all frames periodic); per-frame
    # grids otherwise (per-frame loop path).
    nk_shared = nk_all.max(dim=0).values if bool(periodic.all()) else None

    J = torch.zeros(num_atoms, num_atoms, device=device, dtype=dtype)
    arange = torch.arange(num_atoms, device=device)
    for f in range(num_graphs):
        idx = (batch == f).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            continue
        r_f = positions[idx]                                   # [n, 3]
        if bool(periodic[f]):
            nk = nk_shared if nk_shared is not None else nk_all[f]
            nk = tuple(int(v) for v in nk.tolist())
            # Same integer grid + mask as gaussian.py:1030-1042 (loop) and
            # 1226-1238 (batched).  The grid is cached by SOG itself.
            k_int, zero_mask, _ = g._get_cached_kgrid_base(nk, device, dtype)
            g_cart = two_pi * (torch.linalg.inv(cell[f]) @ k_int.reshape(3, -1))
            k_sq = (g_cart * g_cart).sum(dim=0)                # [G]
            mode_mask = ~zero_mask.reshape(-1) & (k_sq <= k_sq_max)
            k_sq_sel = k_sq[mode_mask]                         # [K]
            kfac = (
                amp.unsqueeze(0)
                * torch.exp(-0.5 * bw2.unsqueeze(0) * k_sq_sel.unsqueeze(-1))
            ).sum(dim=-1)                                      # [K]
            kvec = g_cart[:, mode_mask].transpose(0, 1)        # [K, 3]
            # |S(k)|² = Σ_ij q_i q_j [cos_i cos_j + sin_i sin_j]; the sin·sin
            # cross term is even under k → −k and does NOT cancel on the full
            # symmetric grid — it must be kept.
            cos_all = torch.cos(r_f @ kvec.transpose(0, 1))    # [n, K]
            sin_all = torch.sin(r_f @ kvec.transpose(0, 1))    # [n, K]
            J_f = cos_all @ (kfac.unsqueeze(1) * cos_all.transpose(0, 1))
            J_f = J_f + sin_all @ (kfac.unsqueeze(1) * sin_all.transpose(0, 1))
            J_f = J_f / volumes[f]
            if remove_self:
                J_f = J_f - torch.diag_embed(torch.diagonal(J_f))
            J_f = J_f + (kfac_eff[f] / volumes[f])             # k=0 pair term
            if remove_self:
                J_f[arange[:n], arange[:n]] -= kfac_eff[f] / volumes[f]  # k=0 self term
            if self_coeff != 0.0:
                J_f[arange[:n], arange[:n]] -= self_coeff
            J_f = J_f * norm
        else:
            # Realspace path (gaussian.py:822-845): no k=0 correction.
            r_ij = r_f.unsqueeze(0) - r_f.unsqueeze(1)         # [n, n, 3]
            r_sq = (r_ij * r_ij).sum(dim=-1)                   # [n, n]
            J_f = (
                amp.view(1, 1, -1) * torch.exp(-0.5 * r_sq.unsqueeze(-1) / bw2.view(1, 1, -1))
            ).sum(dim=-1)
            if remove_self:
                J_f = J_f - torch.diag_embed(torch.diagonal(J_f))
            else:
                J_f[arange[:n], arange[:n]] = amp.sum()
            J_f = J_f * norm
        J[idx.unsqueeze(0), idx.unsqueeze(1)] = J_f
    return J


@torch.no_grad()
def qeq_equilibrate_direct(
    chi: torch.Tensor,
    hardness: torch.Tensor,
    batch: torch.Tensor,
    J: torch.Tensor,
    total_charge: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact QEq equilibrium by direct dense solve, one frame at a time.

    Solves the same system as :func:`qeq_equilibrate` but with the explicit
    kernel matrix instead of CG iterations:

        A q = −χ − μ·1,   A := diag(η) + J,

    via two right-hand sides u = A⁻¹(−χ), w = A⁻¹(1), then closes charge
    conservation λ = (Σ_g u − Q_g)/Σ_g w and returns q* = u − λ w.  For
    n ≤ ~260-atom frames this is sub-millisecond (O(n³) LU), replacing the
    ~20 SOG-pass CG solve.  J must come from :func:`build_sog_kernel` so the
    solve and the energy share the same Coulomb operator.

    Parameters
    ----------
    chi : [n_atoms] or [n_atoms, nheads] learned electronegativity.  A
        leading head dimension is solved independently (shared A).
    hardness : [n_atoms] fixed element hardness η_i > 0.
    batch : [n_atoms] graph index per atom.
    J : [n_atoms, n_atoms] block-diagonal SOG kernel (see build_sog_kernel).
    total_charge : [n_graphs] target total charge per structure (default 0).

    Returns
    -------
    q : same shape as chi — equilibrium charges, detached (frozen-charge /
        envelope-theorem approximation, exact at the minimiser).
    """
    device = chi.device
    dtype = chi.dtype
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
    if total_charge is None:
        total_charge = torch.zeros(num_graphs, device=device, dtype=dtype)
    if chi.dim() == 1:
        chi = chi.unsqueeze(-1)
    nheads = chi.shape[-1]

    q = torch.empty_like(chi)
    for g in range(num_graphs):
        idx = (batch == g).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            continue
        A = torch.diag_embed(hardness[idx]) + J[idx.unsqueeze(0), idx.unsqueeze(1)]
        # Two RHS per head: u = A⁻¹(−χ), w = A⁻¹(1); λ closes charge conservation.
        minus_chi = -chi[idx]                                  # [n, nheads]
        rhs = torch.cat(
            [minus_chi.unsqueeze(-1), torch.ones_like(minus_chi).unsqueeze(-1)],
            dim=-1,
        ).transpose(0, 1)  # [nheads, n, 2]
        sol = torch.linalg.solve(A.unsqueeze(0), rhs)  # LU, robust to mild indefiniteness
        u, w = sol[..., 0], sol[..., 1]  # [nheads, n] each
        lam = (u.sum(-1) - total_charge[g]) / w.sum(-1)  # [nheads]
        q[idx] = (u - lam.unsqueeze(-1) * w).transpose(0, 1)  # [n, nheads]
    return q.detach()
