import importlib.util

import pytest
import torch

from mace.modules.utils import get_outputs


def _compute_forces_from_energy(energy: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    forces, _, _, _, _ = get_outputs(
        energy=energy,
        positions=positions,
        displacement=None,
        vectors=None,
        cell=torch.eye(3, dtype=positions.dtype, device=positions.device).view(1, 3, 3),
        training=False,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
        compute_hessian=False,
        compute_edge_forces=False,
    )
    assert forces is not None
    return forces


def test_get_outputs_excludes_sog_branch_when_energy_not_passed() -> None:
    torch.manual_seed(7)
    base_positions = torch.randn(9, 3, dtype=torch.float64)
    inter_coeff = torch.randn_like(base_positions)
    sog_coeff = torch.randn_like(base_positions)

    pos_inter_only = base_positions.clone().requires_grad_(True)
    inter_e = (pos_inter_only * inter_coeff).sum().reshape(1)
    forces_inter_only = _compute_forces_from_energy(inter_e, pos_inter_only)

    pos_full = base_positions.clone().requires_grad_(True)
    full_e = (pos_full * inter_coeff).sum().reshape(1) + (
        pos_full * sog_coeff
    ).sum().reshape(1)
    forces_full = _compute_forces_from_energy(full_e, pos_full)

    # If energy excludes the SOG branch, autograd should not include it in forces.
    assert torch.allclose(forces_inter_only, -inter_coeff, atol=1e-12, rtol=1e-12)
    # If energy includes both branches, autograd includes both in forces.
    assert torch.allclose(
        forces_full,
        -(inter_coeff + sog_coeff),
        atol=1e-12,
        rtol=1e-12,
    )


_sog_spec = importlib.util.find_spec("sog")
SOG_AVAILABLE = _sog_spec is not None

if SOG_AVAILABLE:
    from sog import Sog
    from sog.module.gaussian import HAS_PYTORCH_FINUFFT
else:
    HAS_PYTORCH_FINUFFT = False


@pytest.mark.skipif(not SOG_AVAILABLE, reason="sog package is not available")
@pytest.mark.skipif(
    not HAS_PYTORCH_FINUFFT,
    reason="pytorch_finufft is not available",
)
def test_sog_explicit_matches_autograd_batched() -> None:
    torch.manual_seed(11)

    model = Sog(
        {
            "use_atomwise": False,
            "use_nufft": True,
            "nufft_eps": 1e-4,
            "remove_self_interaction": True,
            "trainable_kernel": False,
        }
    )

    n_per_batch = 12
    n_total = 2 * n_per_batch
    positions = torch.rand(n_total, 3, dtype=torch.float64)
    latent = torch.rand(n_total, 3, dtype=torch.float64) - 0.5
    batch = torch.cat(
        [
            torch.zeros(n_per_batch, dtype=torch.long),
            torch.ones(n_per_batch, dtype=torch.long),
        ]
    )

    for bid in [0, 1]:
        mask = batch == bid
        latent[mask] = latent[mask] - latent[mask].mean(dim=0, keepdim=True)

    cell = torch.stack(
        [
            torch.tensor(
                [[9.0, 0.0, 0.0], [0.6, 8.7, 0.0], [0.3, 0.5, 8.1]],
                dtype=torch.float64,
            ),
            torch.tensor(
                [[8.5, 0.0, 0.0], [0.4, 8.2, 0.0], [0.2, 0.6, 7.9]],
                dtype=torch.float64,
            ),
        ],
        dim=0,
    )

    out_exp = model(
        positions=positions.clone(),
        cell=cell,
        latent_charges=latent,
        batch=batch,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
        use_explicit_derivatives=True,
    )
    out_auto = model(
        positions=positions.clone(),
        cell=cell,
        latent_charges=latent,
        batch=batch,
        compute_energy=True,
        compute_force=True,
        compute_virial=True,
        use_explicit_derivatives=False,
    )

    assert out_exp["used_explicit_derivatives"] is True

    e_exp = out_exp["E_lr"]
    e_auto = out_auto["E_lr"]
    f_exp = out_exp["forces"]
    f_auto = out_auto["forces"]
    v_exp = out_exp["virial"]
    v_auto = out_auto["virial"]

    assert e_exp is not None and e_auto is not None
    assert f_exp is not None and f_auto is not None
    assert v_exp is not None and v_auto is not None

    assert torch.allclose(e_exp, e_auto, rtol=5e-4, atol=5e-5)
    assert torch.allclose(f_exp, f_auto, rtol=8e-3, atol=8e-4)
    assert torch.allclose(v_exp, v_auto, rtol=8e-3, atol=8e-4)
