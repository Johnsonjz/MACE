"""Unit tests for the scalar vs vector 1-layer feature choice in models.py.

The change makes the num_interactions==1 branch depend on whether hidden_irreps
carries vector channels (lmax > 0):

  * scalar 128x0e       -> legacy architecture: scalar downcast, single
                           LinearReadoutBlock readout, NO auto-switch (Residual
                           interaction block retained).
  * vector 128x0e+128x1o -> current behavior unchanged: dual readout
                            (Linear + NonLinear), Residual auto-switched to
                            RealAgnosticInteractionBlock.
"""
import os
import sys

import numpy as np
import pytest
import torch
from e3nn import o3

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mace import data, modules, tools
from mace.modules.blocks import (
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.extensions import MACESOG
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

table = tools.AtomicNumberTable([1, 8])
atomic_energies = np.array([1.0, 3.0], dtype=float)

# Periodic water-like config so both MACE and MACESOG forwards run.
config = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.239987, 0.926627, 0.0],
        ]
    ),
    cell=np.diag([5.0, 5.0, 5.0]),
    pbc=(True, True, True),
    properties={"forces": np.zeros((3, 3)), "energy": 0.0},
    property_weights={"forces": 1.0, "energy": 1.0},
)


def _base_config(hidden_irreps):
    return dict(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=1,
        num_elements=2,
        hidden_irreps=hidden_irreps,
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def _batch():
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=5.0)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False, drop_last=False
    )
    return next(iter(loader))


def test_1layer_scalar_legacy_architecture():
    model = modules.ScaleShiftMACE(**_base_config(o3.Irreps("32x0e")))
    assert type(model.interactions[0]) is RealAgnosticResidualInteractionBlock
    assert len(model.readouts) == 1
    assert type(model.readouts[0]) is LinearReadoutBlock
    assert model.products[0].linear.irreps_out == o3.Irreps("32x0e")
    out = model(_batch().to_dict())
    assert torch.isfinite(out["energy"]).all()
    assert torch.isfinite(out["forces"]).all()


def test_1layer_vector_dual_readout_unchanged():
    model = modules.ScaleShiftMACE(**_base_config(o3.Irreps("16x0e + 16x1o")))
    assert type(model.interactions[0]) is RealAgnosticInteractionBlock
    assert len(model.readouts) == 2
    assert type(model.readouts[0]) is LinearReadoutBlock
    assert type(model.readouts[1]) is NonLinearReadoutBlock
    out = model(_batch().to_dict())
    assert torch.isfinite(out["energy"]).all()
    assert torch.isfinite(out["forces"]).all()


def test_1layer_scalar_macesog_single_readout():
    pytest.importorskip("sog")
    model = MACESOG(
        sog_arguments={"use_atomwise": False, "remove_self_interaction": False},
        **_base_config(o3.Irreps("32x0e")),
    )
    assert len(model.readouts) == 1
    assert len(model.sog_readouts) == 1
    assert type(model.readouts[0]) is LinearReadoutBlock
    assert type(model.sog_readouts[0]) is LinearReadoutBlock
    out = model(_batch().to_dict())
    assert torch.isfinite(out["energy"]).all()


def test_latent_charge_dim_configurable():
    """latent_charge_dim is settable via sog_arguments and the charge readout
    emits [n_atoms, latent_charge_dim]; the direct branch must NOT broadcast
    sog_q[n] + shift[n,1] into a bogus [n, n] charge matrix."""
    pytest.importorskip("sog")
    model = MACESOG(
        sog_arguments={
            "use_atomwise": False,
            "remove_self_interaction": False,
            "latent_charge_dim": 1,
        },
        **_base_config(o3.Irreps("32x0e")),
    )
    assert model.latent_charge_dim == 1
    assert model.sog_readouts[0].linear.irreps_out == o3.Irreps("1x0e")
    out = model(_batch().to_dict())
    assert tuple(out["latent_charges"].shape) == (3, 1)
    assert torch.isfinite(out["energy"]).all()

    # With charge_init=network attached, sog_q + shift must stay [n_atoms, 1]
    # (the old [n_atoms] + [n_atoms,1] → [n_atoms,n_atoms] broadcast is gone).
    from mace.modules.charge_init import NetworkBiasInit

    NetworkBiasInit(
        {"strategy": "network", "hidden_dim": 8, "num_layers": 1, "init_scale": 0.01}
    ).initialize(model.sog_readouts, model.atomic_numbers, model)
    assert model._use_per_element_charge_init is True
    assert model.latent_charge_dim == 1
    out2 = model(_batch().to_dict())
    assert tuple(out2["latent_charges"].shape) == (3, 1)
    assert torch.isfinite(out2["energy"]).all()


if __name__ == "__main__":
    test_1layer_scalar_legacy_architecture()
    test_1layer_vector_dual_readout_unchanged()
    test_1layer_scalar_macesog_single_readout()
    print("All scalar/vector 1-layer tests passed.")
