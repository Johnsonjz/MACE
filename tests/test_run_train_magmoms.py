import importlib.util
import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace import data, modules
from mace.tools import scripts_utils
from mace.tools import torch_geometric
from mace.tools.utils import AtomicNumberTable

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"
SOG_AVAILABLE = importlib.util.find_spec("sog") is not None


def _make_magmoms_configs() -> list[Atoms]:
    np.random.seed(7)

    base = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, -1.8, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=[True, True, True],
    )

    configs: list[Atoms] = [
        Atoms(numbers=[8], positions=[[0.0, 0.0, 0.0]], cell=[6.0, 6.0, 6.0]),
        Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]], cell=[6.0, 6.0, 6.0]),
    ]
    configs[0].info["REF_energy"] = 0.0
    configs[0].info["config_type"] = "IsolatedAtom"
    configs[0].new_array("REF_forces", np.zeros((1, 3)))
    configs[0].new_array("magmoms", np.zeros((1, 3)))

    configs[1].info["REF_energy"] = 0.0
    configs[1].info["config_type"] = "IsolatedAtom"
    configs[1].new_array("REF_forces", np.zeros((1, 3)))
    configs[1].new_array("magmoms", np.zeros((1, 3)))

    for _ in range(10):
        atoms = base.copy()
        atoms.positions += np.random.normal(0.05, size=atoms.positions.shape)
        atoms.info["REF_energy"] = float(np.random.normal(0.0, 0.1))
        atoms.new_array("REF_forces", np.random.normal(0.0, 0.1, size=(3, 3)))
        atoms.new_array("magmoms", np.random.normal(0.0, 0.5, size=(3, 3)))
        configs.append(atoms)

    return configs


def _run_train_and_check_output(tmp_path: Path, model_name: str) -> None:
    configs = _make_magmoms_configs()
    train_file = tmp_path / f"fit_{model_name}.xyz"
    ase.io.write(train_file, configs)

    params = {
        "name": f"{model_name}_magmoms",
        "model": model_name,
        "loss": "energy_forces_magmoms",
        "error_table": "EnergyForcesMagmomsRMSE",
        "train_file": str(train_file),
        "valid_fraction": 0.2,
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "magmoms_key": "magmoms",
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "magmoms_weight": 1.0,
        "r_max": 3.5,
        "hidden_irreps": "16x0e+16x1o",
        "batch_size": 4,
        "max_num_epochs": 2,
        "eval_interval": 1,
        "device": "cpu",
        "seed": 11,
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "results_dir": str(tmp_path),
        "log_dir": str(tmp_path),
        "num_workers": 0,
        "use_reduced_cg": False,
    }

    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)

    cmd = [
        sys.executable,
        str(run_train),
    ] + [
        (f"--{k}" if v is None else f"--{k}={v}")
        for k, v in params.items()
    ]

    completed = subprocess.run(cmd, env=run_env, check=True)
    assert completed.returncode == 0

    model_path = tmp_path / f"{model_name}_magmoms.model"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()

    key_spec = data.KeySpecification.from_defaults().update(
        arrays_keys={"magmoms": "magmoms"}
    )
    config = data.config_from_atoms(configs[-1], key_specification=key_spec)
    z_table = AtomicNumberTable([1, 8])
    atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=3.5)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False, drop_last=False
    )
    batch = next(iter(loader)).to("cpu")
    batch_dict = batch.to_dict()
    model_dtype = next(model.parameters()).dtype
    for key, value in batch_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            batch_dict[key] = value.to(model_dtype)

    output = model(
        batch_dict,
        training=False,
        compute_force=False,
        compute_virials=False,
        compute_stress=False,
    )

    assert output.get("magmoms") is not None
    assert output["magmoms"].shape == (len(configs[-1]), 3)
    assert torch.isfinite(output["magmoms"]).all()


def test_run_train_magmoms_mace(tmp_path):
    _run_train_and_check_output(tmp_path, model_name="MACE")


@pytest.mark.skipif(not SOG_AVAILABLE, reason="sog package is not available")
def test_run_train_magmoms_macesog(tmp_path):
    _run_train_and_check_output(tmp_path, model_name="MACESOG")


def test_get_params_options_includes_magmom_readout():
    model = modules.MACE(
        r_max=3.5,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.array([0.0, 0.0], dtype=float),
        avg_num_neighbors=2.0,
        atomic_numbers=[1, 8],
        correlation=2,
        gate=torch.nn.functional.silu,
        radial_type="bessel",
        compute_magmoms=True,
    )

    args = SimpleNamespace(
        lr_params_factors=(
            '{"embedding_lr_factor": 1.0, "interactions_lr_factor": 1.0, '
            '"products_lr_factor": 1.0, "readouts_lr_factor": 1.0}'
        ),
        freeze=0,
        weight_decay=1e-8,
        lr=1e-3,
        amsgrad=False,
        beta=0.9,
    )

    param_options = scripts_utils.get_params_options(args, model)
    names = [group["name"] for group in param_options["params"]]
    assert "magmom_readout" in names

    magmom_group = next(
        group for group in param_options["params"] if group["name"] == "magmom_readout"
    )
    assert list(magmom_group["params"])
    assert np.isclose(magmom_group["lr"], args.lr)
