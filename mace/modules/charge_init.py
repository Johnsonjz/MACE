"""Charge initialization strategies for SOG charge readouts.

The SOG long-range electrostatic energy is bilinear in latent charges
(E_LR ∝ q_i·q_j), so ∂E_LR/∂q ∝ q ≈ 0 at random initialization.
This module provides configurable strategies to initialize sog_readouts
to produce non-zero charges, breaking the gradient deadlock.

Strategy overview:
  - none              : No modification (current default behaviour)
  - constant          : Single global bias, all atoms get the same charge offset
  - electronegativity : Per-element bias from Pauling electronegativity differences
  - element           : Per-element bias from a user-provided {Z: charge} dict
  - weight_shift      : Shift linear.weight mean to produce a target output
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ── Pauling electronegativity table ──────────────────────────────────────
# Values from the Pauling scale.  Covers common inorganic elements.
PAULING_ELECTRONEGATIVITY: Dict[int, float] = {
    1: 2.20, 2: 0.00,
    3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 10: 0.00,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 18: 0.00,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 36: 0.00,
    37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90,
    44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
    51: 2.05, 52: 2.10, 53: 2.66, 54: 0.00,
    55: 0.79, 56: 0.89, 57: 1.10,
    # Lanthanides (58-71) — estimated values (trivalent, ~1.1-1.3)
    58: 1.12, 59: 1.13, 60: 1.14, 61: 1.15, 62: 1.17, 63: 1.18,
    64: 1.20, 65: 1.22, 66: 1.23, 67: 1.24, 68: 1.24, 69: 1.25, 70: 1.26, 71: 1.27,
    72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90, 76: 2.20,
    77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02,
    # Actinides (89-94) — estimated values
    89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38, 93: 1.36, 94: 1.28,
}


# ── Wrapper modules ──────────────────────────────────────────────────────

class AddBiasWrapper(nn.Module):
    """Wrap a readout module and add a learnable scalar bias to its output.

    The bias is a single scalar shared across all atoms.  It provides a
    simple, non-zero charge baseline that breaks the q≈0 gradient deadlock
    while keeping the architecture otherwise unchanged.
    """

    def __init__(self, base_readout: nn.Module, bias_init: float = 0.0):
        super().__init__()
        self.base = base_readout
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))

    def forward(self, x: torch.Tensor, heads: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.base(x, heads) + self.bias


class PerElementBiasModule(nn.Module):
    """Per-element learnable charge offset.

    Maps one-hot node attributes (atomic species) to a scalar charge bias
    per element, then adds it to the latent charges in the forward pass.

    Parameters
    ----------
    num_elements : int
        Number of chemical elements in the model.
    init_values : Optional[torch.Tensor]
        Initial per-element bias values, shape ``(num_elements,)``.
        Defaults to zeros.
    """

    def __init__(self, num_elements: int, init_values: Optional[torch.Tensor] = None):
        super().__init__()
        if init_values is None:
            init_values = torch.zeros(num_elements)
        else:
            init_values = init_values.to(dtype=torch.get_default_dtype())
        self.element_bias = nn.Parameter(init_values)

    def forward(self, node_attrs: torch.Tensor) -> torch.Tensor:
        """Return per-atom charge shifts, shape ``(n_atoms, 1)``."""
        return (node_attrs @ self.element_bias).unsqueeze(-1)


class AtomTypeChargeNetwork(nn.Module):
    """Network-based per-element charge offset predictor.

    Uses a small MLP over atom-type one-hot vectors to produce per-atom
    charge offsets.  The first layer is equivalent to an element embedding
    lookup (each row of the weight matrix corresponds to one element), and
    subsequent layers allow non-linear interactions between element features.

    This is more expressive than ``PerElementBiasModule`` while still being
    compact — the network learns to map atomic species to charge offsets
    via a learned representation.

    Parameters
    ----------
    num_elements : int
        Number of chemical elements.
    hidden_dim : int
        Hidden layer dimension (default 32).
    num_layers : int
        Number of hidden layers (default 2, giving 3 layers total).
    activation : callable
        Activation between hidden layers (default SiLU).
    charge_neutral : bool
        If True, subtract the per-structure mean so that Σq_i = 0 for each
        structure (essential for periodic systems with SOG Ewald).
    """

    def __init__(
        self,
        num_elements: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        activation: Optional[callable] = None,
        charge_neutral: bool = True,
    ):
        super().__init__()
        if activation is None:
            activation = nn.SiLU()
        layers = []
        in_dim = num_elements
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.charge_neutral = charge_neutral

    def forward(self, node_attrs: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return per-atom charge shifts, shape ``(n_atoms, 1)``.

        If ``charge_neutral=True`` and ``batch`` is given, the mean charge
        is subtracted per structure so that Σq_i = 0 for each graph.
        """
        q = self.mlp(node_attrs)
        if self.charge_neutral and batch is not None:
            # Per-graph mean subtraction — ensures Σq_i = 0 for each graph.
            # Loop over unique graphs: avoid scatter ops that may interact
            # poorly with the compiled graph or mixed dtypes.
            for g in batch.unique():
                mask = batch == g
                q[mask] = q[mask] - q[mask].mean()
        return q


# ── Strategy interface ───────────────────────────────────────────────────

class ChargeInitializer(ABC):
    """Abstract base class for charge initialisation strategies."""

    @abstractmethod
    def initialize(
        self,
        sog_readouts: nn.ModuleList,
        atomic_numbers: List[int],
        model: nn.Module,
    ) -> None:
        """Modify *sog_readouts* and/or *model* in-place.

        Parameters
        ----------
        sog_readouts:
            The model's ``sog_readouts`` ModuleList.  Strategies may wrap or
            replace entries.
        atomic_numbers:
            List of atomic numbers Z supported by the model, in order.
        model:
            The parent MACESOG (or ZeroInteractionMPASOG) module.  Strategies
            may attach extra sub-modules here (e.g. ``per_element_charge_shift``).
        """
        ...

    @staticmethod
    def create(config: Optional[Dict] = None) -> "ChargeInitializer":
        """Factory: return the appropriate strategy for *config*."""
        if config is None:
            return NoneInit()
        strategy = str(config.get("strategy", "none")).lower().strip()
        if strategy == "none":
            return NoneInit()
        if strategy == "constant":
            return ConstantBiasInit(config)
        if strategy == "electronegativity":
            return ElectronegativityInit(config)
        if strategy == "element":
            return ElementBiasInit(config)
        if strategy == "weight_shift":
            return WeightShiftInit(config)
        if strategy == "network":
            return NetworkBiasInit(config)
        raise ValueError(
            f"Unknown charge_init strategy: '{strategy}'. "
            f"Valid: none, constant, electronegativity, element, weight_shift, network."
        )


# ── Concrete strategies ──────────────────────────────────────────────────

class NoneInit(ChargeInitializer):
    """No charge initialisation — preserves current default behaviour."""

    def initialize(self, sog_readouts, atomic_numbers, model) -> None:
        pass


class ConstantBiasInit(ChargeInitializer):
    """Wrap every ``sog_readout`` with ``AddBiasWrapper``.

    All atoms receive the same initial charge offset, which then evolves
    independently during training.
    """

    def __init__(self, config: Dict):
        self.default_charge = float(config.get("default_charge", 0.05))

    def initialize(self, sog_readouts, atomic_numbers, model) -> None:
        for i, readout in enumerate(sog_readouts):
            sog_readouts[i] = AddBiasWrapper(readout, bias_init=self.default_charge)


class ElectronegativityInit(ChargeInitializer):
    """Per-element charge bias from Pauling electronegativity differences.

    .. math::
        q_z = \\text{scale} \\cdot (\\chi_z - \\bar{\\chi})

    where :math:`\\chi_z` is the Pauling electronegativity of element *z*
    and :math:`\\bar{\\chi}` is the mean across all elements in the model.

    Electropositive elements (Li, Na, K, …) get a positive charge;
    electronegative elements (O, F, Cl, …) get a negative charge.
    """

    def __init__(self, config: Dict):
        self.scale = float(config.get("scale", 0.1))

    def initialize(self, sog_readouts, atomic_numbers, model) -> None:
        z_list = [int(z) for z in atomic_numbers]
        chi_list = []
        missing = []
        for z in z_list:
            chi = PAULING_ELECTRONEGATIVITY.get(z)
            if chi is None:
                missing.append(z)
                chi_list.append(2.0)  # fallback: near-average electronegativity
            else:
                chi_list.append(chi)
        if missing:
            import warnings
            warnings.warn(
                f"Pauling electronegativity not available for Z={sorted(set(missing))}. "
                f"Using fallback χ=2.0. Consider using 'element' strategy instead."
            )

        chi_tensor = torch.tensor(chi_list, dtype=torch.get_default_dtype())
        mean_chi = chi_tensor.mean()
        init_values = self.scale * (chi_tensor - mean_chi)
        model.per_element_charge_shift = PerElementBiasModule(
            num_elements=len(z_list), init_values=init_values
        )
        model._use_per_element_charge_init = True


class ElementBiasInit(ChargeInitializer):
    """Per-element charge bias from a user-provided ``{Z: charge}`` dictionary.

    Example config::

        charge_init:
          strategy: element
          element_charges: {3: 0.8, 8: -1.2, 15: 0.5, 16: -0.4}
    """

    def __init__(self, config: Dict):
        raw = config.get("element_charges", {})
        # Accept both int-keyed and str-keyed YAML dicts.
        self.element_charges: Dict[int, float] = {
            int(k): float(v) for k, v in raw.items()
        }

    def initialize(self, sog_readouts, atomic_numbers, model) -> None:
        z_list = [int(z) for z in atomic_numbers]
        init_values = torch.zeros(len(z_list), dtype=torch.get_default_dtype())
        for i, z in enumerate(z_list):
            init_values[i] = self.element_charges.get(z, 0.0)
        model.per_element_charge_shift = PerElementBiasModule(
            num_elements=len(z_list), init_values=init_values
        )
        model._use_per_element_charge_init = True


class WeightShiftInit(ChargeInitializer):
    """Shift ``linear.weight`` so that expected output equals *target_mean*.

    Modifies the existing weight tensor in-place — no new parameters.
    More invasive than bias-based strategies but requires no forward-pass
    changes.
    """

    def __init__(self, config: Dict):
        self.target_mean = float(config.get("target_mean", 0.05))

    def initialize(self, sog_readouts, atomic_numbers, model) -> None:
        for readout in sog_readouts:
            if hasattr(readout, "linear") and hasattr(readout.linear, "weight"):
                w = readout.linear.weight.data
                current_mean = w.mean()
                shift = self.target_mean - current_mean
                w.add_(shift)


class NetworkBiasInit(ChargeInitializer):
    """Per-element charge bias from a learned atom-type embedding network.

    Builds a small MLP that maps one-hot atom-type vectors to per-atom
    charge offsets, then attaches it as ``model.per_element_charge_shift``.

    The first layer is equivalent to a per-element embedding lookup; deeper
    layers allow the model to learn non-linear relationships between element
    features.

    Compared to ``ConstantBiasInit`` (same offset for all atoms) this
    provides element-specific offsets.  Compared to ``ElectronegativityInit``
    it learns the offsets from the data rather than fixing them to a
    heuristic, while starting from a small random initialization that
    preserves approximate charge neutrality.

    Example config::

        charge_init:
          strategy: network
          hidden_dim: 32
          num_layers: 2
          init_scale: 0.01    # std of random weight init
    """

    def __init__(self, config: Dict):
        self.hidden_dim = int(config.get("hidden_dim", 32))
        self.num_layers = int(config.get("num_layers", 2))
        self.init_scale = float(config.get("init_scale", 0.1))
        self.charge_neutral = bool(config.get("charge_neutral", True))

    def initialize(self, sog_readouts, atomic_numbers, model) -> None:
        z_list = [int(z) for z in atomic_numbers]
        net = AtomTypeChargeNetwork(
            num_elements=len(z_list),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            charge_neutral=self.charge_neutral,
        )
        # Initialize last layer to near-zero so initial charges are small
        # but non-zero (breaks symmetry while keeping energies stable).
        last_linear = net.mlp[-1]
        nn.init.normal_(last_linear.weight, std=self.init_scale)
        nn.init.normal_(last_linear.bias, std=self.init_scale)
        # Also initialize earlier layers with small weights
        for layer in net.mlp:
            if isinstance(layer, nn.Linear) and layer is not last_linear:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        model.per_element_charge_shift = net
        model._use_per_element_charge_init = True
