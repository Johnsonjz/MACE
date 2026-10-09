import logging
from typing import Optional

import torch

from mace.tools.utils import AtomicNumberTable


def _copy_readout_weights(
    source: torch.nn.Module, target: torch.nn.Module
) -> None:
    """Copy weights from *source* readout to *target* readout.

    Both must be the same type (LinearReadoutBlock or NonLinearReadoutBlock).
    """
    src_state = source.state_dict()
    tgt_state = target.state_dict()
    for key, src_param in src_state.items():
        if key in tgt_state and tgt_state[key].shape == src_param.shape:
            tgt_state[key].copy_(src_param)


def _scale_module_weights(module: torch.nn.Module, factor: float) -> None:
    """Multiply all Parameter weights in *module* by *factor* in-place."""
    with torch.no_grad():
        for param in module.parameters():
            param.mul_(factor)


def _zero_readout_head(param: torch.nn.Parameter, num_heads: int, head_idx: int) -> bool:
    """Zero the FINAL-projection slice of a readout weight that drives one head.

    Only the last projection needs touching, and it is always laid out with the
    head axis last:

      LinearReadoutBlock.linear.weight      -> (C, H)
      NonLinearReadoutBlock.linear_2.weight -> (S1, H)

    Zeroing column `head_idx` there makes that head's output identically zero
    for every input, and provably leaves the other heads untouched (the other
    columns are separate).  The intermediate `linear_1` is deliberately left
    alone: it feeds ALL heads, its head blocking is not "H last"
    (flat index is c*S1 + i, so unit i = head h sits at h*J + j), and it does
    not need zeroing because its contribution to the target head is multiplied
    by the zeroed column downstream.

    Returns True if applied; False if the shape is inconsistent with num_heads
    (caller treats that as a hard error rather than silently skipping).
    """
    if num_heads <= 0:
        return False
    with torch.no_grad():
        flat = param.data.reshape(-1)
        if flat.numel() % num_heads != 0 or flat.numel() == 0:
            return False
        flat.view(-1, num_heads)[:, head_idx] = 0.0
    return True


def load_foundations_elements(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    load_readout=False,
    use_shift=True,
    use_scale=True,
    max_L=2,
    default_dtype: Optional[torch.dtype] = None,
    fresh_readout_heads=None,
):
    """
    Load the foundations of a model into a model for fine-tuning.

    fresh_readout_heads: optional iterable of head indices whose energy readout
        should be re-initialised to zero instead of inheriting the foundation's.
        The graft replicates a single-head readout into EVERY head, so in a
        multi-head setup a new head otherwise starts as a copy of the
        foundation's (e.g. an OMat head starting as the MP head) and has to
        unlearn that domain before it can learn its own.  Zeroing gives it a
        neutral start (it predicts the per-head E0s and nothing else).
    """
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    target_dtype = default_dtype or next(model.parameters()).dtype
    model_heads = model.heads
    new_z_table = table
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    # Build mapping: for each element in new_z_table, find foundation index
    # For new elements (not in foundation), use None
    indices_weights = []
    new_element_mask = []  # True for new elements
    for z in new_z_table.zs:
        try:
            idx = z_table.z_to_index(z)
            indices_weights.append(idx)
            new_element_mask.append(False)
        except ValueError:
            indices_weights.append(-1)  # placeholder, will be expanded
            new_element_mask.append(True)
    num_radial = model.radial_embedding.out_dim
    num_species = len(indices_weights)
    max_ell = model.spherical_harmonics._lmax  # pylint: disable=protected-access

    # Helper: expand foundation weights to include new elements
    # foundation_weights shape: (num_species_foundations, ...)
    # Returns expanded weights of shape (num_species, ...)
    def _expand_species_weights(foundation_weights):
        """Expand species-dependent weights, using mean for new elements."""
        flat = foundation_weights.reshape(num_species_foundations, -1)
        mean_w = flat.mean(dim=0)
        expanded = []
        for i in range(num_species):
            if new_element_mask[i]:
                expanded.append(mean_w.clone())
            else:
                expanded.append(flat[indices_weights[i]].clone())
        return torch.stack(expanded)

    # Node embedding
    foundation_node = model_foundations.node_embedding.linear.weight
    model.node_embedding.linear.weight = torch.nn.Parameter(
        _expand_species_weights(foundation_node).flatten()
        / (num_species_foundations / num_species) ** 0.5
    )
    if hasattr(model, "joint_embedding"):
        for (_, param_1), (_, param_2) in zip(
            model.joint_embedding.named_parameters(),
            model_foundations.joint_embedding.named_parameters(),
        ):
            param_1.data.copy_(param_2.data)
    if hasattr(model, "embedding_readout"):
        for (_, param_1), (_, param_2) in zip(
            model.embedding_readout.named_parameters(),
            model_foundations.embedding_readout.named_parameters(),
        ):
            param_1.data.copy_(
                param_2.data.reshape(-1, 1)
                .repeat(1, len(model_heads))
                .flatten()
                .clone()
            )
    if model.radial_embedding.bessel_fn.__class__.__name__ == "BesselBasis":
        model.radial_embedding.bessel_fn.bessel_weights = torch.nn.Parameter(
            model_foundations.radial_embedding.bessel_fn.bessel_weights.clone()
        )
    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors

        for (_, param_1), (_, param_2) in zip(
            model.interactions[i].conv_tp_weights.named_parameters(),
            model_foundations.interactions[i].conv_tp_weights.named_parameters(),
        ):
            if param_1.shape == param_2.shape:
                param_1.data.copy_(param_2.data)
            else:
                param_1.data.copy_(param_2.data[: (num_radial + 2 * num_species_foundations), ...])
        if hasattr(model.interactions[i], "linear"):
            model.interactions[i].linear.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_1"):
            model.interactions[i].linear_1.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_1.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_2"):
            model.interactions[i].linear_2.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_2.weight.clone()
            )
        if hasattr(model.interactions[i], "linear_res"):
            model.interactions[i].linear_res.weight = torch.nn.Parameter(
                model_foundations.interactions[i].linear_res.weight.clone()
            )
        if hasattr(model.interactions[i], "source_embedding"):
            src_w = model_foundations.interactions[i].source_embedding.weight
            model.interactions[i].source_embedding.weight = torch.nn.Parameter(
                _expand_species_weights(src_w).flatten()
                / (num_species_foundations / num_species) ** 0.5
            )
        if hasattr(model.interactions[i], "target_embedding"):
            tgt_w = model_foundations.interactions[i].target_embedding.weight
            model.interactions[i].target_embedding.weight = torch.nn.Parameter(
                _expand_species_weights(tgt_w).flatten()
                / (num_species_foundations / num_species) ** 0.5
            )
        if hasattr(model.interactions[i], "alpha"):
            model.interactions[i].alpha = torch.nn.Parameter(
                model_foundations.interactions[i].alpha.clone()
            )
        if hasattr(model.interactions[i], "beta"):
            model.interactions[i].beta = torch.nn.Parameter(
                model_foundations.interactions[i].beta.clone()
            )
        if model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ]:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                _expand_species_weights(
                    model_foundations.interactions[i]
                    .skip_tp.weight.reshape(
                        num_channels_foundation,
                        num_species_foundations,
                        num_channels_foundation,
                    )
                )
                .reshape(-1)
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        elif model.interactions[i].__class__.__name__ in [
            "RealAgnosticResidualNonLinearInteractionBlock",
        ]:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i].skip_tp.weight
            )
        else:
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    (max_ell + 1),
                    num_species_foundations,
                    num_channels_foundation,
                )[:, :, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        if hasattr(model.interactions[i], "density_fn"):
            for (_, param_1), (_, param_2) in zip(
                model.interactions[i].density_fn.named_parameters(),
                model_foundations.interactions[i].density_fn.named_parameters(),
            ):
                param_1.data.copy_(param_2.data)

    # Transferring products
    for i, product in enumerate(model.products):
        indices_weights_prod = indices_weights
        if hasattr(product, "use_agnostic_product"):
            if product.use_agnostic_product:
                indices_weights_prod = [0]
        max_range = max_L + 1 if i < len(model.products) - 1 else 1
        for j in range(max_range):  # Assuming 3 contractions in symmetric_contractions
            product.symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights_prod, :, :]
                    .clone()
                )
            )

            target_weights = product.symmetric_contractions.contractions[j].weights
            source_weights = (
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
                .weights
            )
            for k, _ in enumerate(target_weights):
                target_weights[k] = torch.nn.Parameter(
                    source_weights[k][indices_weights_prod, :, :].clone()
                )
        product.linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )

        # Copy U_matrix buffers (CG coefficients) from foundation.
        # These are static buffers computed during __init__; they may differ
        # between code versions (e.g. different sympy/e3nn CG decompositions).
        # The weights were trained with the foundation's U_matrices, so they
        # must match.
        for j in range(max_range):
            new_ct = product.symmetric_contractions.contractions[j]
            found_ct = (
                model_foundations.products[i]
                .symmetric_contractions.contractions[j]
            )
            # Copy buffers (U_matrix_1, U_matrix_2, U_matrix_3, ...)
            for buf_name, found_buf in found_ct.named_buffers():
                if hasattr(new_ct, buf_name):
                    new_buf = getattr(new_ct, buf_name)
                    if new_buf.shape == found_buf.shape:
                        new_buf.copy_(found_buf)
                    else:
                        # Shape mismatch – replace the buffer wholesale
                        new_ct.register_buffer(
                            buf_name, found_buf.clone().detach()
                        )
            # Replace compiled graph module to match buffer dimensions
            if hasattr(found_ct, "graph_opt_main"):
                new_ct.graph_opt_main = found_ct.graph_opt_main

    if load_readout:
        # Transferring readouts
        for i, readout in enumerate(model.readouts):
            if readout.__class__.__name__ == "LinearReadoutBlock":
                model_readouts_zero_linear_weight = readout.linear.weight.clone()
                model_readouts_zero_linear_weight = (
                    model_foundations.readouts[i]
                    .linear.weight.view(num_channels_foundation, -1)
                    .repeat(1, len(model_heads))
                    .flatten()
                    .clone()
                )
                readout.linear.weight = torch.nn.Parameter(
                    model_readouts_zero_linear_weight
                )
            if readout.__class__.__name__ in [
                "NonLinearBiasReadoutBlock",
                "NonLinearReadoutBlock",
            ]:
                assert hasattr(readout, "linear_1") or hasattr(
                    readout, "linear_mid"
                ), "Readout block must have linear_1 or linear_mid"
                if hasattr(readout, "linear_1"):
                    shape_input_1 = (
                        model_foundations.readouts[i]
                        .linear_1.__dict__["irreps_out"]
                        .num_irreps
                    )
                    shape_output_1 = readout.linear_1.__dict__["irreps_out"].num_irreps
                else:
                    raise ValueError("Readout block must have linear_1")
                if hasattr(readout, "linear_1"):
                    model_readouts_one_linear_1_weight = readout.linear_1.weight.clone()
                    model_readouts_one_linear_1_weight = (
                        model_foundations.readouts[i]
                        .linear_1.weight.view(num_channels_foundation, -1)
                        .repeat(1, len(model_heads))
                        .flatten()
                        .clone()
                    )
                    readout.linear_1.weight = torch.nn.Parameter(
                        model_readouts_one_linear_1_weight
                    )
                    if readout.linear_1.bias is not None:
                        model_readouts_one_linear_1_bias = readout.linear_1.bias.clone()
                        model_readouts_one_linear_1_bias = (
                            model_foundations.readouts[i]
                            .linear_1.bias.view(-1)
                            .repeat(len(model_heads))
                            .clone()
                        )
                        readout.linear_1.bias = torch.nn.Parameter(
                            model_readouts_one_linear_1_bias
                        )
                if hasattr(readout, "linear_mid"):
                    readout.linear_mid.weight = torch.nn.Parameter(
                        model_foundations.readouts[i]
                        .linear_mid.weight.view(
                            shape_input_1,
                            shape_input_1,
                        )
                        .repeat(len(model_heads), len(model_heads))
                        .flatten()
                        .clone()
                        / ((shape_input_1) / (shape_output_1)) ** 0.5
                    )
                    # if it has biases transfer them too
                    if readout.linear_mid.bias is not None:
                        readout.linear_mid.bias = torch.nn.Parameter(
                            model_foundations.readouts[i]
                            .linear_mid.bias.repeat(len(model_heads))
                            .clone()
                        )
                if hasattr(readout, "linear_2"):
                    model_readouts_one_linear_2_weight = readout.linear_2.weight.clone()
                    model_readouts_one_linear_2_weight = model_foundations.readouts[
                        i
                    ].linear_2.weight.view(shape_input_1, -1).repeat(
                        len(model_heads), len(model_heads)
                    ).flatten().clone() / (
                        ((shape_input_1) / (shape_output_1)) ** 0.5
                    )
                    readout.linear_2.weight = torch.nn.Parameter(
                        model_readouts_one_linear_2_weight
                    )
                    if readout.linear_2.bias is not None:
                        model_readouts_one_linear_2_bias = readout.linear_2.bias.clone()
                        model_readouts_one_linear_2_bias = (
                            model_foundations.readouts[i]
                            .linear_2.bias.view(-1)
                            .repeat(len(model_heads))
                            .flatten()
                            .clone()
                        )
                        readout.linear_2.bias = torch.nn.Parameter(
                            model_readouts_one_linear_2_bias
                        )

    # Optionally give selected heads a NEUTRAL readout instead of the
    # foundation's.  Done after the transfer loop but BEFORE the sog_readouts
    # block below, which seeds sog_readouts from model.readouts — so the charge
    # readout of a re-initialised head starts neutral too, consistently.
    #
    # Everything here is verified numerically rather than trusted: after
    # zeroing we probe each readout block on a fixed dummy input and require
    # (a) the re-initialised head's output is exactly 0, and (b) every OTHER
    # head's output is bit-for-bit unchanged.  Getting the head layout wrong is
    # otherwise silent and catastrophic - it corrupts the heads you meant to
    # keep (observed: zeroing the wrong axis dropped a healthy warm-started MP
    # head from E=26.6 to E=203.6 meV/atom).
    if load_readout and fresh_readout_heads:
        _n_heads = len(model.heads)
        _head_names = list(model.heads)
        _fresh = list(fresh_readout_heads)
        for _h in _fresh:
            if not 0 <= _h < _n_heads:
                raise ValueError(
                    f"fresh_readout_heads: head {_h} out of range (model has "
                    f"{_n_heads} heads: {_head_names})"
                )

        # Per-readout fixed probe input.  The blocks do NOT share an input width
        # (the scalar block consumes 128x0e, the non-linear one consumes
        # 128x0e+128x1o = 512), so derive it from each block's own irreps_in.
        def _probe_for(_readout):
            for _mod_name in ("linear", "linear_1"):
                _mod = getattr(_readout, _mod_name, None)
                if _mod is not None and hasattr(_mod, "irreps_in"):
                    # match the module's own dtype and device: a float64 or CPU
                    # probe against a float32/cuda model raises deep inside
                    # e3nn's tensordot / cuda mm.
                    return torch.randn(
                        3, _mod.irreps_in.dim,
                        dtype=_mod.weight.dtype, device=_mod.weight.device,
                    )
            raise RuntimeError(
                f"fresh_readout_heads: cannot determine the input width of "
                f"{type(_readout).__name__}; refusing to continue silently."
            )

        _probes = [_probe_for(_r) for _r in model.readouts]
        _before = []
        for _readout, _p in zip(model.readouts, _probes):
            with torch.no_grad():
                _before.append(_readout(_p).clone())

        _applied = []
        for _readout in model.readouts:
            _cls = _readout.__class__.__name__
            if _cls == "LinearReadoutBlock":
                _target = _readout.linear
            elif _cls in ("NonLinearBiasReadoutBlock", "NonLinearReadoutBlock"):
                _target = getattr(_readout, "linear_2", None)
            else:
                _target = None
            if _target is None or not hasattr(_target, "weight"):
                raise RuntimeError(
                    f"fresh_readout_heads: don't know which weight drives the "
                    f"heads in {_cls}; refusing to continue silently."
                )
            if not _zero_readout_head(_target.weight, _n_heads, _h):
                raise RuntimeError(
                    f"fresh_readout_heads: {_cls} weight has "
                    f"{_target.weight.numel()} elements, not divisible by "
                    f"{_n_heads} heads — refusing to continue silently."
                )
            _applied.append(f"{_cls}.{type(_target).__name__}.weight")

        # --- verify on the probe ---
        for _i, _readout in enumerate(model.readouts):
            with torch.no_grad():
                _after = _readout(_probes[_i]).clone()
            _b, _a = _before[_i], _after
            if _a.ndim < 2 or _a.shape[-1] != _n_heads:
                raise RuntimeError(
                    f"fresh_readout_heads: readout {_i} output shape "
                    f"{tuple(_a.shape)} does not end in {_n_heads} heads."
                )
            if _a[:, _h].abs().max().item() != 0.0:
                raise RuntimeError(
                    f"fresh_readout_heads: head {_h} of readout {_i} is not "
                    "zero after re-initialisation — aborting."
                )
            _others = [k for k in range(_n_heads) if k not in _fresh]
            for _k in _others:
                if not torch.equal(_b[:, _k], _a[:, _k]):
                    _d = (_b[:, _k] - _a[:, _k]).abs().max().item()
                    raise RuntimeError(
                        f"fresh_readout_heads: zeroing head {_h} perturbed "
                        f"head {_k} of readout {_i} (max|delta|={_d:.3e}). "
                        "Head layout mismatch — refusing to corrupt a head."
                    )
        logging.info(
            "Re-initialised readout for head(s) %s (%s): zeroed %s; "
            "verified other heads unchanged on a probe input.",
            _fresh, [_head_names[k] for k in _fresh], ", ".join(_applied),
        )

    # Copy readout weights to sog_readouts (if present).
    # sog_readouts are created as architectural copies of readouts during
    # MACESOG.__init__ but with random weights.  We seed them from the
    # foundation's readout weights so the initial charge predictions are
    # non-zero, then scale down (energy readout outputs eV, charges should
    # be ~0.01-0.1 e), breaking the q≈0 deadlock while keeping SOG stable.
    if hasattr(model, "sog_readouts"):
        for i, (readout, sog_readout) in enumerate(
            zip(model.readouts, model.sog_readouts)
        ):
            # If sog_readout is wrapped (e.g. by AddBiasWrapper), unwrap to
            # copy weights into the underlying readout block.
            target = sog_readout
            if hasattr(target, "base"):
                target = target.base
            _copy_readout_weights(readout, target)
            # Scale down: energy readouts produce eV, charges should be ~0.01-0.1 e.
            # A scale factor of 0.01 maps typical atomic energies (1-10 eV) to
            # charges (0.01-0.1 e), which is physically reasonable.
            _scale_module_weights(target, 0.01)

    _handled_attrs = {"interactions", "products", "readouts", "sog", "sog_readouts",
                       "per_element_charge_shift", "hardness", "chi_bias"}
    for attr_name, module in model.named_children():
        if attr_name in _handled_attrs:
            continue
        submodules = (
            list(zip(module, model_foundations.__dict__["_modules"][attr_name]))
            if isinstance(module, torch.nn.ModuleList)
            else [(module, getattr(model_foundations, attr_name))]
        )
        for sub_new, sub_found in submodules:
            for emb_name in ("source_embedding", "target_embedding"):
                if not hasattr(sub_new, emb_name):
                    continue
                emb_new = getattr(sub_new, emb_name)
                emb_found = getattr(sub_found, emb_name)
                if (
                    hasattr(emb_new, "weight")
                    and hasattr(emb_found, "weight")
                    and emb_found.weight.shape[0]
                    == num_species_foundations * num_channels_foundation
                    and emb_new.weight.shape[0] == num_species * num_channels_foundation
                ):
                    emb_new.weight = torch.nn.Parameter(
                        emb_found.weight.view(num_species_foundations, -1)[
                            indices_weights, :
                        ]
                        .flatten()
                        .clone()
                        / (num_species_foundations / num_species) ** 0.5
                    )

    if model_foundations.scale_shift is not None:
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.repeat(
                len(model_heads)
            ).clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.repeat(
                len(model_heads)
            ).clone()

    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    for name, param in foundation_state.items():
        if name not in model_state:
            continue
        if not load_readout and name.startswith("readouts."):
            continue
        # sog_readouts are handled separately in model_script_utils.py with the
        # correct use_qeq logic (χ vs direct-charge readout); copying them here
        # would overwrite the charge readout with the foundation's χ readout.
        if name.startswith("sog_readouts."):
            continue
        if model_state[name].shape != param.shape:
            continue
        model_state[name].copy_(param)

    model.to(target_dtype)

    return model


def load_foundations(
    model,
    model_foundations,
    include_readouts: bool = False,
):
    model_state = model.state_dict()
    foundation_state = model_foundations.state_dict()
    for name, param in foundation_state.items():
        if name not in model_state:
            continue
        if not include_readouts and name.startswith("readouts."):
            continue
        if model_state[name].shape != param.shape:
            continue
        model_state[name].copy_(param)
    return model
