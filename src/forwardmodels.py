import torch # type: ignore
import numpy as np # type: ignore
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from .propagation import propagate_z, angular_spectrum_propagation, qdht_order_l_propagation
from .sources import plane_wave, gsm_modes_1d, gsm_modes_2d_lg
from .elements import ArbitraryElement
from .simparams import SimParams
from .util import spherize_1d_torch, spherize_1d_array, complex_to_real_dtype

def propagate_arbg_N_elements(
    U: torch.Tensor, 
    sim_params: SimParams, 
    elements: tuple[ArbitraryElement, ...],
    z_distances: torch.Tensor,
    propagation_method: str = "angular",
    propagation_padding: float = 2.0,
    propagation_sequential_wavelengths: bool = False,
    propagation_checkpoint_qdht: bool = True,
    z_last_per_wavelength: torch.Tensor | None = None,
    stop_before_last_hop: bool = False,
    ) -> torch.Tensor:
    """
    Apply N elements with N propagation distances between them.
    
    Args:
        U: Input field
        sim_params: Simulation parameters
        elements: Tuple of N ArbitraryElement objects
        z_distances: 1D tensor of N distances after each element
                    (supports gradients for learnable distances)
        z_last_per_wavelength: Optional length-``num_wavelengths`` tensor that
                    overrides ``z_distances[-1]`` so each wavelength can be
                    sent to a different plane. Inter-element gaps stay shared.
        stop_before_last_hop: If True, apply the last element and return the
                    field at that plane without the final free-space hop
                    (used for axial z-scans).
    """
    if len(elements) != z_distances.shape[0]:
        raise ValueError(f"Number of elements ({len(elements)}) must match number of z distances ({z_distances.shape[0]})")
    if z_last_per_wavelength is not None:
        n_wvl = int(sim_params.weights.shape[0])
        if int(z_last_per_wavelength.numel()) != n_wvl:
            raise ValueError(
                f"z_last_per_wavelength must have {n_wvl} entries, "
                f"got shape {tuple(z_last_per_wavelength.shape)}"
            )
    
    n = len(elements)
    current_U = U
    for i, (element, z) in enumerate(zip(elements, z_distances)):
        current_U = element.apply_element(current_U, sim_params)
        is_last = i == n - 1
        if is_last and stop_before_last_hop:
            break
        z_hop = z_last_per_wavelength if (is_last and z_last_per_wavelength is not None) else z
        current_U = propagate_z(
            current_U,
            z_hop,
            sim_params,
            method=propagation_method,
            pad=True,
            padding=propagation_padding,
            sequential_wavelengths=propagation_sequential_wavelengths,
            checkpoint_qdht=propagation_checkpoint_qdht,
        )
    
    return current_U


def field_arbg_N_elements(
    x: tuple[torch.Tensor, ...],
    sim_params: SimParams,
    elem_params: dict,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    propagation_method: str = "angular",
    propagation_padding: float = 2.0,
    propagation_sequential_wavelengths: bool = False,
    propagation_checkpoint_qdht: bool = True,
    z_last_per_wavelength: torch.Tensor | None = None,
    stop_before_last_hop: bool = False,
    ) -> torch.Tensor:
    """
    Apply N arbitrary elements to a plane wave with N propagation distances.
    
    Args:
        x: Tuple of N parameter tensors, one for each element
        sim_params: Simulation parameters
        elem_params: Element parameters
        z_distances: 1D tensor of N distances after each element
                    (supports gradients for learnable distances)
        center_offsets: Tuple of N (x, y) tuples defining the center offset for each element.
                       Defaults to all zeros for backwards compatibility.
        z_last_per_wavelength: Optional per-wavelength override of the last hop.
        stop_before_last_hop: If True, return the field just after the last element.
    """
    if len(x) != z_distances.shape[0]:
        raise ValueError(f"Number of x tensors ({len(x)}) must match number of z distances ({z_distances.shape[0]})")
    
    N = len(x)
    # Set default center_offsets to all zeros if not provided
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))
    
    if len(center_offsets) != N:
        raise ValueError(f"Number of center offsets ({len(center_offsets)}) must match number of elements ({N})")
    
    elements = tuple(
        ArbitraryElement(
            name=f"ArbitraryElement{i+1}", 
            thickness=elem_params["thickness"], 
            elem_map=elem_params["elem_map"], 
            gap_map=elem_params["gap_map"], 
            x=x_i,
            membrane_thickness=elem_params["membrane_thickness"],
            membrane_map=elem_params["membrane_map"],
            center=center_offsets[i]
        )
        for i, x_i in enumerate(x)
    )

    return propagate_arbg_N_elements(
        U=plane_wave(sim_params), 
        sim_params=sim_params, 
        elements=elements,
        z_distances=z_distances,
        propagation_method=propagation_method,
        propagation_padding=propagation_padding,
        propagation_sequential_wavelengths=propagation_sequential_wavelengths,
        propagation_checkpoint_qdht=propagation_checkpoint_qdht,
        z_last_per_wavelength=z_last_per_wavelength,
        stop_before_last_hop=stop_before_last_hop,
    )


def forward_model_N_elements_mask(
    x: torch.Tensor, 
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    ) -> tuple[float, torch.Tensor]:
    
    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))
    # Split x into N equal parts for the N elements
    x_part_size = x.shape[0] // N
    x_parts = [x[i*x_part_size:(i+1)*x_part_size] for i in range(N)]
    
    # Concatenate each part with its backwards version and repeat
    x_opt_parts = []
    for x_part in x_parts:
        x_part_dbl = torch.cat((x_part, torch.flip(x_part, dims=(0,)))).view(1, -1)
        x_opt_parts.append(x_part_dbl)
    propagation_method = elem_params.get("propagation_method", "angular")
    if "propagation_padding" in elem_params:
        propagation_padding = float(elem_params["propagation_padding"])
    else:
        # QDHT is O(N^2); angular-style padding=2 can be prohibitive.
        propagation_padding = 2.0 if propagation_method == "qdht" else 2.0
    if "propagation_sequential_wavelengths" in elem_params:
        propagation_sequential_wavelengths = bool(elem_params["propagation_sequential_wavelengths"])
    else:
        # Memory-safe default for QDHT; can be overridden per run.
        propagation_sequential_wavelengths = propagation_method == "qdht"
    if "propagation_checkpoint_qdht" in elem_params:
        propagation_checkpoint_qdht = bool(elem_params["propagation_checkpoint_qdht"])
    else:
        propagation_checkpoint_qdht = True

    U_opt = field_arbg_N_elements(
        x = x_opt_parts,
        sim_params = sim_params,
        elem_params = elem_params,
        z_distances = z_distances,
        center_offsets = center_offsets,
        propagation_method=propagation_method,
        propagation_padding=propagation_padding,
        propagation_sequential_wavelengths=propagation_sequential_wavelengths,
        propagation_checkpoint_qdht=propagation_checkpoint_qdht,
    )

    weights_t = sim_params.weights.view(-1, 1, 1)
    I_out = torch.sum((U_opt.abs()**2) * weights_t, dim=0).reshape(sim_params.Nx)

    radial_objective = bool(elem_params.get("radial_objective", False))
    if radial_objective:
        # For cylindrically-symmetric (QDHT) propagation, use the radial
        # area element so the 1D objective approximates encircled power.
        r_weights = torch.abs(sim_params.x).reshape(sim_params.Nx)
        obj = torch.sum(I_out * mask.reshape(sim_params.Nx) * r_weights)
    else:
        # Cartesian line-integral objective (legacy/default behavior).
        obj = torch.sum(I_out * mask)

    return obj, I_out


def _propagation_options(elem_params: dict) -> tuple[str, float, bool, bool]:
    propagation_method = elem_params.get("propagation_method", "angular")
    if "propagation_padding" in elem_params:
        propagation_padding = float(elem_params["propagation_padding"])
    else:
        propagation_padding = 2.0 if propagation_method == "qdht" else 2.0
    if "propagation_sequential_wavelengths" in elem_params:
        propagation_sequential_wavelengths = bool(elem_params["propagation_sequential_wavelengths"])
    else:
        propagation_sequential_wavelengths = propagation_method == "qdht"
    if "propagation_checkpoint_qdht" in elem_params:
        propagation_checkpoint_qdht = bool(elem_params["propagation_checkpoint_qdht"])
    else:
        propagation_checkpoint_qdht = True
    return (
        propagation_method,
        propagation_padding,
        propagation_sequential_wavelengths,
        propagation_checkpoint_qdht,
    )


def _split_symmetric_design(x: torch.Tensor, n_elements: int) -> list[torch.Tensor]:
    if x.shape[0] % n_elements != 0:
        raise ValueError(
            f"Parameter vector length ({x.shape[0]}) must be divisible by "
            f"number of elements ({n_elements})"
        )
    x_part_size = x.shape[0] // n_elements
    x_opt_parts = []
    for i in range(n_elements):
        x_part = x[i * x_part_size : (i + 1) * x_part_size]
        x_part_dbl = torch.cat((x_part, torch.flip(x_part, dims=(0,)))).view(1, -1)
        x_opt_parts.append(x_part_dbl)
    return x_opt_parts


def forward_model_N_elements_mask_chromatic(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    z_last_per_wavelength: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Intensity-mask objective with a per-wavelength last-hop distance.

    Same optical stack as ``forward_model_N_elements_mask`` (shared geometry
    and inter-element gaps) except the final free-space hop can send each
    wavelength to its own plane. The scalar objective is

        sum_λ w_λ  ⟨ I_λ(x, z_λ), mask(x) ⟩

    so the optimizer can prescribe axial chromatic aberration (or flatten it
    by setting every ``z_λ`` equal to the same focal length).

    Args:
        x: Concatenated half-profiles, length ``N * Nx / 2``.
        sim_params: Multi-wavelength simulation parameters.
        elem_params: Element / propagation parameters.
        mask: Spatial focusing mask, broadcast over wavelengths.
        z_distances: Length-N tensor of inter-element gaps plus a placeholder
            last hop (used when ``z_last_per_wavelength`` is None).
        center_offsets: Optional per-element (x, y) offsets.
        z_last_per_wavelength: Optional length-``N_λ`` tensor of last-hop
            distances. Overrides ``z_distances[-1]``.

    Returns:
        obj: Weighted sum of per-wavelength mask overlaps.
        I_out: Spectrally weighted intensity, each color taken at its own
            plane, shape ``(Nx,)``.
    """
    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))
    x_opt_parts = _split_symmetric_design(x, N)
    (
        propagation_method,
        propagation_padding,
        propagation_sequential_wavelengths,
        propagation_checkpoint_qdht,
    ) = _propagation_options(elem_params)

    U_opt = field_arbg_N_elements(
        x=x_opt_parts,
        sim_params=sim_params,
        elem_params=elem_params,
        z_distances=z_distances,
        center_offsets=center_offsets,
        propagation_method=propagation_method,
        propagation_padding=propagation_padding,
        propagation_sequential_wavelengths=propagation_sequential_wavelengths,
        propagation_checkpoint_qdht=propagation_checkpoint_qdht,
        z_last_per_wavelength=z_last_per_wavelength,
    )

    I_lambda = (U_opt.abs() ** 2).reshape(len(sim_params.weights), sim_params.Nx)
    mask_flat = mask.reshape(sim_params.Nx)
    weights = sim_params.weights.to(dtype=I_lambda.dtype)

    radial_objective = bool(elem_params.get("radial_objective", False))
    if radial_objective:
        r_weights = torch.abs(sim_params.x).reshape(sim_params.Nx)
        spatial = mask_flat * r_weights
    else:
        spatial = mask_flat
    per_lam_obj = torch.sum(I_lambda * spatial.view(1, -1), dim=-1)
    obj = torch.sum(per_lam_obj * weights)
    I_out = torch.sum(I_lambda * weights.view(-1, 1), dim=0)
    return obj, I_out


def scan_cascade_intensity_z(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    z_distances: torch.Tensor,
    z_eval: torch.Tensor,
    mask: torch.Tensor | None = None,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Axial scan of per-wavelength intensity after a fixed cascade.

    Propagates through all elements once, then only the last free-space hop
    is repeated over ``z_eval``. Existing forward models are unchanged.

    Args:
        x: Concatenated half-profiles, same layout as the 1D forward models.
        z_distances: Length-N tensor; only ``z_distances[:-1]`` are used as
            inter-element gaps. ``z_distances[-1]`` is ignored.
        z_eval: 1D tensor of last-hop distances, shape ``(Nz,)``.
        mask: Optional focusing mask used to accumulate in-mask power.

    Returns:
        I_lambda: Intensity of shape ``(N_λ, Nz, Nx)``.
        onaxis: On-axis intensity of shape ``(N_λ, Nz)``.
        inmask: In-mask power of shape ``(N_λ, Nz)``. Zeros if ``mask`` is None.
    """
    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))
    x_opt_parts = _split_symmetric_design(x, N)
    (
        propagation_method,
        propagation_padding,
        propagation_sequential_wavelengths,
        propagation_checkpoint_qdht,
    ) = _propagation_options(elem_params)

    U_at_last = field_arbg_N_elements(
        x=x_opt_parts,
        sim_params=sim_params,
        elem_params=elem_params,
        z_distances=z_distances,
        center_offsets=center_offsets,
        propagation_method=propagation_method,
        propagation_padding=propagation_padding,
        propagation_sequential_wavelengths=propagation_sequential_wavelengths,
        propagation_checkpoint_qdht=propagation_checkpoint_qdht,
        stop_before_last_hop=True,
    )

    n_wvl = int(sim_params.weights.shape[0])
    n_z = int(z_eval.shape[0])
    nx = int(sim_params.Nx)
    real_dtype = complex_to_real_dtype(sim_params.dtype)
    I_lambda = torch.empty((n_wvl, n_z, nx), dtype=real_dtype, device=sim_params.device)
    center_idx = nx // 2
    mask_flat = None if mask is None else mask.reshape(nx).to(dtype=real_dtype)

    for iz in range(n_z):
        U_z = propagate_z(
            U_at_last,
            z_eval[iz],
            sim_params,
            method=propagation_method,
            pad=True,
            padding=propagation_padding,
            sequential_wavelengths=propagation_sequential_wavelengths,
            checkpoint_qdht=propagation_checkpoint_qdht,
        )
        I_z = (U_z.abs() ** 2).reshape(n_wvl, nx)
        I_lambda[:, iz, :] = I_z

    onaxis = I_lambda[:, :, center_idx]
    if mask_flat is None:
        inmask = torch.zeros((n_wvl, n_z), dtype=real_dtype, device=sim_params.device)
    else:
        inmask = torch.sum(I_lambda * mask_flat.view(1, 1, nx), dim=-1)
    return I_lambda, onaxis, inmask


def cascade_per_wavelength_intensity(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    z_last_per_wavelength: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-wavelength intensity after the cascade, shape ``(N_λ, Nx)``.

    If ``z_last_per_wavelength`` is set, each color is evaluated at its own
    last-hop distance; otherwise every color uses ``z_distances[-1]``.
    """
    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))
    x_opt_parts = _split_symmetric_design(x, N)
    (
        propagation_method,
        propagation_padding,
        propagation_sequential_wavelengths,
        propagation_checkpoint_qdht,
    ) = _propagation_options(elem_params)
    U_opt = field_arbg_N_elements(
        x=x_opt_parts,
        sim_params=sim_params,
        elem_params=elem_params,
        z_distances=z_distances,
        center_offsets=center_offsets,
        propagation_method=propagation_method,
        propagation_padding=propagation_padding,
        propagation_sequential_wavelengths=propagation_sequential_wavelengths,
        propagation_checkpoint_qdht=propagation_checkpoint_qdht,
        z_last_per_wavelength=z_last_per_wavelength,
    )
    return (U_opt.abs() ** 2).reshape(len(sim_params.weights), sim_params.Nx)


def forward_model_N_elements_mask_partial_coherence(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward model with partially coherent Gaussian Schell-model illumination.

    Identical interface to ``forward_model_N_elements_mask`` but replaces the
    single plane-wave source with an incoherent sum of Hermite-Gaussian
    coherent modes.  Each mode is propagated independently through the element
    cascade, and the resulting intensities are weighted by the mode
    eigenvalues.

    Coherence parameters are read from *elem_params*:
        sigma_s (float): RMS beam width (m). Defaults to ``Nx * dx`` (full
            aperture width) for approximately uniform illumination.
        sigma_g (float): RMS coherence length (m). **Required.**
        n_modes (int): Number of coherent modes. Defaults to 10.

    When ``sigma_g`` is much larger than the aperture this reduces smoothly
    to the fully coherent model (only the zeroth mode carries weight).

    Returns:
        obj: Scalar objective (intensity–mask overlap) computed under partial
             coherence.
        I_out: 1-D output intensity, shape ``(Nx,)``.
    """
    sigma_s = elem_params.get("sigma_s", sim_params.Nx * sim_params.dx)
    sigma_g = elem_params["sigma_g"]
    n_modes = elem_params.get("n_modes", 10)

    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))

    x_part_size = x.shape[0] // N
    x_parts = [x[i * x_part_size : (i + 1) * x_part_size] for i in range(N)]
    x_opt_parts = []
    for x_part in x_parts:
        x_part_dbl = torch.cat((x_part, torch.flip(x_part, dims=(0,)))).view(1, -1)
        x_opt_parts.append(x_part_dbl)

    propagation_method = elem_params.get("propagation_method", "angular")
    if "propagation_padding" in elem_params:
        propagation_padding = float(elem_params["propagation_padding"])
    else:
        propagation_padding = 1.0 if propagation_method == "qdht" else 2.0
    if "propagation_sequential_wavelengths" in elem_params:
        propagation_sequential_wavelengths = bool(
            elem_params["propagation_sequential_wavelengths"]
        )
    else:
        propagation_sequential_wavelengths = propagation_method == "qdht"
    if "propagation_checkpoint_qdht" in elem_params:
        propagation_checkpoint_qdht = bool(elem_params["propagation_checkpoint_qdht"])
    else:
        propagation_checkpoint_qdht = True

    elements = tuple(
        ArbitraryElement(
            name=f"ArbitraryElement{i+1}",
            thickness=elem_params["thickness"],
            elem_map=elem_params["elem_map"],
            gap_map=elem_params["gap_map"],
            x=x_i,
            membrane_thickness=elem_params["membrane_thickness"],
            membrane_map=elem_params["membrane_map"],
            center=center_offsets[i],
        )
        for i, x_i in enumerate(x_opt_parts)
    )

    modes, eigenvalues = gsm_modes_1d(sim_params, sigma_s, sigma_g, n_modes)

    num_wvl = len(sim_params.weights)
    weights_t = sim_params.weights.view(-1, 1, 1)
    I_out = torch.zeros(sim_params.Nx, dtype=torch.float64, device=sim_params.device)

    for m in range(n_modes):
        U_init = modes[m : m + 1].expand(num_wvl, -1, -1).to(sim_params.dtype)
        U_out = propagate_arbg_N_elements(
            U_init,
            sim_params,
            elements,
            z_distances,
            propagation_method=propagation_method,
            propagation_padding=propagation_padding,
            propagation_sequential_wavelengths=propagation_sequential_wavelengths,
            propagation_checkpoint_qdht=propagation_checkpoint_qdht,
        )
        I_mode = torch.sum((U_out.abs() ** 2) * weights_t, dim=0).reshape(
            sim_params.Nx
        )
        I_out = I_out + eigenvalues[m] * I_mode

    radial_objective = bool(elem_params.get("radial_objective", False))
    if radial_objective:
        # Match the coherent model objective definition when using radial
        # propagation assumptions.
        r_weights = torch.abs(sim_params.x).reshape(sim_params.Nx)
        obj = torch.sum(I_out * mask.reshape(sim_params.Nx) * r_weights)
    else:
        obj = torch.sum(I_out * mask)
    return obj, I_out


def forward_model_N_elements_mask_partial_coherence_2d(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    inference_only: bool = False,
    padding: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2D partially coherent version of ``forward_model_N_elements_mask_2d``.

    The source is modeled as a separable 2D Gaussian Schell-model whose Mercer
    decomposition is the outer product of the existing 1D Hermite-Gaussian
    modes along x and y. This preserves the coherent-limit behavior of
    ``forward_model_N_elements_mask_partial_coherence`` while enabling 2D
    post-processing on circularly symmetric structures.
    """
    sigma_s = elem_params.get("sigma_s", sim_params.Nx * sim_params.dx)
    sigma_g = elem_params["sigma_g"]
    n_modes = elem_params.get("n_modes", 10)

    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))

    x_part_size = x.shape[0] // N
    x_parts = tuple(
        torch.flip(x[i * x_part_size : (i + 1) * x_part_size], dims=(0,))
        for i in range(N)
    )
    x_parts_2d = tuple(spherize_1d_torch(x_part) for x_part in x_parts)

    real_dtype = complex_to_real_dtype(sim_params.dtype)

    Nx = sim_params.Nx
    Ny = sim_params.Ny
    lams = sim_params.lams
    weights = sim_params.weights
    device = sim_params.device
    dtype = sim_params.dtype
    dx = sim_params.dx

    modes_1d, eigenvalues_1d = gsm_modes_1d(sim_params, sigma_s, sigma_g, n_modes)
    modes_1d = modes_1d[:, 0, :]

    def _forward() -> torch.Tensor:
        I_acc = torch.zeros((Ny, Nx), dtype=real_dtype, device=device)

        for w in range(len(weights)):
            lam_w = lams[w : w + 1]
            sim_params_w = SimParams(
                Nx=Nx,
                Ny=Ny,
                dx=dx,
                device=device,
                dtype=dtype,
                lams=lams,
                weights=weights,
            )
            transmissions_w = []
            for i, x_part_2d in enumerate(x_parts_2d):
                element = ArbitraryElement(
                    name=f"ArbitraryElement{i + 1}",
                    thickness=elem_params["thickness"],
                    elem_map=elem_params["elem_map"],
                    gap_map=elem_params["gap_map"],
                    x=x_part_2d,
                    membrane_thickness=elem_params["membrane_thickness"],
                    membrane_map=elem_params["membrane_map"],
                    center=center_offsets[i],
                )
                transmissions_w.append(
                    element.transmission(lam_w, sim_params_w).to(dtype=dtype)
                )

            for mx in range(n_modes):
                mode_x = modes_1d[mx]
                eig_x = eigenvalues_1d[mx]

                for my in range(n_modes):
                    mode_weight = eig_x * eigenvalues_1d[my]
                    if float(mode_weight.detach().cpu()) <= 1e-12:
                        continue

                    mode_y = modes_1d[my]
                    U_w = (
                        mode_y[:, None] * mode_x[None, :]
                    ).unsqueeze(0).to(dtype=dtype)

                    for transmission_w, z in zip(transmissions_w, z_distances):
                        U_w = U_w * transmission_w
                        U_w = angular_spectrum_propagation(
                            U_w, lam_w, z, dx, device, pad=True, padding=padding
                        )

                    I_acc = I_acc + weights[w] * mode_weight.to(I_acc.dtype) * (
                        U_w.abs() ** 2
                    ).squeeze(0).to(I_acc.dtype)

        return I_acc

    if inference_only:
        with torch.no_grad():
            I_out = _forward()
    else:
        I_out = _forward()

    mask_radial = mask[0, mask.shape[1] // 2 :]
    mask_2d_np = spherize_1d_array(mask_radial.detach().cpu().numpy())
    mask_2d = torch.tensor(mask_2d_np, dtype=real_dtype, device=device)
    del mask_2d_np

    my, mx = mask_2d.shape
    if my > Ny or mx > Nx:
        crop_y = (my - Ny) // 2
        crop_x = (mx - Nx) // 2
        mask_2d = mask_2d[
            crop_y : crop_y + Ny,
            crop_x : crop_x + Nx,
        ]
    elif my < Ny or mx < Nx:
        pad_y = Ny - my
        pad_x = Nx - mx
        pad_top = pad_y // 2
        pad_left = pad_x // 2
        mask_2d = torch.nn.functional.pad(
            mask_2d,
            (pad_left, pad_x - pad_left, pad_top, pad_y - pad_top),
        )

    obj = torch.sum(I_out * mask_2d)
    return obj, I_out


def forward_model_N_elements_mask_partial_coherence_2d_qdht(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    inference_only: bool = False,
    padding: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2D partial-coherence forward model using Laguerre-Gaussian (LG) modes and
    arbitrary-order quasi-discrete Hankel transforms (QDHT) for propagation.

    The separable 2D Gaussian Schell-model used by
    ``forward_model_N_elements_mask_partial_coherence_2d`` is equivalent to the
    circularly symmetric 2D GSM; its Mercer decomposition in polar coordinates
    yields LG modes

        LG_{p,l}(r, phi) = R_{p,|l|}(r) * exp(i l phi),

    with the same eigenvalue ladder beta^k used in 1D (k = 2p + |l|). For a
    rotation-invariant optical system (radial transmissions + free space) the
    azimuthal phase exp(i l phi) is preserved, so each LG mode propagates as a
    1D radial problem via an order-|l| Hankel transform. This replaces the
    n_modes * n_modes full 2D angular-spectrum propagations used by the HGxHG
    evaluator with roughly n_modes * (n_modes + 1) / 2 much cheaper QDHT
    propagations (triangular truncation 2p + |l| <= n_modes - 1, matching the
    1D tail tolerance exactly).

    Because each coherent LG mode's intensity is circularly symmetric
    (|exp(i l phi)|^2 = 1) and we accumulate intensities incoherently, the
    output is radial; it is spherized to 2D at the end to match the interface
    of ``forward_model_N_elements_mask_partial_coherence_2d`` so this function
    can be used as a drop-in replacement in ``compute_opt_and_fzp_metrics_2d``.

    Coherence parameters (``sigma_s``, ``sigma_g``, ``n_modes``) and element
    parameters are read from ``elem_params`` as in the other partial-coherence
    models.

    Args:
        x: 1D parameter vector. Split into N equal radial profiles (same
           center-to-edge convention as ``forward_model_N_elements_mask_2d``).
        sim_params: 2D SimParams (Ny == Nx, typically). The radial grid used
                    internally has length ``N_rad = Nx // 2`` with spacing dx.
        elem_params: Element parameters. Must contain ``sigma_g`` and
                     ``thickness``, ``elem_map``, ``gap_map``,
                     ``membrane_thickness``, ``membrane_map``. Optional
                     ``sigma_s`` defaults to ``Nx * dx`` and ``n_modes``
                     defaults to 10.
        mask: 1D-style intensity mask (same convention as the existing 2D
              partial-coherence model): shape (1, Nx); the right half is
              spherized to form a 2D circular mask.
        z_distances: 1D tensor of N propagation distances.
        center_offsets: Must be None or all-zero -- this QDHT path assumes
                        perfect circular symmetry.
        inference_only: If True, run under ``torch.no_grad()``.
        padding: QDHT radial-grid expansion factor (>= 1). Unlike the 2D
                 angular-spectrum padding this pads the outer edge only.

    Returns:
        obj: Scalar objective (intensity-mask overlap) under partial coherence.
        I_out: 2D output intensity of shape (Ny, Nx).
    """
    sigma_s = elem_params.get("sigma_s", sim_params.Nx * sim_params.dx)
    sigma_g = elem_params["sigma_g"]
    n_modes = int(elem_params.get("n_modes", 10))

    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))

    for i, co in enumerate(center_offsets):
        cx = float(co[0]) if co is not None else 0.0
        cy = float(co[1]) if co is not None else 0.0
        if cx != 0.0 or cy != 0.0:
            raise ValueError(
                "forward_model_N_elements_mask_partial_coherence_2d_qdht requires "
                f"center_offsets == (0.0, 0.0) for all elements (got {co} at index {i})."
            )

    x_part_size = x.shape[0] // N
    x_parts_center_to_edge = tuple(
        torch.flip(x[i * x_part_size : (i + 1) * x_part_size], dims=(0,))
        for i in range(N)
    )

    real_dtype = complex_to_real_dtype(sim_params.dtype)
    Nx = int(sim_params.Nx)
    Ny = int(sim_params.Ny)
    dx = float(sim_params.dx)
    lams = sim_params.lams
    weights = sim_params.weights
    device = sim_params.device
    dtype = sim_params.dtype

    N_rad = x_part_size
    if N_rad < 1:
        raise ValueError(
            f"Radial grid length must be >= 1 (got x.shape[0] // N = {N_rad})."
        )

    # SimParams view used for constructing radial transmissions (Ny == 1,
    # Nx == N_rad); ArbitraryElement.transmission only reads Nx, Ny, dx, device.
    sim_params_rad = SimParams(
        Ny=1,
        Nx=N_rad,
        dx=dx,
        device=device,
        dtype=dtype,
        lams=lams,
        weights=weights,
    )
    # SimParams view used for LG-mode construction (gsm_modes_2d_lg uses
    # N_rad = Nx // 2 internally, so give it Nx == 2 * N_rad).
    sim_params_mode = SimParams(
        Ny=1,
        Nx=2 * N_rad,
        dx=dx,
        device=device,
        dtype=dtype,
        lams=lams,
        weights=weights,
    )

    radials, m_orders, eigenvalues, mults = gsm_modes_2d_lg(
        sim_params_mode, sigma_s, sigma_g, n_modes
    )

    def _forward() -> torch.Tensor:
        I_rad = torch.zeros(N_rad, dtype=real_dtype, device=device)

        for w in range(len(weights)):
            lam_w = lams[w : w + 1]

            # Per-wavelength radial transmissions, one tensor per element.
            # Each has shape (1, 1, N_rad) after ArbitraryElement.transmission
            # on the (Ny=1, Nx=N_rad) radial grid.
            transmissions_w: list[torch.Tensor] = []
            for i, x_part in enumerate(x_parts_center_to_edge):
                element = ArbitraryElement(
                    name=f"ArbitraryElement{i + 1}",
                    thickness=elem_params["thickness"],
                    elem_map=elem_params["elem_map"],
                    gap_map=elem_params["gap_map"],
                    x=x_part.view(1, N_rad),
                    membrane_thickness=elem_params["membrane_thickness"],
                    membrane_map=elem_params["membrane_map"],
                    center=(0.0, 0.0),
                )
                t_w = element.transmission(lam_w, sim_params_rad).to(dtype=dtype)
                transmissions_w.append(t_w)

            # Incoherent sum over unique (p, |l|) LG modes with multiplicity.
            for mode_idx in range(radials.shape[0]):
                m_val = int(m_orders[mode_idx].item())
                mode_weight = eigenvalues[mode_idx] * mults[mode_idx]
                if float(mode_weight.detach().cpu()) <= 1e-20:
                    continue

                U_rad = radials[mode_idx].view(1, N_rad).to(dtype=dtype)

                for t_elem, z in zip(transmissions_w, z_distances):
                    # t_elem: (1, 1, N_rad); collapse the Ny dim to match U_rad.
                    U_rad = U_rad * t_elem.squeeze(1).to(U_rad.dtype)
                    U_rad = qdht_order_l_propagation(
                        U_rad,
                        lam_w,
                        z,
                        dx,
                        device,
                        l=m_val,
                        pad=True,
                        padding=padding,
                        checkpoint_qdht=False,
                    )

                I_rad = I_rad + (
                    weights[w].to(real_dtype) * mode_weight.to(real_dtype)
                ) * (U_rad.abs() ** 2).squeeze(0).to(real_dtype)

        return I_rad

    if inference_only:
        with torch.no_grad():
            I_rad = _forward()
    else:
        I_rad = _forward()

    # Spherize the radial intensity to a 2D circularly symmetric map.
    I_2d = spherize_1d_torch(I_rad)

    my, mx = I_2d.shape
    if my > Ny or mx > Nx:
        crop_y = (my - Ny) // 2
        crop_x = (mx - Nx) // 2
        I_2d = I_2d[
            crop_y : crop_y + Ny,
            crop_x : crop_x + Nx,
        ]
    elif my < Ny or mx < Nx:
        pad_y = Ny - my
        pad_x = Nx - mx
        pad_top = pad_y // 2
        pad_left = pad_x // 2
        I_2d = torch.nn.functional.pad(
            I_2d,
            (pad_left, pad_x - pad_left, pad_top, pad_y - pad_top),
        )

    mask_radial = mask[0, mask.shape[1] // 2 :]
    mask_2d_np = spherize_1d_array(mask_radial.detach().cpu().numpy())
    mask_2d = torch.tensor(mask_2d_np, dtype=real_dtype, device=device)
    del mask_2d_np

    my, mx = mask_2d.shape
    if my > Ny or mx > Nx:
        crop_y = (my - Ny) // 2
        crop_x = (mx - Nx) // 2
        mask_2d = mask_2d[
            crop_y : crop_y + Ny,
            crop_x : crop_x + Nx,
        ]
    elif my < Ny or mx < Nx:
        pad_y = Ny - my
        pad_x = Nx - mx
        pad_top = pad_y // 2
        pad_left = pad_x // 2
        mask_2d = torch.nn.functional.pad(
            mask_2d,
            (pad_left, pad_x - pad_left, pad_top, pad_y - pad_top),
        )

    obj = torch.sum(I_2d * mask_2d)
    return obj, I_2d


def forward_model_N_elements_mask_multi_z(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    z_distances_set: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    z_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an intensity-mask objective for multiple z-distance tensors.

    This is a thin wrapper around `forward_model_N_elements_mask` that evaluates the
    objective for each provided `z_distances` tensor and returns their weighted sum.
    Each z-distance evaluation can use either a shared mask or its own mask.

    Only the intensity for the center z-distance tensor (index K // 2) is retained
    and returned; per-z intensities and objectives are not stored.

    Args:
        x: Parameter vector for the N elements (same as `forward_model_N_elements_mask`).
        sim_params: Simulation parameters.
        elem_params: Element parameters.
        mask: Intensity mask(s) used in the objective. Either:
            - a single 2D tensor [Ny, Nx] (same mask for all K z-distance tensors), or
            - a 3D tensor [K, Ny, Nx] (one mask per z-distance tensor), or
            - a list/tuple of K 2D tensors, each [Ny, Nx].
        z_distances_set: Either
            - a 2D tensor of shape [K, N], where each row is a z-distance tensor, or
            - a list/tuple of K 1D tensors, each of shape [N].
        center_offsets: Optional tuple of N (x, y) offsets for each element (shared across K).
        z_weights: Optional weights for each z-distance tensor. Can be:
            - a 1D tensor of shape [K]
            - a list/tuple of K floats
            - None (defaults to all 1.0, i.e., equal weighting)
            Negative weights are allowed and act as penalties in the total objective.

    Returns:
        obj_total: Weighted sum of objectives across all K z-distance tensors.
        I_center: Output intensity for the center z-distance tensor with shape [Ny, Nx].
    """
    # Normalize to an iterable of 1D tensors without materializing copies.
    if isinstance(z_distances_set, torch.Tensor):
        if z_distances_set.ndim == 1:
            z_iter = (z_distances_set,)
            K = 1
        elif z_distances_set.ndim == 2:
            z_iter = z_distances_set
            K = z_distances_set.shape[0]
        else:
            raise ValueError(
                f"z_distances_set must be 1D or 2D when provided as a tensor, got shape {tuple(z_distances_set.shape)}"
            )
    else:
        if len(z_distances_set) == 0:
            raise ValueError("z_distances_set must contain at least one z-distance tensor")
        z_iter = z_distances_set
        K = len(z_distances_set)

    # Normalize mask to a sequence of K masks.
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 2:
            mask_iter = [mask] * K
        elif mask.ndim == 3 and mask.shape[0] == K:
            mask_iter = [mask[i] for i in range(K)]
        else:
            raise ValueError(
                f"mask must be 2D [Ny, Nx], or 3D [K, Ny, Nx] with K={K}, got shape {tuple(mask.shape)}"
            )
    else:
        if len(mask) != K:
            raise ValueError(
                f"mask must have length {K} to match number of z-distance tensors, got length {len(mask)}"
            )
        mask_iter = list(mask)

    # Normalize weights up front so we can weight each objective as it's computed.
    if z_weights is None:
        weights = torch.ones(K, dtype=torch.float64, device=x.device)
    elif isinstance(z_weights, torch.Tensor):
        weights = z_weights.to(dtype=torch.float64, device=x.device)
        if weights.shape != (K,):
            raise ValueError(
                f"z_weights must have shape ({K},) to match number of z-distance tensors, got shape {tuple(weights.shape)}"
            )
    else:
        if len(z_weights) != K:
            raise ValueError(
                f"z_weights must have length {K} to match number of z-distance tensors, got length {len(z_weights)}"
            )
        weights = torch.tensor(z_weights, dtype=torch.float64, device=x.device)

    center_idx = K // 2
    obj_total = torch.tensor(0.0, dtype=torch.float64, device=x.device)
    I_center: torch.Tensor | None = None

    for k, (z_distances, mask_i) in enumerate(zip(z_iter, mask_iter)):
        obj_i, I_out_i = forward_model_N_elements_mask(
            x=x,
            sim_params=sim_params,
            elem_params=elem_params,
            mask=mask_i,
            z_distances=z_distances,
            center_offsets=center_offsets,
        )
        obj_total = obj_total + obj_i * weights[k]
        if k == center_idx:
            I_center = I_out_i

    return obj_total, I_center


def forward_model_N_elements_complex_mask(
    x: torch.Tensor, 
    sim_params: SimParams,
    elem_params: dict,
    field_mask: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply N arbitrary elements to a plane wave with N propagation distances,
    then compute the mean-squared error between the resulting complex field
    and a desired complex field mask.

    This uses the same optical computation as `forward_model_N_elements_mask`
    up to the complex field `U_opt`, but replaces the intensity-mask objective
    with a complex-valued MSE objective.

    Args:
        x: Parameter vector with N times the usual number of parameters
           (split into N equal parts, one per element).
        sim_params: Simulation parameters.
        elem_params: Element parameters.
        field_mask: Desired complex field at the output plane with shape
                    [Nwvl, Ny, Nx], matching `U_opt`.
        z_distances: 1D tensor of N propagation distances, one after each element.
        center_offsets: Optional tuple of N (x, y) offsets for each element.

    Returns:
        obj: Mean-squared error between `U_opt` and `field_mask`.
        U_opt: Complex field at the output plane (for analysis/visualization).
    """
    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))

    # Split x into N equal parts for the N elements
    x_part_size = x.shape[0] // N
    x_parts = [x[i * x_part_size:(i + 1) * x_part_size] for i in range(N)]

    # Concatenate each part with its backwards version (same as forward_model_N_elements_mask)
    x_opt_parts = []
    for x_part in x_parts:
        x_part_dbl = torch.cat((x_part, torch.flip(x_part, dims=(0,)))).view(1, -1)
        x_opt_parts.append(x_part_dbl)

    U_opt = field_arbg_N_elements(
        x=x_opt_parts,
        sim_params=sim_params,
        elem_params=elem_params,
        z_distances=z_distances,
        center_offsets=center_offsets,
    )

    if field_mask.shape != U_opt.shape:
        raise ValueError(
            f"field_mask shape {field_mask.shape} must match propagated field shape {U_opt.shape}"
        )

    # Complex MSE between desired field and actual field, weighted over wavelengths
    weights_t = sim_params.weights.view(-1, 1, 1)
    diff = U_opt - field_mask
    obj = torch.mean(torch.abs(diff) ** 2 * weights_t)

    return obj.real, U_opt


def forward_model_N_elements_mask_2d(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    inference_only: bool = False,
    padding: float = 2.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2D version of forward_model_N_elements_mask.

    Each element is defined by a radial profile that is expanded to a 2D
    circularly-symmetric transmission via spherize_1d_torch. Elements are
    created and propagated one at a time to limit memory usage.

    Propagation is performed one wavelength at a time. When inference_only is
    False, the entire wavelength loop is wrapped in
    ``torch.utils.checkpoint.checkpoint`` so that autograd does NOT retain the
    large intermediate tensors (FFT buffers, transfer functions, etc.) between
    the forward and backward passes. During backward the loop is re-executed to
    recompute them, trading wall-clock time for GPU memory. When
    inference_only is True, no gradients are tracked and checkpoint is skipped,
    reducing memory use for forward-only evaluation (e.g. plotting).

    Args:
        x: 1D parameter vector. Split into N equal parts, where each part is
           a radial profile (center-to-edge). NOT doubled/flipped -- spherize
           handles the symmetry.
        sim_params: Simulation parameters (must be configured for 2D, i.e.
                    Ny > 1 and typically Ny == Nx == 2*part_size - 1).
        elem_params: Element parameters dict (thickness, elem_map, gap_map,
                     membrane_thickness, membrane_map).
        mask: 1D tensor representing the mask diameter. The second half
              (center-to-edge) is spherized into a 2D mask.
        z_distances: 1D tensor of N propagation distances, one after each element.
        center_offsets: Optional tuple of N (x, y) offsets for each element.
        inference_only: If True, run under torch.no_grad() and skip gradient
                        checkpoint; gradients are not tracked and memory use is
                        lower. Use for forward-only evaluation.

    Returns:
        obj: Sum of intensity overlap with the 2D mask.
        I_out: 2D output intensity of shape (Ny, Nx).
    """
    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))

    x_part_size = x.shape[0] // N
    x_parts = tuple(torch.flip(x[i * x_part_size:(i + 1) * x_part_size], dims=(0,)) for i in range(N))

    real_dtype = complex_to_real_dtype(sim_params.dtype)

    # get all parts from sim_params so we can delete it
    Nx = sim_params.Nx
    Ny = sim_params.Ny
    lams = sim_params.lams
    weights = sim_params.weights
    device = sim_params.device
    dtype = sim_params.dtype
    dx = sim_params.dx
    del sim_params

    def _forward(*x_parts_inner):
        I_acc = torch.zeros(
            (Ny, Nx), dtype=real_dtype, device=device
        )
        for w in range(len(weights)):
            lam_w = lams[w : w + 1]
            U_w = torch.ones(
                (1, Ny, Nx),
                dtype=dtype,
                device=device,
            )
            for i, (x_part, z) in enumerate(zip(x_parts_inner, z_distances)):
                x_2d = spherize_1d_torch(x_part)
                element = ArbitraryElement(
                    name=f"ArbitraryElement{i + 1}",
                    thickness=elem_params["thickness"],
                    elem_map=elem_params["elem_map"],
                    gap_map=elem_params["gap_map"],
                    x=x_2d,
                    membrane_thickness=elem_params["membrane_thickness"],
                    membrane_map=elem_params["membrane_map"],
                    center=center_offsets[i],
                )
                del x_2d
                transmission_w = element.transmission(
                    lam_w, SimParams(Nx=Nx, Ny=Ny, dx=dx, device=device, dtype=dtype, lams=lams, weights=weights)
                    )
                U_w = U_w * transmission_w.to(U_w.dtype)
                del transmission_w
                U_w = angular_spectrum_propagation(
                    U_w, lam_w, z, dx, device, pad=True, padding=padding
                )
            I_acc = I_acc + weights[w] * (U_w.abs() ** 2).squeeze(0)
        return I_acc

    if inference_only:
        with torch.no_grad():
            I_out = _forward(*x_parts)
    else:
        I_out = grad_checkpoint(_forward, *x_parts, use_reentrant=False)


    mask_radial = mask[0, mask.shape[1] // 2:]
    mask_2d_np = spherize_1d_array(mask_radial.detach().cpu().numpy())
    mask_2d = torch.tensor(mask_2d_np, dtype=real_dtype, device=device)
    del mask_2d_np

    my, mx = mask_2d.shape
    if my > Ny or mx > Nx:
        crop_y = (my - Ny) // 2
        crop_x = (mx - Nx) // 2
        mask_2d = mask_2d[
            crop_y : crop_y + Ny,
            crop_x : crop_x + Nx,
        ]
    elif my < Ny or mx < Nx:
        pad_y = Ny - my
        pad_x = Nx - mx
        pad_top = pad_y // 2
        pad_left = pad_x // 2
        mask_2d = torch.nn.functional.pad(
            mask_2d,
            (pad_left, pad_x - pad_left, pad_top, pad_y - pad_top),
        )

    obj = torch.sum(I_out * mask_2d)
    return obj, I_out


def forward_model_N_elements_mask_2d_coherent_qdht(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    inference_only: bool = False,
    padding: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully coherent 2D forward model with cylindrically symmetric elements and
    order-0 QDHT free-space propagation.

    This is the single-mode (plane-wave) analogue of
    ``forward_model_N_elements_mask_partial_coherence_2d_qdht``: a uniform
    incident field is represented on the center-to-edge radial grid, then each
    element multiplies the radial field and propagation uses
    ``qdht_order_l_propagation`` with ``l == 0``. The result is a radial
    intensity that is spherized to 2D for the same mask and objective interface
    as ``forward_model_N_elements_mask_2d`` and
    ``forward_model_N_elements_mask_partial_coherence_2d_qdht``.

    Use this for post-optimization 2D metrics when you want QDHT-based
    diffraction (rather than 2D angular spectrum) while the optimizer still
    runs a 1D angular-spectrum model.

    ``center_offsets`` must be all zeros (circular symmetry). Optional
    ``propagation_checkpoint_qdht`` in *elem_params* (default False) matches the
    partial-coherence 2D QDHT model.
    """
    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))

    for i, co in enumerate(center_offsets):
        cx = float(co[0]) if co is not None else 0.0
        cy = float(co[1]) if co is not None else 0.0
        if cx != 0.0 or cy != 0.0:
            raise ValueError(
                "forward_model_N_elements_mask_2d_coherent_qdht requires "
                f"center_offsets == (0.0, 0.0) for all elements (got {co} at index {i})."
            )

    x_part_size = x.shape[0] // N
    x_parts_center_to_edge = tuple(
        torch.flip(x[i * x_part_size : (i + 1) * x_part_size], dims=(0,))
        for i in range(N)
    )

    real_dtype = complex_to_real_dtype(sim_params.dtype)
    Nx = int(sim_params.Nx)
    Ny = int(sim_params.Ny)
    dx = float(sim_params.dx)
    lams = sim_params.lams
    weights = sim_params.weights
    device = sim_params.device
    dtype = sim_params.dtype

    N_rad = x_part_size
    if N_rad < 1:
        raise ValueError(
            f"Radial grid length must be >= 1 (got x.shape[0] // N = {N_rad})."
        )

    if "propagation_checkpoint_qdht" in elem_params:
        propagation_checkpoint_qdht = bool(elem_params["propagation_checkpoint_qdht"])
    else:
        propagation_checkpoint_qdht = False

    sim_params_rad = SimParams(
        Ny=1,
        Nx=N_rad,
        dx=dx,
        device=device,
        dtype=dtype,
        lams=lams,
        weights=weights,
    )

    def _forward() -> torch.Tensor:
        I_rad = torch.zeros(N_rad, dtype=real_dtype, device=device)

        for w in range(len(weights)):
            lam_w = lams[w : w + 1]

            transmissions_w: list[torch.Tensor] = []
            for i, x_part in enumerate(x_parts_center_to_edge):
                element = ArbitraryElement(
                    name=f"ArbitraryElement{i + 1}",
                    thickness=elem_params["thickness"],
                    elem_map=elem_params["elem_map"],
                    gap_map=elem_params["gap_map"],
                    x=x_part.view(1, N_rad),
                    membrane_thickness=elem_params["membrane_thickness"],
                    membrane_map=elem_params["membrane_map"],
                    center=(0.0, 0.0),
                )
                t_w = element.transmission(lam_w, sim_params_rad).to(dtype=dtype)
                transmissions_w.append(t_w)

            U_rad = torch.ones((1, N_rad), dtype=dtype, device=device)
            for t_elem, z in zip(transmissions_w, z_distances):
                U_rad = U_rad * t_elem.squeeze(1).to(U_rad.dtype)
                U_rad = qdht_order_l_propagation(
                    U_rad,
                    lam_w,
                    z,
                    dx,
                    device,
                    l=0,
                    pad=True,
                    padding=padding,
                    checkpoint_qdht=propagation_checkpoint_qdht,
                )

            I_rad = I_rad + weights[w].to(real_dtype) * (U_rad.abs() ** 2).squeeze(0).to(
                real_dtype
            )
        return I_rad

    if inference_only:
        with torch.no_grad():
            I_rad = _forward()
    else:
        I_rad = _forward()

    I_2d = spherize_1d_torch(I_rad)

    my, mx = I_2d.shape
    if my > Ny or mx > Nx:
        crop_y = (my - Ny) // 2
        crop_x = (mx - Nx) // 2
        I_2d = I_2d[crop_y : crop_y + Ny, crop_x : crop_x + Nx]
    elif my < Ny or mx < Nx:
        pad_y = Ny - my
        pad_x = Nx - mx
        pad_top = pad_y // 2
        pad_left = pad_x // 2
        I_2d = torch.nn.functional.pad(
            I_2d,
            (pad_left, pad_x - pad_left, pad_top, pad_y - pad_top),
        )

    mask_radial = mask[0, mask.shape[1] // 2 :]
    mask_2d_np = spherize_1d_array(mask_radial.detach().cpu().numpy())
    mask_2d = torch.tensor(mask_2d_np, dtype=real_dtype, device=device)
    del mask_2d_np

    my, mx = mask_2d.shape
    if my > Ny or mx > Nx:
        crop_y = (my - Ny) // 2
        crop_x = (mx - Nx) // 2
        mask_2d = mask_2d[crop_y : crop_y + Ny, crop_x : crop_x + Nx]
    elif my < Ny or mx < Nx:
        pad_y = Ny - my
        pad_x = Nx - mx
        pad_top = pad_y // 2
        pad_left = pad_x // 2
        mask_2d = torch.nn.functional.pad(
            mask_2d,
            (pad_left, pad_x - pad_left, pad_top, pad_y - pad_top),
        )

    obj = torch.sum(I_2d * mask_2d)
    return obj, I_2d


def forward_model_N_elements_mask_2d_multi_z(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    mask: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    z_distances_set: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    z_weights: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    pad: bool = False,
    inference_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2D version of forward_model_N_elements_mask_multi_z.

    Evaluates forward_model_N_elements_mask_2d for each provided z_distances
    tensor and returns their weighted sum. Only the intensity for the center
    z-distance tensor (index K // 2) is retained.

    Args:
        x: Parameter vector for the N elements.
        sim_params: Simulation parameters (2D).
        elem_params: Element parameters.
        mask: Intensity mask(s). Either a single 1D tensor (shared), or a
              list/tuple of K 1D tensors (one per z-distance tensor).
        z_distances_set: Either a 2D tensor [K, N] or list/tuple of K 1D tensors.
        center_offsets: Optional tuple of N (x, y) offsets (shared across K).
        z_weights: Optional weights for each z-distance tensor. Defaults to
                   all 1.0 (equal weighting). Negative weights act as penalties.

    Returns:
        obj_total: Weighted sum of objectives across all K z-distance tensors.
        I_center: Output intensity for the center z-distance tensor (Ny, Nx).
    """
    if isinstance(z_distances_set, torch.Tensor):
        if z_distances_set.ndim == 1:
            z_iter = (z_distances_set,)
            K = 1
        elif z_distances_set.ndim == 2:
            z_iter = z_distances_set
            K = z_distances_set.shape[0]
        else:
            raise ValueError(
                f"z_distances_set must be 1D or 2D when provided as a tensor, "
                f"got shape {tuple(z_distances_set.shape)}"
            )
    else:
        if len(z_distances_set) == 0:
            raise ValueError("z_distances_set must contain at least one z-distance tensor")
        z_iter = z_distances_set
        K = len(z_distances_set)

    # Normalize mask to a sequence of K masks
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 1:
            mask_iter = [mask] * K
        elif mask.ndim == 2 and mask.shape[0] == K:
            mask_iter = [mask[i] for i in range(K)]
        else:
            raise ValueError(
                f"mask must be 1D (shared) or 2D [K, D] with K={K}, "
                f"got shape {tuple(mask.shape)}"
            )
    else:
        if len(mask) != K:
            raise ValueError(
                f"mask must have length {K} to match z-distance tensors, "
                f"got length {len(mask)}"
            )
        mask_iter = list(mask)

    if z_weights is None:
        weights = torch.ones(K, dtype=torch.float64, device=x.device)
    elif isinstance(z_weights, torch.Tensor):
        weights = z_weights.to(dtype=torch.float64, device=x.device)
        if weights.shape != (K,):
            raise ValueError(
                f"z_weights must have shape ({K},), got {tuple(weights.shape)}"
            )
    else:
        if len(z_weights) != K:
            raise ValueError(
                f"z_weights must have length {K}, got {len(z_weights)}"
            )
        weights = torch.tensor(z_weights, dtype=torch.float64, device=x.device)

    center_idx = K // 2
    obj_total = torch.tensor(0.0, dtype=torch.float64, device=x.device)
    I_center: torch.Tensor | None = None

    for k, (z_distances, mask_i) in enumerate(zip(z_iter, mask_iter)):
        obj_i, I_out_i = forward_model_N_elements_mask_2d(
            x=x,
            sim_params=sim_params,
            elem_params=elem_params,
            mask=mask_i,
            z_distances=z_distances,
            center_offsets=center_offsets,
            pad=pad,
            inference_only=inference_only,
        )
        obj_total = obj_total + obj_i * weights[k]
        if k == center_idx:
            I_center = I_out_i

    return obj_total, I_center


def orthonormal_lg_basis_by_order(
    radials: torch.Tensor,
    m_orders: torch.Tensor,
    multiplicities: torch.Tensor,
    radial_weights: torch.Tensor,
    rtol: float,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]], list[str]]:
    """Build orthonormal finite-aperture LG analysis modes from sampled profiles."""
    basis_data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}
    labels: list[str] = []
    offset = 0
    for m_order_tensor in torch.unique(m_orders):
        m_order = int(m_order_tensor.item())
        group = torch.nonzero(m_orders == m_order, as_tuple=False).flatten()
        basis = radials[group]
        gram = torch.sum(
            torch.conj(basis[:, None, :])
            * basis[None, :, :]
            * radial_weights.view(1, 1, -1),
            dim=-1,
        )
        evals, evecs = torch.linalg.eigh(gram)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]
        keep = evals > rtol * evals[0]
        evals = evals[keep]
        evecs = evecs[:, keep]

        basis_orth = (evecs.T @ basis) / torch.sqrt(evals).view(-1, 1)
        n_basis = basis_orth.shape[0]
        output_slice = torch.arange(offset, offset + n_basis, device=radials.device)
        multiplicity = float(multiplicities[group[0]].detach().cpu())

        basis_data[m_order] = (group, basis_orth, output_slice, multiplicity)
        labels.extend([f"q={q}, |l|={m_order}" for q in range(n_basis)])
        offset += n_basis
    return basis_data, labels


def project_onto_radial_basis(
    basis_orth: torch.Tensor,
    field: torch.Tensor,
    radial_weights: torch.Tensor,
) -> torch.Tensor:
    """Project a radial field onto one or more orthonormal radial basis modes."""
    return torch.sum(
        torch.conj(basis_orth) * field.view(1, -1) * radial_weights.view(1, -1),
        dim=-1,
    )


def _partial_coherence_lg_qdht_setup(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None,
) -> tuple[
    int,
    tuple[torch.Tensor, ...],
    SimParams,
    SimParams,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.dtype,
    float,
    float,
]:
    N = z_distances.shape[0]
    if center_offsets is None:
        center_offsets = tuple((0.0, 0.0) for _ in range(N))

    for i, co in enumerate(center_offsets):
        cx = float(co[0]) if co is not None else 0.0
        cy = float(co[1]) if co is not None else 0.0
        if cx != 0.0 or cy != 0.0:
            raise ValueError(
                "Partial-coherence LG QDHT models require center_offsets == (0.0, 0.0) "
                f"for all elements (got {co} at index {i})."
            )

    x_part_size = x.shape[0] // N
    x_parts_center_to_edge = tuple(
        torch.flip(x[i * x_part_size : (i + 1) * x_part_size], dims=(0,))
        for i in range(N)
    )

    real_dtype = complex_to_real_dtype(sim_params.dtype)
    Nx = int(sim_params.Nx)
    dx = float(sim_params.dx)
    lams = sim_params.lams
    weights = sim_params.weights
    device = sim_params.device
    dtype = sim_params.dtype
    N_rad = x_part_size

    sim_params_rad = SimParams(
        Ny=1,
        Nx=N_rad,
        dx=dx,
        device=device,
        dtype=dtype,
        lams=lams,
        weights=weights,
    )
    sim_params_mode = SimParams(
        Ny=1,
        Nx=2 * N_rad,
        dx=dx,
        device=device,
        dtype=dtype,
        lams=lams,
        weights=weights,
    )

    sigma_s = elem_params.get("sigma_s", Nx * dx)
    sigma_g = elem_params["sigma_g"]
    n_modes = int(elem_params.get("n_modes", 10))

    if "lg_radials" in elem_params:
        radials = elem_params["lg_radials"]
        m_orders = elem_params["lg_m_orders"]
        eigenvalues = elem_params["lg_eigenvalues"]
        multiplicities = elem_params["lg_multiplicities"]
    else:
        radials, m_orders, eigenvalues, multiplicities = gsm_modes_2d_lg(
            sim_params_mode, sigma_s, sigma_g, n_modes
        )

    radial_weights = elem_params.get(
        "radial_weights",
        2.0 * torch.pi * torch.arange(N_rad, dtype=real_dtype, device=device) * dx,
    )

    return (
        N_rad,
        x_parts_center_to_edge,
        sim_params_rad,
        sim_params_mode,
        radials,
        m_orders,
        eigenvalues,
        multiplicities,
        radial_weights,
        lams,
        weights,
        dtype,
        dx,
        float(elem_params.get("propagation_padding", 1.5)),
    )


def _partial_coherence_lg_qdht_propagate(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None,
    analysis_basis_orth: torch.Tensor | None = None,
    target_mode_indices: torch.Tensor | None = None,
    inference_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Core partial-coherence LG/QDHT propagation.

    Returns focal radial intensity and, when ``analysis_basis_orth`` is given,
    projected output powers for each analysis mode (incoherent GSM sum).
    """
    (
        N_rad,
        x_parts_center_to_edge,
        sim_params_rad,
        _sim_params_mode,
        radials,
        m_orders,
        eigenvalues,
        multiplicities,
        radial_weights,
        lams,
        weights,
        dtype,
        dx,
        padding,
    ) = _partial_coherence_lg_qdht_setup(x, sim_params, elem_params, z_distances, center_offsets)

    real_dtype = complex_to_real_dtype(dtype)
    device = sim_params.device
    checkpoint_qdht = bool(elem_params.get("propagation_checkpoint_qdht", True))

    n_analysis = 0 if analysis_basis_orth is None else int(analysis_basis_orth.shape[0])
    output_weights = (
        None
        if analysis_basis_orth is None
        else torch.zeros(n_analysis, dtype=real_dtype, device=device)
    )

    def _run() -> torch.Tensor:
        nonlocal output_weights
        I_rad = torch.zeros(N_rad, dtype=real_dtype, device=device)

        for w in range(len(weights)):
            lam_w = lams[w : w + 1]
            wavelength_weight = weights[w]

            transmissions_w: list[torch.Tensor] = []
            for i, x_part in enumerate(x_parts_center_to_edge):
                element = ArbitraryElement(
                    name=f"ArbitraryElement{i + 1}",
                    thickness=elem_params["thickness"],
                    elem_map=elem_params["elem_map"],
                    gap_map=elem_params["gap_map"],
                    x=x_part.view(1, N_rad),
                    membrane_thickness=elem_params["membrane_thickness"],
                    membrane_map=elem_params["membrane_map"],
                    center=(0.0, 0.0),
                )
                transmissions_w.append(
                    element.transmission(lam_w, sim_params_rad).to(dtype=dtype)
                )

            for mode_idx in range(radials.shape[0]):
                mode_weight = eigenvalues[mode_idx] * multiplicities[mode_idx]
                if float(mode_weight.detach().cpu()) <= 1e-20:
                    continue

                m_val = int(m_orders[mode_idx].item())
                U_rad = radials[mode_idx].view(1, N_rad).to(dtype=dtype)

                for t_elem, z in zip(transmissions_w, z_distances):
                    U_rad = U_rad * t_elem.squeeze(1).to(U_rad.dtype)
                    U_rad = qdht_order_l_propagation(
                        U_rad,
                        lam_w,
                        z,
                        dx,
                        device,
                        l=m_val,
                        pad=True,
                        padding=padding,
                        checkpoint_qdht=checkpoint_qdht,
                    )

                incoherent_weight = wavelength_weight.to(real_dtype) * mode_weight.to(real_dtype)
                I_rad = I_rad + incoherent_weight * (U_rad.abs() ** 2).squeeze(0).to(real_dtype)

                if analysis_basis_orth is not None:
                    coeffs = project_onto_radial_basis(
                        analysis_basis_orth, U_rad.squeeze(0), radial_weights
                    )
                    output_weights = output_weights + incoherent_weight * (coeffs.abs() ** 2)

        return I_rad

    if inference_only:
        with torch.no_grad():
            I_rad = _run()
    else:
        I_rad = _run()

    if target_mode_indices is not None and output_weights is not None:
        target_power = torch.sum(output_weights[target_mode_indices.to(device)])
        return I_rad, target_power

    return I_rad, output_weights


def forward_model_N_elements_partial_coherence_lg_mode_select(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    target_mode_indices: torch.Tensor,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    inference_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Partially coherent LG-mode forward model for cascade mode selection.

    A circularly symmetric Gaussian Schell-model source is decomposed into
    LG coherent modes. Each mode is propagated independently through the
    cascade via order-|l| QDHT propagation, and the objective is the
    incoherent output power projected onto the selected analysis mode(s).

    Precomputed analysis data may be supplied through ``elem_params``:
        lg_radials, lg_m_orders, lg_eigenvalues, lg_multiplicities,
        analysis_basis_orth, radial_weights.

    Coherence parameters (when modes are not precomputed):
        sigma_s, sigma_g, n_modes.

    Args:
        x: 1D design vector split into N radial profiles (center-to-edge).
        sim_params: Simulation parameters with even ``Nx`` (radial length is
                    ``Nx // 2``).
        elem_params: Element and GSM/analysis parameters.
        target_mode_indices: 1D integer tensor of analysis-mode indices to
                             maximize.
        z_distances: Propagation distances after each element.
        center_offsets: Must be None or all-zero.
        inference_only: If True, run under ``torch.no_grad()``.

    Returns:
        obj: Incoherent output power in the selected mode(s).
        I_rad: Radial focal-plane intensity, shape ``(N_rad,)``.
    """
    I_rad, target_power = _partial_coherence_lg_qdht_propagate(
        x,
        sim_params,
        elem_params,
        z_distances,
        center_offsets,
        analysis_basis_orth=elem_params["analysis_basis_orth"],
        target_mode_indices=target_mode_indices,
        inference_only=inference_only,
    )
    if target_power is None:
        raise ValueError("target_mode_indices requires analysis_basis_orth in elem_params.")
    return target_power, I_rad


def partial_coherence_lg_mode_weights(
    x: torch.Tensor,
    sim_params: SimParams,
    elem_params: dict,
    z_distances: torch.Tensor,
    center_offsets: tuple[tuple[float, float], ...] | None = None,
    inference_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate projected LG mode powers and focal intensity for a cascade design.

    Returns:
        output_weights: Projected output power per analysis mode.
        I_rad: Radial focal-plane intensity.
    """
    I_rad, output_weights = _partial_coherence_lg_qdht_propagate(
        x,
        sim_params,
        elem_params,
        z_distances,
        center_offsets,
        analysis_basis_orth=elem_params["analysis_basis_orth"],
        inference_only=inference_only,
    )
    if output_weights is None:
        raise ValueError("analysis_basis_orth is required in elem_params.")
    return output_weights, I_rad


def input_lg_mode_weights(
    radials: torch.Tensor,
    eigenvalues: torch.Tensor,
    radial_weights: torch.Tensor,
    basis_data: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]],
    n_analysis_modes: int,
) -> torch.Tensor:
    """Project the GSM LG source onto the orthonormal analysis basis."""
    weights = torch.zeros(
        n_analysis_modes, dtype=eigenvalues.dtype, device=eigenvalues.device
    )
    for _, (group, basis_orth, output_slice, multiplicity) in basis_data.items():
        for n_idx_tensor in group:
            n_idx = int(n_idx_tensor.item())
            coeffs = project_onto_radial_basis(
                basis_orth, radials[n_idx], radial_weights
            )
            weights[output_slice] = weights[output_slice] + (
                multiplicity * eigenvalues[n_idx] * (coeffs.abs() ** 2)
            )
    return weights