import math
import numpy as np # type: ignore
import torch # type: ignore
from typing import Tuple, Callable, Any, Dict, List
from datetime import datetime
import pandas as pd
import scipy
import matplotlib
from pathlib import Path
from scipy.ndimage import distance_transform_edt


def colors_list(N: int) -> List:
    """Return N evenly spaced colors from the coolwarm colormap."""
    cmap = matplotlib.colormaps["coolwarm"]
    return [cmap(x) for x in np.linspace(0, 1, N)]

def quasi_monochromatic_spectrum(
        central_energy_ev: float, 
        N: int, 
        bandwidth: float,
        device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    h = 4.135667696e-15 # eV s
    c = 299792458 # m/s

    energy_spread_ev = central_energy_ev * bandwidth
    min_energy_ev = central_energy_ev - energy_spread_ev / 2.0
    max_energy_ev = central_energy_ev + energy_spread_ev / 2.0

    # Create a linearly spaced array of energies across the band
    # If only one wavelength is requested, use the central energy
    if N == 1:
        energies_ev = np.array([central_energy_ev])
    else:
        energies_ev = np.linspace(min_energy_ev, max_energy_ev, N)

    wavelengths_m = (h * c) / energies_ev

    weights = np.ones(N) / N # uniform distribution

    return torch.tensor(wavelengths_m, dtype=torch.float32, device=device), torch.tensor(weights, dtype=torch.float32, device=device)


def gaussian_energy_spectrum(
        central_energy_ev: float,
        N: int,
        bandwidth: float,
        device: torch.device = torch.device("cpu"),
        window_sigmas: float = 3.0,
        bandwidth_in_wavelength: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a spectrum whose weights follow a Gaussian distribution centered at
    `central_energy_ev`.

    Behavior depends on `bandwidth_in_wavelength`:
    - If False (default): Gaussian is defined in ENERGY. FWHM(E) = `bandwidth * central_energy_ev`.
      Sampling is uniform in energy over +/- `window_sigmas` sigmas.
    - If True: Gaussian is defined in WAVELENGTH. FWHM(λ) = `bandwidth * λ0` where
      λ0 = hc / `central_energy_ev`. Sampling is uniform in wavelength over +/- `window_sigmas` sigmas.

    Args:
        central_energy_ev: Central photon energy in eV.
        N: Number of discrete samples to return.
        bandwidth: Fractional FWHM relative to central energy, in (0, 1].
        device: Torch device for returned tensors.
        window_sigmas: Half-width of the sampling window, in units of Gaussian sigma.
        bandwidth_in_wavelength: If True, interpret `bandwidth` as fractional FWHM in wavelength
            and construct Gaussian weights in wavelength domain. Otherwise, in energy domain.

    Returns:
        wavelengths_m (torch.Tensor): Shape (N,), wavelengths corresponding to sampled energies.
        weights (torch.Tensor): Shape (N,), non-negative weights normalized to sum to 1.
    """
    h = 4.135667696e-15  # eV s
    c = 299792458        # m/s

    # Handle trivial cases
    if N <= 0:
        return (torch.empty(0, dtype=torch.float32, device=device),
                torch.empty(0, dtype=torch.float32, device=device))

    lam0 = (h * c) / central_energy_ev

    if N == 1 or bandwidth <= 0:
        return (torch.tensor([lam0], dtype=torch.float32, device=device),
                torch.tensor([1.0], dtype=torch.float32, device=device))

    two_sqrt_2ln2 = 2.0 * np.sqrt(2.0 * np.log(2.0))

    if not bandwidth_in_wavelength:
        # ENERGY-domain Gaussian
        fwhm_ev = max(central_energy_ev * bandwidth, 1e-12)
        sigma_ev = fwhm_ev / two_sqrt_2ln2

        # Sample energies uniformly over +/- window_sigmas * sigma
        e_min = max(central_energy_ev - window_sigmas * sigma_ev, 1e-12)
        e_max = max(central_energy_ev + window_sigmas * sigma_ev, e_min * (1.0 + 1e-9))
        energies_ev = np.linspace(e_min, e_max, N)

        # Gaussian weights in energy
        weights = np.exp(-0.5 * ((energies_ev - central_energy_ev) / sigma_ev) ** 2)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones_like(weights) / N
        else:
            weights = weights / weights_sum

        wavelengths_m = (h * c) / energies_ev
    else:
        # WAVELENGTH-domain Gaussian
        fwhm_lam = max(lam0 * bandwidth, 1e-24)
        sigma_lam = fwhm_lam / two_sqrt_2ln2

        # Sample wavelengths uniformly over +/- window_sigmas * sigma
        lam_min = max(lam0 - window_sigmas * sigma_lam, 1e-24)
        lam_max = max(lam0 + window_sigmas * sigma_lam, lam_min * (1.0 + 1e-12))
        wavelengths_m = np.linspace(lam_min, lam_max, N)

        # Gaussian weights in wavelength
        weights = np.exp(-0.5 * ((wavelengths_m - lam0) / sigma_lam) ** 2)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones_like(weights) / N
        else:
            weights = weights / weights_sum

    return (
        torch.tensor(wavelengths_m, dtype=torch.float32, device=device),
        torch.tensor(weights, dtype=torch.float32, device=device),
    )


H_EV_S = 4.135667696e-15
C_M_S = 299792458.0
HC_EV_M = H_EV_S * C_M_S


def wavelengths_to_energy_ev(lams: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert wavelength (m) to photon energy (eV)."""
    if isinstance(lams, torch.Tensor):
        return HC_EV_M / lams
    return HC_EV_M / np.asarray(lams)


def linear_energy_focal_shift(
    lams: torch.Tensor,
    f: float,
    half_width: float,
) -> torch.Tensor:
    """Map sampled wavelengths onto last-hop distances linear in energy.

    Lowest energy (longest wavelength) is sent to ``f - half_width`` and
    highest energy to ``f + half_width``, matching the sign of FZP chromatic
    walk (``f ∝ E``). A single wavelength maps to ``f``.
    """
    energies = wavelengths_to_energy_ev(lams)
    if energies.numel() == 1:
        return torch.full_like(energies, float(f), dtype=torch.float64)
    e_min = torch.min(energies)
    e_max = torch.max(energies)
    t = (energies - e_min) / (e_max - e_min)
    z = float(f) + float(half_width) * (2.0 * t - 1.0)
    return z.to(dtype=torch.float64)


def create_material_map(
        material_name: str, 
) -> list[np.ndarray]:
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir.parent / "data" / f"{material_name}.csv"
    df = pd.read_csv(csv_path)
    wavelengths = df["wl"].to_numpy() / 1e9 # convert to m
    delta = df["delta"].to_numpy()
    beta = df["beta"].to_numpy()
    return [wavelengths, 1.0 - delta + 1j*beta]


def refractive_index_at_wvl(
        wvl: torch.Tensor, 
        material_map: list[np.ndarray], 
) -> torch.Tensor:
    wavelengths = material_map[0]
    refractive_indices = material_map[1]
    return torch.tensor(np.interp(wvl.cpu().numpy(), wavelengths, refractive_indices), dtype=torch.complex64, device=wvl.device)


def spherize_1d_array(radial_profile: np.ndarray) -> np.ndarray:
    """
    Expands a 1D radial profile into a 2D array with circular symmetry.

    This function takes a 1D array, representing function values along a radius,
    and generates a 2D square array where the value of each pixel is determined
    by its radial distance from the center. This effectively "rotates" the 1D
    profile around the center to fill a 2D space.

    Args:
        radial_profile: A 1D NumPy array of size N representing the function's
                        values along a radius.

    Returns:
        A 2D NumPy array of shape (2*N-1, 2*N-1) representing the
        circularly symmetric function.
    """
    # Get the radius N from the length of the input array.

    if radial_profile.ndim != 1:
        radial_profile = radial_profile.reshape(-1)
    n = radial_profile.shape[0]
    
    diameter = 2 * n - 1
    
    center_x, center_y = n - 1, n - 1

    x = np.arange(diameter) - center_x
    y = np.arange(diameter) - center_y
    xx, yy = np.meshgrid(x, y)

    radius_grid = np.sqrt(xx**2 + yy**2)

    index_grid = np.round(radius_grid).astype(int)

    output_2d = np.zeros((diameter, diameter), dtype=radial_profile.dtype)

    valid_mask = index_grid < n

    output_2d[valid_mask] = radial_profile[index_grid[valid_mask]]

    return output_2d


def spherize_1d_torch(radial_profile: torch.Tensor) -> torch.Tensor:
    """
    Differentiable PyTorch version of spherize_1d_array.

    Takes a 1D tensor of length N (radial profile from center to edge) and
    returns a 2D tensor of shape (2*N-1, 2*N-1) with circular symmetry.
    Pixels at radius >= N are zeroed out.

    The index grid is geometry-only and carries no gradient; the gather via
    fancy indexing is differentiable w.r.t. radial_profile.
    """
    if radial_profile.ndim != 1:
        radial_profile = radial_profile.reshape(-1)
    n = radial_profile.shape[0]
    diameter = 2 * n - 1
    center = n - 1

    coords = torch.arange(diameter, device=radial_profile.device, dtype=torch.float32) - center
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    index_grid = torch.round(torch.sqrt(xx ** 2 + yy ** 2)).long()

    valid_mask = index_grid < n
    safe_indices = index_grid.clamp(0, n - 1)

    output_2d = radial_profile[safe_indices.reshape(-1)].reshape(diameter, diameter)
    output_2d = output_2d * valid_mask.to(output_2d.dtype)
    return output_2d


def complex_to_real_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if complex_dtype == torch.complex128 else torch.float32


def width_central_peak(arr, threshold):
    """
    Calculate the width of the central peak of an array.
    Args:
        arr: The array to calculate the width of the central peak of.
        threshold: The threshold to use to determine the width of the central peak.

    Returns:
        The width of the central peak of the array.
    """
    max_idx = arr.shape[0] // 2
    max_val = arr[max_idx]
    
    below_thresh = arr < (max_val * threshold)
    
    left_below = np.where(below_thresh[:max_idx])[0]
    left_boundary = left_below[-1] if left_below.size > 0 else 0
    
    right_below = np.where(below_thresh[max_idx:])[0]
    right_boundary = (right_below[0] + max_idx) if right_below.size > 0 else len(arr) - 1
    
    width = right_boundary - left_boundary

    return width


def focusing_gain(I_arr, threshold):
    """
    Calculate the focusing gain of an array.

    Args:
        I_arr: The array to calculate the focusing gain of.
        threshold: The threshold to use to determine the width of the central peak.

    Returns:
        The focusing gain of the array.
    """
    peak_width = width_central_peak(I_arr, threshold)
    arr_width = I_arr.shape[0]

    power_center = np.sum(I_arr[arr_width//2 - peak_width//2 : arr_width//2 + peak_width//2])
    efficiency = power_center / arr_width

    gain = efficiency * (arr_width**2 / peak_width**2)

    return gain


def compute_width_and_efficiency_from_2d(
    intensity_2d: np.ndarray,
    *,
    Nx: int,
    dx: float,
    min_feature_size: float,
    focusing_threshold: float,
    crop_width: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Common helper: given a 2D focal-plane intensity, compute
    - cropped intensity around the center (crop_width x crop_width)
    - effective focal spot width (in pixels, radius)
    - efficiency (power in circular aperture / power in pupil of radius Nx/2)

    The logic mirrors the post-processing in scripts/xray_focusing_testing.py
    (central symmetric 1D profile for width, circular aperture for efficiency).
    """
    if intensity_2d.ndim != 2:
        raise ValueError("intensity_2d must be a 2D array")

    ny, nx = intensity_2d.shape
    cy = ny // 2
    cx = nx // 2

    # Safety check: ensure crop_width does not exceed array bounds
    if crop_width > min(ny, nx):
        crop_width = min(ny, nx)

    half = crop_width // 2
    y_start = cy - half
    y_end = cy + half
    x_start = cx - half
    x_end = cx + half

    cropped = intensity_2d[y_start:y_end, x_start:x_end]

    # Symmetric 1D profile from the central row, using the full 2D field
    center_row_half = intensity_2d[cy, cx:]
    center_row = np.concatenate((np.flip(center_row_half, axis=0), center_row_half))

    width = width_central_peak(center_row, focusing_threshold) // 2
    if width > Nx / 10:
        airy_width = int(2 * 1.22 * min_feature_size / dx) // 2
        width = airy_width

    y, x = np.ogrid[:crop_width, :crop_width]
    dist_sq = (x - crop_width // 2) ** 2 + (y - crop_width // 2) ** 2
    eff_mask = dist_sq <= width**2

    efficiency = cropped[eff_mask].sum() / (np.pi * (Nx / 2) ** 2)

    return float(width), float(efficiency), cropped, center_row


def compute_opt_and_fzp_metrics_2d(
    rho_bar: torch.Tensor,
    sim_params_1d: Any,
    fwd_model_args: tuple,
    *,
    min_feature_size: float,
    focusing_threshold: float,
    crop_width: int,
    forward_model_1d: Callable[..., Any],
    forward_model_2d: Callable[..., Any],
    zp_init_func: Callable[..., Any],
    compute_fzp: bool = True,
) -> Dict[str, Any]:
    """
    Standardized computation of optimized and (optionally) FZP metrics.

    This helper supports both:
    - angular/2D propagation (evaluate directly with `forward_model_2d`)
    - qdht/1D propagation (evaluate with `forward_model_1d`, then radially
      expand the 1D focal profile to a synthetic 2D map for consistent width
      and efficiency post-processing).

    In both cases, width/efficiency and stored 1D traces come from the same
    `compute_width_and_efficiency_from_2d` path so metric definitions are
    consistent across propagation methods.
    """
    if rho_bar is None:
        raise ValueError("rho_bar must be a tensor")

    if not isinstance(fwd_model_args, tuple) or len(fwd_model_args) != 4:
        raise ValueError(
            "fwd_model_args must be a 4-tuple (elem_params, mask, z_dists, center_offsets)"
        )

    elem_params, mask, z_dists, center_offsets = fwd_model_args

    # Extract scalar/grid parameters from 1D SimParams.
    Nx = int(getattr(sim_params_1d, "Nx"))
    dx = float(getattr(sim_params_1d, "dx"))
    device = getattr(sim_params_1d, "device")
    dtype = getattr(sim_params_1d, "dtype")
    lams = getattr(sim_params_1d, "lams")
    weights = getattr(sim_params_1d, "weights")

    propagation_method = str(elem_params.get("propagation_method", "angular")).lower()
    use_qdht_eval = propagation_method == "qdht"

    # 2D SimParams built from the 1D instance (same grid, square aperture).
    sim_params_2d = None
    if not use_qdht_eval:
        SimParamsCls = type(sim_params_1d)
        sim_params_2d = SimParamsCls(
            Ny=Nx,
            Nx=Nx,
            dx=dx,
            device=device,
            dtype=dtype,
            lams=lams,
            weights=weights,
        )

    def _tensor_or_array_to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _scalar_to_float(x: Any) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)

    def _match_square_shape(arr2d: np.ndarray, target_size: int) -> np.ndarray:
        """
        Center-crop or center-pad to (target_size, target_size).
        """
        out = arr2d
        h, w = out.shape
        if h > target_size:
            y0 = (h - target_size) // 2
            out = out[y0 : y0 + target_size, :]
            h = out.shape[0]
        if w > target_size:
            x0 = (w - target_size) // 2
            out = out[:, x0 : x0 + target_size]
            w = out.shape[1]
        if h < target_size or w < target_size:
            pad_y = target_size - h
            pad_x = target_size - w
            pad_top = pad_y // 2
            pad_bottom = pad_y - pad_top
            pad_left = pad_x // 2
            pad_right = pad_x - pad_left
            out = np.pad(out, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
        return out

    def _intensity_2d_from_qdht_slice(intensity_1d: np.ndarray) -> np.ndarray:
        """
        Convert a 1D diameter slice into a circularly symmetric 2D intensity map.
        """
        intensity_1d = intensity_1d.reshape(-1)
        radial_half = intensity_1d[intensity_1d.shape[0] // 2 :]
        intensity_2d = spherize_1d_array(radial_half)
        return _match_square_shape(intensity_2d, Nx)

    def _evaluate_metrics_for_design(
        design_x: torch.Tensor,
        design_fwd_model_args: tuple,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        if use_qdht_eval:
            obj_t, intensity_1d_t = forward_model_1d(
                design_x, sim_params_1d, *design_fwd_model_args
            )
            intensity_1d = _tensor_or_array_to_numpy(intensity_1d_t)
            intensity_2d = _intensity_2d_from_qdht_slice(intensity_1d)
            final_obj = _scalar_to_float(obj_t)
        else:
            assert sim_params_2d is not None
            obj_t, intensity_2d_t = forward_model_2d(
                design_x,
                sim_params_2d,
                *design_fwd_model_args,
                inference_only=True,
                padding=1.0,
            )
            intensity_2d = _tensor_or_array_to_numpy(intensity_2d_t)
            final_obj = _scalar_to_float(obj_t)

        width, efficiency, cropped, center_row = compute_width_and_efficiency_from_2d(
            intensity_2d,
            Nx=Nx,
            dx=dx,
            min_feature_size=min_feature_size,
            focusing_threshold=focusing_threshold,
            crop_width=crop_width,
        )
        return final_obj, width, efficiency, cropped, center_row

    opt_final_obj, opt_width, opt_efficiency, opt_cropped, opt_center_row = _evaluate_metrics_for_design(
        rho_bar, fwd_model_args
    )

    result: Dict[str, Any] = {
        "opt_final_obj": float(opt_final_obj),
        "opt_width": float(opt_width),
        "opt_efficiency": float(opt_efficiency),
        "opt_intensity_cropped": opt_cropped,
        "opt_intensity_1d": opt_center_row,
    }

    if compute_fzp:
        # FZP baseline: extract focal length from last z-distance entry
        if hasattr(z_dists, "__getitem__"):
            z_last = z_dists[-1]
        else:
            z_last = z_dists

        if isinstance(z_last, torch.Tensor):
            f_scalar = float(z_last.detach().cpu().item())
        else:
            f_scalar = float(z_last)

        # Build zone-plate design using provided zp_init
        lam_center = lams[lams.argmax()]
        fzp_profile = zp_init_func(lam_center, f_scalar, min_feature_size, 1, sim_params_1d)
        if isinstance(fzp_profile, torch.Tensor):
            fzp_x = fzp_profile.to(device=device, dtype=torch.float64)
        else:
            fzp_x = torch.tensor(fzp_profile, dtype=torch.float64, device=device)

        fzp_z_dists = torch.tensor([f_scalar], device=device, dtype=torch.float64)
        fzp_centers = ((0.0, 0.0),)
        fzp_fwd_model_args = (elem_params, mask, fzp_z_dists, fzp_centers)

        fzp_final_obj, fzp_width, fzp_efficiency, fzp_cropped, fzp_center_row = _evaluate_metrics_for_design(
            fzp_x, fzp_fwd_model_args
        )

        result.update(
            {
                "fzp_final_obj": float(fzp_final_obj),
                "fzp_width": float(fzp_width),
                "fzp_efficiency": float(fzp_efficiency),
                "fzp_intensity_cropped": fzp_cropped,
                "fzp_intensity_1d": fzp_center_row,
                "fzp_x": fzp_x,
            }
        )

    return result

def get_feature_sizes(vector: np.ndarray) -> np.ndarray:
    """
    Calculates the size of consecutive features of 0s in a 1D binary NumPy array.
    Features consisting of 1s are disregarded.

    A feature is defined as an uninterrupted sequence of the same value (e.g., 0, 0, 0).

    Args:
        vector: A 1D NumPy array containing only 0s and 1s.

    Returns:
        A 1D NumPy array of type int containing the sizes of the zero-features
        in the order they appear.
    """
    if vector.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    if vector.size == 0:
        return np.array([], dtype=int)

    change_indices = np.flatnonzero(vector[:-1] != vector[1:])

    boundaries = np.concatenate([
        np.array([-1]),
        change_indices,
        np.array([vector.shape[0] - 1])
    ])

    feature_sizes = np.diff(boundaries)
    
    feature_start_indices = boundaries[:-1] + 1
    feature_values = vector[feature_start_indices]

    feature_sizes = feature_sizes[1:-1]

    return feature_sizes

def get_formatted_datetime():
    now = datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d %H.%M.%S")
    # remove dashes and periods
    formatted_datetime = formatted_datetime.replace("-", "").replace(".", "")
    # replace colons with underscores
    formatted_datetime = formatted_datetime.replace(":", "_")
    # remove spaces
    formatted_datetime = formatted_datetime.replace(" ", "_")
    return formatted_datetime

def airy_1d_intensity(x, diffraction_limit):
    """
    Calculates the 1D Airy intensity profile.
    
    The function is normalized such that the peak intensity at x=0 is 1.0.
    The diffraction_limit is treated as the radius of the first zero (null).
    
    Parameters:
    -----------
    x : numpy array
        The spatial coordinates (1D array of positions).
    diffraction_limit : float
        The distance from the center to the first zero (null) of the pattern.
        (e.g., 1.22 * lambda * f_number)
        
    Returns:
    --------
    intensity : numpy array
        The normalized intensity values corresponding to x.
    """

    j1_first_zero = 3.8317059702025124

    k = j1_first_zero / diffraction_limit
    u = k * x
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # The optical field amplitude E(x) ~ 2*J1(u)/u
        field = 2.0 * scipy.special.j1(u) / u
        
    field = np.nan_to_num(field, nan=1.0)
    intensity = field ** 2
    
    return intensity

def smooth_photonic_parameters(params, blur_radius):
    """
    Applies a Gaussian blur to binary photonic parameters to simulate 
    imperfect (rounded) sidewall profiles.

    Args:
        params (np.ndarray): 1D array of binary parameters (0s and 1s).
        blur_radius (float): Controls the amount of curvature/smoothing. 
                             Analagous to the standard deviation (sigma) of the 
                             Gaussian kernel.
                             - Small value (e.g., 0.5) -> Sharp, nearly binary edges.
                             - Large value (e.g., 5.0) -> Very rounded, gradual transitions.

    Returns:
        np.ndarray: Array of floats in range [0, 1] representing the smoothed structure.
    """
    # Ensure input is float for the convolution
    if type(params) == torch.Tensor:
        params = params.cpu().numpy()
    params_float = params.astype(float)
    
    # Apply Gaussian filter
    # mode='nearest' extends the edge values out, preventing the ends 
    # of the array from dipping towards zero artificially.
    smoothed_params = scipy.ndimage.gaussian_filter(params_float, sigma=blur_radius, mode='nearest')
    
    # Clip to ensure numerical precision didn't push us slightly outside [0, 1]
    smoothed_params = np.clip(smoothed_params, 0.0, 1.0)
    
    return smoothed_params

def apply_morphological_error_1d(vector, operation='dilate', strength=1):
    """
    Applies 1D morphological erosion or dilation to simulate fabrication errors.

    Uses a Euclidean distance transform on the 1D pixel grid, so ``strength``
    may be any non-negative real number (pixel units). Integer strengths match
    repeated ``binary_dilation`` / ``binary_erosion`` with the default
    1-connectivity structuring element along the line.

    Parameters:
    - vector: 1D numpy array of shape (Nx,) containing binary values (0s and 1s).
    - operation: String, either 'erode' or 'dilate'.
    - strength: Pixels of expansion ('dilate') or shrink ('erode'); may be float.

    Returns:
    - Modified 1D numpy array of the same shape and dtype.
    """
    original_dtype = vector.dtype
    fg = vector.astype(bool)
    strength = float(strength)

    if strength <= 0:
        return np.asarray(vector, dtype=original_dtype)

    # Pad with void so EDT boundary behavior matches scipy's binary morphology
    # (implicit void outside the segment).
    pad = int(math.ceil(strength))
    padded_fg = np.pad(fg, pad, mode="constant", constant_values=False)

    if operation == 'dilate':
        if not fg.any():
            return np.asarray(vector, dtype=original_dtype)
        # EDT: distance to nearest pixel with value 0. Mark solid as 0, void as 1.
        dt = distance_transform_edt(np.where(padded_fg, 0, 1))
        result = dt <= strength
        result = result[pad:-pad]
    elif operation == 'erode':
        if not fg.any():
            return np.asarray(vector, dtype=original_dtype)
        # EDT: distance to nearest void (0); solid=1, void=0.
        dt = distance_transform_edt(np.where(padded_fg, 1, 0))
        result = padded_fg & (dt > strength)
        result = result[pad:-pad]
    else:
        raise ValueError("Operation must be either 'erode' or 'dilate'.")

    return result.astype(original_dtype)

def apply_thermal_imperfection(
    binary_profile, 
    dt_max, 
    cte_membrane, 
    cte_grating,
    center_index=None, 
    profile_type='gaussian',
    sigma_fraction=0.25
):
    """
    Simulates heating on a diffractive grating on a distinct membrane substrate.
    Decouples position shifting (membrane) from feature widening (grating).
    """
    n_points = len(binary_profile)
    grid = np.arange(n_points)

    thickness_scale_membrane = 1.0 + (cte_membrane * dt_max)
    thickness_scale_grating = 1.0 + (cte_grating * dt_max)
    
    if center_index is None:
        center_index = n_points // 2

    if profile_type == 'uniform':
        delta_t = np.full_like(grid, dt_max, dtype=float)
    elif profile_type == 'gaussian':
        sigma = n_points * sigma_fraction
        delta_t = dt_max * np.exp(-0.5 * ((grid - center_index) / sigma) ** 2)
    else:
        raise ValueError("Unknown profile_type")

    membrane_strain = cte_membrane * delta_t
    
    displacement_map = np.zeros_like(grid, dtype=float)
    
    displacement_map[center_index:] = np.cumsum(membrane_strain[center_index:])
    if center_index > 0:
        displacement_map[:center_index] = -np.cumsum(membrane_strain[:center_index][::-1])[::-1]

    padded = np.pad(binary_profile, (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]  # Indices where 0->1
    ends = np.where(diffs == -1)[0]   # Indices where 1->0
    
    new_profile = np.zeros_like(binary_profile, dtype=float)

    for start, end in zip(starts, ends):
        width_old = end - start
        center_old = start + (width_old / 2.0) - 0.5
        
        idx_approx = int(np.clip(center_old, 0, n_points-1))
        local_temp = delta_t[idx_approx]
        local_shift = displacement_map[idx_approx]
        
        center_new = center_old + local_shift
        
        width_new = width_old * (1 + cte_grating * local_temp)
        
        left_edge = center_new - (width_new / 2.0)
        right_edge = center_new + (width_new / 2.0)
        
        start_pixel = int(max(0, np.floor(left_edge)))
        end_pixel = int(min(n_points, np.ceil(right_edge)))
        
        for i in range(start_pixel, end_pixel):
            pixel_left = i
            pixel_right = i + 1
            
            overlap_start = max(pixel_left, left_edge)
            overlap_end = min(pixel_right, right_edge)
            
            fill = max(0.0, overlap_end - overlap_start)
            
            new_profile[i] += fill

    new_profile = np.clip(new_profile, 0.0, 1.0)
    
    return new_profile, thickness_scale_membrane, thickness_scale_grating