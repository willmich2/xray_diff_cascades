import torch # type: ignore
import numpy as np # type: ignore
from typing import Callable, Sequence
from .simparams import SimParams
from .elements import ZonePlate
import copy

def create_tracking_objective_function(
    beta: float, 
    forward_model: Callable, 
    sim_params: SimParams, 
    opt_params: dict, 
    forward_model_args: tuple,
    obj_values: list,
    x_values: list,
    intermediate_tensors: list,
    Nthreshold: int = None
    ) -> Callable:
    """
    Create a tracking objective function that logs objective values, parameter vectors, 
    and intermediate tensors during optimization.
    
    Args:
        beta: Beta value for heaviside projection
        forward_model: Forward model function
        sim_params: Simulation parameters
        opt_params: Optimization parameters
        forward_model_args: Additional arguments for forward model
        obj_values: List to track objective values
        x_values: List to track parameter vectors
        intermediate_tensors: List to track intermediate tensors
        Nthreshold: Number of parameters to apply thresholding to (if None, applies to all)
    """
    def tracking_objective_function(x, grad):
        # Convert to PyTorch tensor to call forward model directly
        zero = torch.zeros(0, dtype=sim_params.dtype, device=sim_params.device)
        g = torch.tensor(x, dtype=zero.real.dtype, requires_grad=True, device=sim_params.device)
        
        # Apply same preprocessing as in create_objective_function
        if Nthreshold is not None:
            # only apply preprocessing to parameters with indices below Nthreshold
            g_filtered = g[:Nthreshold]
            g_filtered = density_filtering(g_filtered, opt_params["filter_radius"], sim_params)
            g_thresholded = heaviside_projection(g_filtered, beta=beta)
            g_physical = g_thresholded.view(-1)
            g_physical = torch.cat((g_physical, g[Nthreshold:]), dim=0)
        else:
            g_filtered = density_filtering(g, opt_params["filter_radius"], sim_params)
            g_thresholded = heaviside_projection(g_filtered, beta=beta)
            g_physical = g_thresholded.view(-1)
        
        g_physical.retain_grad()
        
        # Call forward model directly to get tuple result
        forward_result = forward_model(g_physical, sim_params, opt_params, *forward_model_args)
        
        if isinstance(forward_result, tuple):
            obj_val = forward_result[0]
            intermediate_tensor = forward_result[1]
        else:
            obj_val = forward_result
            intermediate_tensor = None
        
        # Handle gradients if needed
        if grad.size > 0:
            obj_val.backward()
            grad[:] = g_physical.grad.detach().cpu().numpy()
        
        # Track values
        obj_values.append(obj_val.item())
        x_values.append(x.copy())
        if intermediate_tensor is not None:
            intermediate_tensors.append(intermediate_tensor.detach().cpu().numpy())
        
        return obj_val.item()
    
    return tracking_objective_function


def zp_init(
        lam: float, 
        f: float, 
        min_feature_size: float, 
        n: int,
        sim_params: SimParams
) -> np.ndarray:
    zone_plate = ZonePlate(
        name = "zp_init", 
        thickness = 1, 
        f = f,
        min_feature_size = min_feature_size, 
        elem_map = [np.array([0, np.inf]), np.array([1., 1.])], 
        gap_map = [np.array([0, np.inf]), np.array([1 + 1j*np.inf, 1 + 1j*np.inf])]
    )

    zp_trans = zone_plate.transmission(lam, lam, sim_params).abs()
    zp_init = torch.where(zp_trans > 0.5, 1.0, 0.0).cpu().reshape(sim_params.Nx)[::n]
    zp_init = zp_init[:zp_init.shape[0]//2].numpy()
    return zp_init


def _scalar_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def fzp_cascade_half_profiles(
    lam: float,
    focal_lengths: Sequence[float],
    min_feature_size: float,
    sim_params: SimParams,
    max_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build intermediate-field FZP half-profiles with coinciding foci.

    Each plate ``i`` is a Fresnel zone plate designed for focal length
    ``focal_lengths[i]`` (distance remaining to a common focus). The most
    upstream plate is patterned out to ``max_radius``. Downstream radii follow
    the first-order cone, ``R_i = max_radius * f_i / f_0``, as in
    intermediate-field FZP stacks. Beyond ``R_i`` the half-profile is open
    (0), i.e. membrane only.

    Returns:
        stacked: concatenated half-profiles, length ``N * (Nx / 2)``
        focal_lengths: ``(N,)`` focal lengths used
        radii_theory: ``(N,)`` unconstrained FZP radii ``λ f_i / (2 Δr_min)``
        radii_used: ``(N,)`` cone-matched radii actually patterned
    """
    lam_f = _scalar_float(lam)
    mfs = _scalar_float(min_feature_size)
    dx = float(sim_params.dx)
    n_half = int(sim_params.Nx) // 2
    if max_radius is None:
        max_radius = n_half * dx
    else:
        max_radius = float(max_radius)

    if isinstance(lam, torch.Tensor):
        lam_t = lam.to(device=sim_params.device)
    else:
        lam_t = torch.tensor(lam_f, dtype=sim_params.x.dtype, device=sim_params.device)

    x_np = sim_params.x.detach().cpu().numpy()
    r_half = np.abs(x_np[:n_half])

    n_plates = len(focal_lengths)
    profiles = []
    f_arr = np.empty(n_plates, dtype=np.float64)
    radii_theory = np.empty(n_plates, dtype=np.float64)
    radii_used = np.empty(n_plates, dtype=np.float64)
    for i, f_i in enumerate(focal_lengths):
        f_i = _scalar_float(f_i)
        f_arr[i] = f_i
        radii_theory[i] = (lam_f * f_i) / (2.0 * mfs)

    if n_plates:
        f_upstream = float(f_arr[0])
        if f_upstream <= 0.0:
            raise ValueError("upstream focal length must be positive")

    for i in range(n_plates):
        r_cone = max_radius * (f_arr[i] / f_upstream)
        r_used = min(r_cone, float(radii_theory[i]), max_radius)
        radii_used[i] = r_used
        profile = np.asarray(
            zp_init(lam_t, f_arr[i], mfs, 1, sim_params),
            dtype=np.float64,
        ).reshape(-1)
        if profile.shape[0] != n_half:
            raise ValueError(
                f"zp_init half-profile length {profile.shape[0]} != Nx/2 ({n_half})"
            )
        profile = profile.copy()
        profile[r_half > r_used] = 0.0
        profiles.append(profile)

    stacked = np.concatenate(profiles, axis=0)
    return stacked, f_arr, radii_theory, radii_used
