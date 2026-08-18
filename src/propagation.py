import numpy as np # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.checkpoint import checkpoint as grad_checkpoint # type: ignore
from scipy.special import j1 as scipy_j1 # type: ignore
from scipy.special import jn_zeros # type: ignore
from scipy.special import jv as scipy_jv # type: ignore
from .simparams import SimParams
from .util import complex_to_real_dtype

_QDHT_GRID_CACHE: dict[tuple[int, float, str, int | None], tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

_QDHT_L_GRID_CACHE: dict[
    tuple[int, float, int, str, int | None],
    tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _normalize_z_for_batch(
    z: float | torch.Tensor,
    batch_size: int,
    real_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a scalar ``()`` tensor or a length-``batch_size`` vector.

    Existing callers pass a float or a 0-d / 1-element tensor. A 1-d tensor
    with one entry per batch element (wavelength) is also accepted so a
    single angular-spectrum call can send each wavelength to a different
    plane. Other lengths raise.
    """
    if isinstance(z, float):
        z_tensor = torch.tensor(z, dtype=real_dtype, device=device)
    else:
        z_tensor = z.to(dtype=real_dtype, device=device)
    n = int(z_tensor.numel())
    if n == 1:
        return z_tensor.reshape(())
    if n == batch_size:
        return z_tensor.reshape(batch_size)
    raise ValueError(
        f"z must be a scalar or have {batch_size} entries (one per wavelength), "
        f"got shape {tuple(z_tensor.shape)}"
    )


def _z_broadcast(z_tensor: torch.Tensor, extra_dims: int) -> torch.Tensor:
    """Broadcast a scalar or (B,) z tensor to (B, 1, ..., 1)."""
    if z_tensor.ndim == 0:
        return z_tensor
    return z_tensor.view(-1, *([1] * extra_dims))


def _get_qdht_grid(
    N: int,
    R: float,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build/cached QDHT grid tensors:
    - S: (N+1)-th zero of J0
    - alpha: first N zeros of J0
    - j1a: |J1(alpha)|
    - r: radial QDHT sampling points
    """
    key = (N, R, device.type, device.index)
    if key in _QDHT_GRID_CACHE:
        return _QDHT_GRID_CACHE[key]

    zeros = jn_zeros(0, N + 1).astype(np.float64)
    alpha_np = zeros[:N]
    S = float(zeros[N])

    alpha = torch.tensor(alpha_np, dtype=torch.float64, device=device)
    j1a = torch.tensor(np.abs(scipy_j1(alpha_np)), dtype=torch.float64, device=device)
    r = alpha * (R / S)

    _QDHT_GRID_CACHE[key] = (S, alpha, j1a, r)
    return _QDHT_GRID_CACHE[key]


def _complex_interp1d(
    x_src: torch.Tensor,
    y_src: torch.Tensor,
    x_dst: torch.Tensor,
) -> torch.Tensor:
    """
    Linear interpolation on the last axis for complex-valued batched data.
    """
    idx_hi = torch.searchsorted(x_src, x_dst, right=False)
    idx_hi = torch.clamp(idx_hi, 1, x_src.shape[0] - 1)
    idx_lo = idx_hi - 1

    x_lo = x_src[idx_lo]
    x_hi = x_src[idx_hi]
    t = (x_dst - x_lo) / (x_hi - x_lo)

    y_lo = y_src[..., idx_lo]
    y_hi = y_src[..., idx_hi]
    return y_lo + (y_hi - y_lo) * t


def _qdht_apply_M(
    u: torch.Tensor,
    alpha: torch.Tensor,
    j1a: torch.Tensor,
    S: float,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Apply QDHT matrix transform using chunked J0 blocks.
    """
    N = alpha.shape[0]
    w_re = u.real / j1a
    w_im = u.imag / j1a
    out_re = torch.zeros_like(w_re)
    out_im = torch.zeros_like(w_im)

    for i0 in range(0, N, chunk_size):
        i1 = min(i0 + chunk_size, N)
        arg = alpha[i0:i1, None] * (alpha[None, :] / S)
        Y_blk = (
            (2.0 / S)
            * torch.special.bessel_j0(arg)
            / (j1a[i0:i1, None] * j1a[None, :])
        )
        out_re[..., i0:i1] = j1a[i0:i1] * (w_re @ Y_blk.mT)
        out_im[..., i0:i1] = j1a[i0:i1] * (w_im @ Y_blk.mT)

    return torch.complex(out_re, out_im)


def qdht_propagation(
    U: torch.Tensor,
    lam: torch.Tensor,
    z: float | torch.Tensor,
    dx: float,
    device: torch.device,
    pad: bool = True,
    padding: float = 2.0,
    chunk_size: int = 1024,
    checkpoint_qdht: bool = True,
) -> torch.Tensor:
    """
    QDHT propagation path for row geometry (Ny == 1).

    The input/output format matches the rest of the codebase:
    (batch, 1, Nx) fields on a uniform x-grid.
    """
    if U.ndim != 3:
        raise ValueError(f"U must have shape (batch, Ny, Nx), got {tuple(U.shape)}")
    batch, Ny, Nx_original = U.shape
    if Ny != 1:
        raise ValueError("QDHT propagation currently supports Ny == 1 only.")
    if Nx_original % 2 != 0:
        raise ValueError(f"QDHT propagation requires even Nx, got {Nx_original}.")

    if not torch.is_complex(U):
        U = U.to(torch.complex128)

    complex_dtype = U.dtype
    real_dtype = complex_to_real_dtype(complex_dtype)
    lam = lam.to(dtype=real_dtype, device=device).reshape(batch)

    z_tensor = _normalize_z_for_batch(z, batch, real_dtype, device)

    if pad:
        if padding < 1.0:
            raise ValueError(f"padding must be >= 1 (got {padding})")
        if padding > 1.0:
            Nx_target = int(np.ceil(padding * Nx_original))
            # QDHT requires an even grid width.
            if Nx_target % 2 != 0:
                Nx_target += 1
            total_pad = max(0, Nx_target - Nx_original)
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            if total_pad > 0:
                U = F.pad(U, (pad_left, pad_right, 0, 0), mode="constant", value=0)

    _, _, Nx = U.shape

    # Convert the current symmetric Cartesian row into a radial profile (right half),
    # propagate in radial QDHT coordinates, then map back.
    Nrad = Nx // 2
    R = float(Nrad * dx)
    r_uniform = torch.arange(Nrad, dtype=real_dtype, device=device) * dx

    U_radial_uniform = U[:, 0, Nx // 2 :].to(torch.complex128)
    U_out_radial_uniform = torch.zeros_like(U_radial_uniform)

    S, alpha, j1a, r_qdht = _get_qdht_grid(Nrad, R, device)
    nu = alpha / R

    pi = torch.acos(torch.tensor(-1.0, dtype=real_dtype, device=device))
    for b in range(batch):
        u_qdht_in = _complex_interp1d(
            r_uniform,
            U_radial_uniform[b:b+1, :],
            r_qdht,
        ).squeeze(0)

        def _apply_qdht_with_fixed_grid(u_in: torch.Tensor) -> torch.Tensor:
            return _qdht_apply_M(u_in, alpha, j1a, S, chunk_size=chunk_size)

        if checkpoint_qdht and torch.is_grad_enabled() and u_qdht_in.requires_grad:
            spectrum = grad_checkpoint(_apply_qdht_with_fixed_grid, u_qdht_in, use_reentrant=False)
        else:
            spectrum = _apply_qdht_with_fixed_grid(u_qdht_in)
        k0 = 2 * pi / lam[b]
        kz = torch.sqrt((k0**2 - nu**2).to(torch.complex128))
        z_b = z_tensor if z_tensor.ndim == 0 else z_tensor[b]
        propagated_spectrum = spectrum * torch.exp(1j * z_b * kz)
        if checkpoint_qdht and torch.is_grad_enabled() and propagated_spectrum.requires_grad:
            u_qdht_out = grad_checkpoint(_apply_qdht_with_fixed_grid, propagated_spectrum, use_reentrant=False)
        else:
            u_qdht_out = _apply_qdht_with_fixed_grid(propagated_spectrum)

        U_out_radial_uniform[b, :] = _complex_interp1d(
            r_qdht,
            u_qdht_out.unsqueeze(0),
            r_uniform,
        ).squeeze(0)

    U_out = torch.zeros((batch, 1, Nx), dtype=complex_dtype, device=device)
    U_out[:, 0, :Nrad] = torch.flip(U_out_radial_uniform, dims=(-1,))
    U_out[:, 0, Nrad:] = U_out_radial_uniform

    # Crop back to the original width after padded propagation.
    if pad and Nx != Nx_original:
        start_x = (Nx - Nx_original) // 2
        U_out = U_out[..., start_x : start_x + Nx_original]

    return U_out.to(complex_dtype)


def _get_qdht_grid_l(
    N: int,
    R: float,
    l: int,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build/cached QDHT grid tensors for arbitrary azimuthal order l:
    - S: (N+1)-th positive zero of J_l
    - alpha: first N positive zeros of J_l
    - jlp1a: |J_{l+1}(alpha)|
    - r: radial QDHT sampling points (alpha * R / S)
    - Y: precomputed (N, N) symmetric transform matrix with entries
         Y[i, j] = (2 / S) * J_l(alpha[i] * alpha[j] / S) / (jlp1a[i] * jlp1a[j])

    Since torch.special has no general-order Bessel, the transform matrix is
    materialized once per unique (N, R, l, device) via scipy on the host and
    then moved to the target device.
    """
    key = (N, R, int(l), device.type, device.index)
    if key in _QDHT_L_GRID_CACHE:
        return _QDHT_L_GRID_CACHE[key]

    zeros = jn_zeros(int(l), N + 1).astype(np.float64)
    alpha_np = zeros[:N]
    S = float(zeros[N])

    jlp1a_np = np.abs(scipy_jv(int(l) + 1, alpha_np)).astype(np.float64)

    # Full (N, N) Bessel kernel. Symmetric by construction.
    arg_np = np.outer(alpha_np, alpha_np) / S
    bessel_vals_np = scipy_jv(int(l), arg_np).astype(np.float64)
    Y_np = (2.0 / S) * bessel_vals_np / (jlp1a_np[:, None] * jlp1a_np[None, :])

    alpha = torch.tensor(alpha_np, dtype=torch.float64, device=device)
    jlp1a = torch.tensor(jlp1a_np, dtype=torch.float64, device=device)
    r = alpha * (R / S)
    Y = torch.tensor(Y_np, dtype=torch.float64, device=device)

    _QDHT_L_GRID_CACHE[key] = (S, alpha, jlp1a, r, Y)
    return _QDHT_L_GRID_CACHE[key]


def _qdht_apply_M_l(
    u: torch.Tensor,
    jlp1a: torch.Tensor,
    Y: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Apply the cached QDHT transform matrix for arbitrary azimuthal order l.

    Mirrors the output convention of `_qdht_apply_M` but reads Y from the cache
    instead of recomputing it on-the-fly (required because torch has no general
    bessel_j). The matmul is chunked along the output index to bound peak
    memory for very large N.
    """
    N = jlp1a.shape[0]
    w_re = u.real / jlp1a
    w_im = u.imag / jlp1a
    out_re = torch.zeros_like(w_re)
    out_im = torch.zeros_like(w_im)

    for i0 in range(0, N, chunk_size):
        i1 = min(i0 + chunk_size, N)
        Y_blk = Y[i0:i1, :]
        out_re[..., i0:i1] = jlp1a[i0:i1] * (w_re @ Y_blk.mT)
        out_im[..., i0:i1] = jlp1a[i0:i1] * (w_im @ Y_blk.mT)

    return torch.complex(out_re, out_im)


def qdht_order_l_propagation(
    U_radial: torch.Tensor,
    lam: torch.Tensor,
    z: float | torch.Tensor,
    dx: float,
    device: torch.device,
    l: int,
    pad: bool = True,
    padding: float = 2.0,
    chunk_size: int = 1024,
    checkpoint_qdht: bool = False,
) -> torch.Tensor:
    """
    QDHT propagation of a radial field at arbitrary azimuthal order l.

    Intended for the radial slice of a circularly symmetric coherent mode
    LG_{p,|l|}(r, phi) = R_{p,|l|}(r) * exp(i l phi). Through a
    rotation-invariant optical system (radial transmissions + free space) the
    azimuthal phase exp(i l phi) is preserved, so only R(r) needs to be
    evolved via a Hankel transform of order |l|.

    Args:
        U_radial: Complex radial field sampled on the uniform half-grid
                  r = (0..N_rad-1) * dx. Shape (batch, N_rad).
        lam: Wavelengths, shape (batch,).
        z: Propagation distance (scalar float or 0-d tensor).
        dx: Radial pixel size (m).
        device: torch device.
        l: Azimuthal order (non-negative int). l == 0 reduces to the
           conventional order-0 QDHT but uses the generic cached path.
        pad: If True, pad the radial grid by extending the outer edge.
        padding: Expansion factor >= 1. N_rad -> next even >= ceil(padding*N_rad).
        chunk_size: Chunk size along the output index in the matmul.
        checkpoint_qdht: If True and gradients are enabled, wrap each QDHT
                         application in ``torch.utils.checkpoint``.

    Returns:
        Propagated complex radial field on the original uniform r-grid.
    """
    if U_radial.ndim != 2:
        raise ValueError(
            f"U_radial must have shape (batch, N_rad), got {tuple(U_radial.shape)}"
        )
    if not isinstance(l, (int, np.integer)) or int(l) < 0:
        raise ValueError(f"l must be a non-negative integer, got {l}")
    l_int = int(l)

    batch, N_rad_original = U_radial.shape

    if not torch.is_complex(U_radial):
        U_radial = U_radial.to(torch.complex128)

    complex_dtype = U_radial.dtype
    real_dtype = complex_to_real_dtype(complex_dtype)
    lam = lam.to(dtype=real_dtype, device=device).reshape(batch)

    z_tensor = _normalize_z_for_batch(z, batch, real_dtype, device)

    if pad:
        if padding < 1.0:
            raise ValueError(f"padding must be >= 1 (got {padding})")
        if padding > 1.0:
            N_rad_target = int(np.ceil(padding * N_rad_original))
            if N_rad_target % 2 != 0:
                N_rad_target += 1
            total_pad = max(0, N_rad_target - N_rad_original)
            if total_pad > 0:
                # Extend outward in r (no left pad; r=0 stays at index 0).
                U_radial = F.pad(U_radial, (0, total_pad), mode="constant", value=0)

    _, N_rad = U_radial.shape
    R = float(N_rad * dx)
    r_uniform = torch.arange(N_rad, dtype=real_dtype, device=device) * dx

    S, alpha, jlp1a, r_qdht, Y = _get_qdht_grid_l(N_rad, R, l_int, device)
    nu = alpha / R

    pi = torch.acos(torch.tensor(-1.0, dtype=real_dtype, device=device))

    U_out_uniform = torch.zeros_like(U_radial)

    for b in range(batch):
        u_qdht_in = _complex_interp1d(
            r_uniform,
            U_radial[b : b + 1, :],
            r_qdht,
        ).squeeze(0)

        def _apply(u: torch.Tensor) -> torch.Tensor:
            return _qdht_apply_M_l(u, jlp1a, Y, chunk_size=chunk_size)

        if checkpoint_qdht and torch.is_grad_enabled() and u_qdht_in.requires_grad:
            spectrum = grad_checkpoint(_apply, u_qdht_in, use_reentrant=False)
        else:
            spectrum = _apply(u_qdht_in)

        k0 = 2 * pi / lam[b]
        kz = torch.sqrt((k0**2 - nu**2).to(torch.complex128))
        z_b = z_tensor if z_tensor.ndim == 0 else z_tensor[b]
        propagated = spectrum * torch.exp(1j * z_b * kz)

        if checkpoint_qdht and torch.is_grad_enabled() and propagated.requires_grad:
            u_qdht_out = grad_checkpoint(_apply, propagated, use_reentrant=False)
        else:
            u_qdht_out = _apply(propagated)

        U_out_uniform[b, :] = _complex_interp1d(
            r_qdht,
            u_qdht_out.unsqueeze(0),
            r_uniform,
        ).squeeze(0)

    if pad and N_rad != N_rad_original:
        U_out_uniform = U_out_uniform[:, :N_rad_original]

    return U_out_uniform.to(complex_dtype)


def angular_spectrum_propagation(
    U: torch.Tensor,
    lam: torch.Tensor,
    z: float | torch.Tensor,
    dx: float,
    device: torch.device,
    pad: bool = True,
    padding: float = 2.0,
) -> torch.Tensor:
    """
    Performs angular spectrum propagation for a batch of fields.

    Args:
        U (torch.Tensor): Input field, shape (batch, Ny, Nx). Can be real or complex.
        lam (torch.Tensor): Wavelength for each field in the batch, shape (batch,).
        z (float | torch.Tensor): Propagation distance. A float or scalar tensor
                                 sends every batch element the same distance
                                 (supports gradients for learnable distances).
                                 A 1-d tensor of length ``batch`` sends each
                                 wavelength to its own plane.
        dx (float): Pixel size.
        device (torch.device): The torch device to use for calculations.
        pad (bool): If True, zero‑pad the input field in the spatial dimensions
                    to mitigate circular convolution aliasing. Default is True.
        padding (float): Expansion factor > 1 applied to each spatial
                         dimension when padding is enabled. For example,
                         `padding=2` doubles the size, `padding=3` triples it.
                         The y‑dimension is not padded if its original size is 1.
                         Default is 2.0.

    Returns:
        torch.Tensor: The propagated complex field, shape (batch, Ny, Nx).
    """

    if not torch.is_complex(U):
        U = U.to(torch.complex128)

    complex_dtype = U.dtype
    real_dtype = complex_to_real_dtype(complex_dtype)
    # Ensure complex dtype

    # Convert z to a scalar or per-wavelength vector.
    z_tensor = _normalize_z_for_batch(z, U.shape[0], real_dtype, device)

    # Store original dimensions before padding
    Ny_original, Nx_original = U.shape[-2], U.shape[-1]

    # Zero-pad to mitigate circular convolution aliasing
    if pad:
        if padding < 1.0:
            raise ValueError(f"padding must be >= 1 (got {padding})")
        # padding == 1.0 means no expansion (no padding applied)
        if padding == 1.0:
            U_padded = U
        else:
            U_padded = pad_double_both(U, padding=padding)
    else:
        U_padded = U
    batch_size, Ny_padded, Nx_padded = U_padded.shape
    del U

    # Constants and frequency grids
    pi = torch.acos(torch.tensor(-1.0, dtype=real_dtype, device=device))
    lam_b = lam.reshape(batch_size, 1, 1).to(real_dtype)
    k0 = 2 * pi / lam_b  # (batch,1,1)

    # Propagate in Fourier domain (handle 1D row-geometry efficiently)
    if Ny_padded == 1:
        kx = torch.fft.fftfreq(Nx_padded, dx, dtype=real_dtype, device=device) * 2 * pi
        ky = torch.fft.fftfreq(Ny_padded, dx, dtype=real_dtype, device=device) * 2 * pi
        KY, KX = torch.meshgrid(ky, kx, indexing='ij')

        kz = torch.sqrt((k0**2 - (KX**2 + KY**2).to(real_dtype)).to(complex_dtype))  # (batch,Ny,Nx)
        H = torch.exp(1j * _z_broadcast(z_tensor, 2) * kz)

        U_fourier = torch.fft.fft(U_padded, dim=-1)
        U_fourier = U_fourier * H
        U_z_padded = torch.fft.ifft(U_fourier, dim=-1)
    else:
        kx = 2 * pi * torch.fft.fftfreq(Nx_padded, dx, dtype=real_dtype, device=device)   # (Nx,)
        ky = 2 * pi * torch.fft.fftfreq(Ny_padded, dx, dtype=real_dtype, device=device)   # (Ny,)

        kx2 = kx**2                      # (Nx,)
        ky2 = ky**2                      # (Ny,)

        lam_b = lam.reshape(batch_size, 1).to(real_dtype)   # (B,1)
        k0 = 2 * pi / lam_b                                  # (B,1)
        k0_sq = (k0**2).to(real_dtype).squeeze(-1)          # (B,)

        U_fourier = torch.fft.fft2(U_padded)
        del U_padded

        for iy in range(Ny_padded):
            kxy2_row = kx2 + ky2[iy]                        # (Nx,)
            # broadcast to (B, Nx) only, not (B, Ny, Nx)
            inside = (k0_sq[:, None] - kxy2_row[None, :]).to(complex_dtype)
            kz_row = torch.sqrt(inside)                     # (B, Nx)
            phase_row = torch.exp(1j * _z_broadcast(z_tensor, 1) * kz_row)   # (B, Nx)
            U_fourier[:, iy, :] *= phase_row

        del kxy2_row, inside, kz_row, phase_row
        U_z_padded = torch.fft.ifft2(U_fourier)
        del U_fourier

    # Crop back to original size
    if pad:
        U_z = unpad_half_both(U_z_padded, Ny_original, Nx_original)
    else:
        U_z = U_z_padded
    return U_z

def propagate_z(
    U: torch.Tensor,
    z: float | torch.Tensor,
    sim_params: SimParams, 
    method: str = "angular",
    pad: bool = True,
    padding: float = 2.0,
    sequential_wavelengths: bool = False,
    checkpoint_qdht: bool = True,
    ) -> torch.Tensor:
    """
    Propagates a multi-wavelength field U over a distance z.

    Args:
        U (torch.Tensor): Input field, shape (num_wavelengths, Ny, Nx).
        z (float | torch.Tensor): Propagation distance. Scalar (all wavelengths
                                 the same distance) or length-``num_wavelengths``
                                 (one last-hop distance per wavelength).
        sim_params (SimParams): Object containing simulation parameters like
                                wavelengths, pixel size, and device.

    Returns:
        torch.Tensor: The propagated complex field, shape (num_wavelengths, Ny, Nx).
    """
    n_wvl = U.shape[0]
    z_is_per_wavelength = torch.is_tensor(z) and int(z.numel()) == n_wvl
    # The for-loop is replaced with a single, batched call to the
    # modified angular_spectrum_propagation function. The first dimension of U
    # (num_wavelengths) is treated as the batch dimension.
    if sequential_wavelengths:
        U_parts = []
        for w in range(n_wvl):
            U_w = U[w : w + 1]
            lam_w = sim_params.lams[w : w + 1]
            z_w = z.reshape(n_wvl)[w] if z_is_per_wavelength else z
            if method == "angular":
                Uz_w = angular_spectrum_propagation(
                    U_w,
                    lam_w,
                    z_w,
                    sim_params.dx,
                    sim_params.device,
                    pad,
                    padding,
                )
            elif method == "qdht":
                Uz_w = qdht_propagation(
                    U_w,
                    lam_w,
                    z_w,
                    sim_params.dx,
                    sim_params.device,
                    pad=pad,
                    padding=padding,
                    checkpoint_qdht=checkpoint_qdht,
                )
            else:
                raise ValueError(
                    f"Invalid propagation method '{method}'. "
                    "Supported methods are: 'angular', 'qdht'."
                )
            U_parts.append(Uz_w)
        return torch.cat(U_parts, dim=0)

    if method == "angular":
        Uz = angular_spectrum_propagation(
            U,
            sim_params.lams,
            z,
            sim_params.dx,
            sim_params.device,
            pad,
            padding,
        )
        return Uz
    if method == "qdht":
        Uz = qdht_propagation(
            U,
            sim_params.lams,
            z,
            sim_params.dx,
            sim_params.device,
            pad=pad,
            padding=padding,
            checkpoint_qdht=checkpoint_qdht,
        )
        return Uz
    else:
        raise ValueError(
            f"Invalid propagation method '{method}'. "
            "Supported methods are: 'angular', 'qdht'."
        )


def pad_double_width(x: torch.Tensor) -> torch.Tensor:
    """
    Zero‑pad a tensor whose last two dims are (1, W) so the width
    becomes 2 W while the single row stays unchanged.

    Parameters
    ----------
    x : torch.Tensor
        Shape (..., 1, W)

    Returns
    -------
    torch.Tensor
        Shape (..., 1, 2*W) with the input centered horizontally.
    """
    if x.shape[-2] != 1:
        raise ValueError("Row dimension must be 1; only width is padded.")

    W = x.shape[-1]
    pad_left  = W // 2
    pad_right = W - pad_left                        # handles odd W

    # (left, right, top, bottom)
    return F.pad(x, (pad_left, pad_right, 0, 0), mode="constant", value=0)


def unpad_half_width(x: torch.Tensor) -> torch.Tensor:
    """
    Undo `pad_double_width`: crop the central width segment, leaving
    the single row intact.

    Parameters
    ----------
    x : torch.Tensor
        Shape (..., 1, 2*W)

    Returns
    -------
    torch.Tensor
        Shape (..., 1, W)
    """
    if x.shape[-2] != 1 or x.shape[-1] % 2:
        raise ValueError(
            "Input must have shape (..., 1, 2*W) with an even width."
        )

    W = x.shape[-1] // 2
    start = (x.shape[-1] - W) // 2                  # == W//2
    return x[..., :, start : start + W]


def _next_fast_len(n: int) -> int:
    """
    Return the smallest integer >= n whose prime factors are only 2, 3, or 5.
    This approximates SciPy's `next_fast_len` and yields FFT-friendly sizes.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer (got {n})")

    def _factor_out_small_primes(m: int) -> int:
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        return m

    k = n
    while True:
        if _factor_out_small_primes(k) == 1:
            return k
        k += 1


def pad_double_both(x: torch.Tensor, padding: float = 2.0) -> torch.Tensor:
    """
    Zero-pad a tensor in the y and x dimensions (last two dims) so each
    dimension is expanded by a factor `padding`. The input is centered in the
    padded output. If the y dimension is 1, only the x dimension is padded.

    Parameters
    ----------
    x : torch.Tensor
        Shape (..., Ny, Nx)
    padding : float, optional
        Expansion factor >= 1 for each spatial dimension. For example,
        2.0 doubles the size and 3.0 triples it. A value of 1.0 applies no
        padding. The y dimension is left unchanged when Ny == 1. Default is 2.0.

    Returns
    -------
    torch.Tensor
        Padded tensor with shape (..., Ny_padded, Nx_padded), where
        Ny_padded = Ny if Ny == 1 else the smallest FFT‑fast integer >= ceil(padding * Ny),
        Nx_padded = the smallest FFT‑fast integer >= ceil(padding * Nx), with the input centered.
    """
    Ny, Nx = x.shape[-2], x.shape[-1]

    if padding < 1.0:
        raise ValueError(f"padding must be >= 1 (got {padding})")

    # No padding requested
    if padding == 1.0:
        return x

    # Y dimension: keep Ny if it is 1, otherwise scale by padding and snap to FFT‑fast length
    if Ny == 1:
        Ny_padded = Ny
        pad_y_top = 0
        pad_y_bottom = 0
    else:
        Ny_target = int(np.ceil(padding * Ny))
        Ny_padded = _next_fast_len(Ny_target)
        total_pad_y = Ny_padded - Ny
        pad_y_top = total_pad_y // 2
        pad_y_bottom = total_pad_y - pad_y_top

    # X dimension always scaled by padding and snapped to FFT‑fast length
    Nx_target = int(np.ceil(padding * Nx))
    Nx_padded = _next_fast_len(Nx_target)
    total_pad_x = Nx_padded - Nx
    pad_x_left = total_pad_x // 2
    pad_x_right = total_pad_x - pad_x_left

    # F.pad format: (left, right, top, bottom) for last two dimensions
    return F.pad(x, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom), mode="constant", value=0)


def unpad_half_both(x: torch.Tensor, Ny_original: int, Nx_original: int) -> torch.Tensor:
    """
    Undo `pad_double_both`: crop the central segment to restore original size.
    If the original y dimension was 1, it is not cropped in y.

    Parameters
    ----------
    x : torch.Tensor
        Shape (..., Ny, 2*Nx) if Ny_original==1, or (..., 2*Ny, 2*Nx) otherwise
    Ny_original : int
        Original height before padding
    Nx_original : int
        Original width before padding

    Returns
    -------
    torch.Tensor
        Shape (..., Ny_original, Nx_original)
    """
    Ny_padded, Nx_padded = x.shape[-2], x.shape[-1]
    
    # Calculate start indices for cropping
    # If y dimension was 1, don't crop it
    if Ny_original == 1:
        start_y = 0
    else:
        start_y = (Ny_padded - Ny_original) // 2
    
    start_x = (Nx_padded - Nx_original) // 2
    
    return x[..., start_y : start_y + Ny_original, start_x : start_x + Nx_original]


# ----------------------------- 1D Utilities ----------------------------- #

def angular_spectrum_propagation_1d(
    U: torch.Tensor,
    lam: float,
    z: float,
    dx: float,
    device: torch.device,
) -> torch.Tensor:
    """
    1D angular spectrum propagation (matches the NumPy reference behavior).

    Args:
        U: Field at z=0. Shape (N,) or (B, N). Real or complex.
        lam: Wavelength (float, meters).
        z: Propagation distance (meters).
        dx: Sample spacing (meters).
        device: Torch device.

    Returns:
        Propagated complex field with same shape as U.
    """
    if U.ndim == 1:
        U = U.unsqueeze(0)
        squeeze_back = True
    elif U.ndim == 2:
        squeeze_back = False
    else:
        raise ValueError("U must have shape (N,) or (B, N)")

    # Use double precision for spectral calculations to avoid precision issues
    # with very large N and tiny dx.
    U = U.to(torch.complex64)

    B, N = U.shape
    pi64 = torch.acos(torch.tensor(-1.0, dtype=torch.float64, device=device))
    k64 = 2.0 * pi64 / torch.tensor(lam, dtype=torch.float64, device=device)

    kx64 = 2.0 * pi64 * torch.fft.fftfreq(N, d=dx, dtype=torch.float64, device=device)
    kx64 = kx64.unsqueeze(0).expand(B, -1)

    kz64 = torch.sqrt((k64**2 - kx64**2).to(torch.complex128))
    H64 = torch.exp(1j * kz64 * torch.tensor(z, dtype=torch.float64, device=device))

    U_k = torch.fft.fft(U.to(torch.complex128), dim=-1)
    U_k = U_k * H64
    U_z = torch.fft.ifft(U_k, dim=-1).to(torch.complex64)

    if squeeze_back:
        U_z = U_z.squeeze(0)
    return U_z
