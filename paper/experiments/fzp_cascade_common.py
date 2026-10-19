"""Simulation, geometry, optimization, and metrics shared by FZP comparisons."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from paper.sweeps import standard_params as defaults
from src.forwardmodels import (
    forward_model_N_elements_mask,
    forward_model_N_elements_mask_2d_coherent_qdht,
)
from src.inversedesign_utils import fzp_cascade_half_profiles, zp_init
from src.optimizer import run_torch_optimization
from src.simparams import SimParams
from src.util import compute_opt_and_fzp_metrics_2d, focusing_gain, gaussian_energy_spectrum

ASPECT_RATIO = 8.0
FZP_INTER_ELEM_DIST_DEFAULT = 1e-3


def add_spacing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inter-elem-dist", type=float, default=None,
        help="Shared cascade spacing (m); explicit per-cascade options take precedence.",
    )
    parser.add_argument(
        "--opt-inter-elem-dist", type=float, default=None,
        help=f"Optimized-cascade spacing (m; default: {defaults.INTER_ELEM_DIST_DEFAULT:g}).",
    )
    parser.add_argument(
        "--fzp-inter-elem-dist", type=float, default=None,
        help=f"FZP-cascade spacing (m; default: {FZP_INTER_ELEM_DIST_DEFAULT:g}).",
    )


def resolve_inter_elem_dists(args: argparse.Namespace) -> tuple[float, float]:
    shared = args.inter_elem_dist
    opt = args.opt_inter_elem_dist
    fzp = args.fzp_inter_elem_dist
    if opt is None:
        opt = shared if shared is not None else defaults.INTER_ELEM_DIST_DEFAULT
    if fzp is None:
        fzp = shared if shared is not None else FZP_INTER_ELEM_DIST_DEFAULT
    opt, fzp = float(opt), float(fzp)
    if not np.isfinite(opt) or not np.isfinite(fzp) or opt <= 0 or fzp <= 0:
        raise ValueError("inter-element distances must be finite and > 0")
    return opt, fzp


def coinciding_focal_lengths(nelem: int, f: float, inter_elem_dist: float) -> list[float]:
    return [f + (nelem - 1 - i) * inter_elem_dist for i in range(nelem)]


@dataclass
class FZPComparison:
    sim_params: SimParams
    elem_params: dict
    focusing_mask: torch.Tensor
    opt_params: dict
    opt_inter_elem_dist: float
    fzp_inter_elem_dist: float
    f: float
    focusing_threshold: float
    crop_width: int

    @classmethod
    def from_defaults(cls, device: torch.device, opt_dist: float, fzp_dist: float) -> FZPComparison:
        lams, weights = gaussian_energy_spectrum(
            central_energy_ev=defaults.CENTRAL_ENERGY_EV_DEFAULT,
            N=defaults.N_WVL_DEFAULT, bandwidth=defaults.BANDWIDTH_DEFAULT,
            device=device, bandwidth_in_wavelength=False,
        )
        sim = SimParams(
            Ny=1, Nx=defaults.NX_DEFAULT, dx=defaults.DX_DEFAULT,
            device=device, dtype=torch.complex128, lams=lams, weights=weights,
        )
        elem_params = {
            "thickness": ASPECT_RATIO * defaults.MIN_FEATURE_SIZE_DEFAULT,
            "elem_map": defaults.MATERIAL_MAP,
            "gap_map": defaults.GAP_MAP_DEFAULT,
            "membrane_map": defaults.MEMBRANE_MAP_SI3N4,
            "membrane_thickness": defaults.MEMBRANE_THICKNESS_DEFAULT,
            "propagation_method": "angular",
        }
        opt_params = {
            "min_feature_size": defaults.MIN_FEATURE_SIZE_DEFAULT,
            "epsilon": defaults.EPSILON_DEFAULT,
            "tolerance": defaults.TOLERANCE_DEFAULT,
            "param_tolerance": defaults.PARAM_TOLERANCE_DEFAULT,
            "max_eval": defaults.MAX_EVAL_DEFAULT,
            "min_beta": defaults.MIN_BETA_DEFAULT,
            "beta_schedule": list(defaults.BETA_SCHEDULE_DEFAULT),
            "constraint_fac": defaults.CONSTRAINT_FAC_DEFAULT,
            "P": defaults.P_DEFAULT,
            "constraint_method": defaults.CONSTRAINT_METHOD_DEFAULT,
            "constraint_aggregation": defaults.CONSTRAINT_AGGREGATION_DEFAULT,
            "morph_agg_beta": defaults.MORPH_AGG_BETA_DEFAULT,
            "morph_beta": defaults.MORPH_BETA_DEFAULT,
        }
        n_center = int(2 * 1.22 * defaults.MIN_FEATURE_SIZE_DEFAULT / sim.dx)
        mask = torch.zeros(1, sim.Nx, device=device)
        mask[0, sim.Nx // 2 - n_center // 2 : sim.Nx // 2 + n_center // 2] = 1.0
        return cls(
            sim, elem_params, mask, opt_params, opt_dist, fzp_dist,
            defaults.F_DEFAULT, defaults.FOCUSING_THRESHOLD_DEFAULT, defaults.CROP_WIDTH_DEFAULT,
        )

    @property
    def min_feature_size(self) -> float:
        return self.opt_params["min_feature_size"]

    @property
    def r_max(self) -> float:
        return (self.sim_params.Nx // 2) * self.sim_params.dx

    @property
    def lam_center(self) -> torch.Tensor:
        # Preserve the wavelength selection used by the standalone experiments.
        return self.sim_params.lams[self.sim_params.lams.argmax()]

    def forward_args(self, nelem: int, *, fzp: bool = False) -> tuple:
        spacing = self.fzp_inter_elem_dist if fzp else self.opt_inter_elem_dist
        distances = torch.tensor(
            (nelem - 1) * (spacing,) + (self.f,),
            device=self.sim_params.device, dtype=torch.float64,
        )
        return self.elem_params, self.focusing_mask, distances, None

    def optimize(self, nelem: int) -> tuple[torch.Tensor, np.ndarray]:
        raw_design, objectives, _, _, model = run_torch_optimization(
            self.sim_params, {**self.opt_params, "Nelem": nelem}, self.forward_args(nelem),
            objective_function=forward_model_N_elements_mask,
        )
        design = torch.tensor(raw_design, dtype=torch.float64, device=self.sim_params.device)
        filtered, _ = model.filter_density(design)
        density = (filtered > 0.5).to(dtype=torch.float64)
        return density, np.asarray([float(value) for value in objectives], dtype=np.float64)

    def cascade_design(self, nelem: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return fzp_cascade_half_profiles(
            self.lam_center, coinciding_focal_lengths(nelem, self.f, self.fzp_inter_elem_dist),
            self.min_feature_size, self.sim_params, max_radius=self.r_max,
        )

    def evaluate(self, density: torch.Tensor, forward_args: tuple, *, compute_fzp: bool = False) -> dict:
        metrics = compute_opt_and_fzp_metrics_2d(
            density, self.sim_params, forward_args,
            min_feature_size=self.min_feature_size,
            focusing_threshold=self.focusing_threshold, crop_width=self.crop_width,
            forward_model_1d=forward_model_N_elements_mask,
            forward_model_2d=forward_model_N_elements_mask_2d_coherent_qdht,
            zp_init_func=zp_init, compute_fzp=compute_fzp,
            efficiency_normalization="full_pupil",
        )
        metrics["opt_gain"] = focusing_gain(np.asarray(metrics["opt_intensity_1d"]), self.focusing_threshold)
        if compute_fzp:
            metrics["fzp_gain"] = focusing_gain(np.asarray(metrics["fzp_intensity_1d"]), self.focusing_threshold)
        return metrics

    def single_fzp(self) -> tuple[torch.Tensor, dict]:
        profile = zp_init(self.lam_center, self.f, self.min_feature_size, 1, self.sim_params)
        design = torch.as_tensor(profile, dtype=torch.float64, device=self.sim_params.device)
        elem, mask, distances, _ = self.forward_args(1)
        return design, self.evaluate(design, (elem, mask, distances, ((0.0, 0.0),)))

    def metadata(self) -> dict:
        """Common saved parameters; callers add element counts and run counts."""
        return {
            "Nx": int(self.sim_params.Nx), "dx": float(self.sim_params.dx),
            "N_wvl": int(self.sim_params.lams.numel()),
            "central_energy_ev": float(defaults.CENTRAL_ENERGY_EV_DEFAULT),
            "bandwidth": float(defaults.BANDWIDTH_DEFAULT),
            "min_feature_size": float(self.min_feature_size), "f": float(self.f),
            "inter_elem_dist": float(self.opt_inter_elem_dist),
            "opt_inter_elem_dist": float(self.opt_inter_elem_dist),
            "fzp_inter_elem_dist": float(self.fzp_inter_elem_dist),
            "membrane_thickness": float(self.elem_params["membrane_thickness"]),
            "element_thickness": float(self.elem_params["thickness"]),
            "aspect_ratio": float(ASPECT_RATIO), "r_max": float(self.r_max),
            "material": defaults.MATERIAL_DEFAULT,
            "focusing_threshold": float(self.focusing_threshold), "crop_width": int(self.crop_width),
            "optimizer": "run_torch_optimization", "metric_model_2d": "coherent_2d_qdht",
            "efficiency_normalization": "full_pupil",
            "opt_params": {key: value for key, value in self.opt_params.items() if key != "min_feature_size"},
        }
