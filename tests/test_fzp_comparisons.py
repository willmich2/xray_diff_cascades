"""Exercise both FZP drivers with real CPU metrics and a deterministic optimizer."""
import argparse
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from paper.experiments import fzp_cascade_common as common
from paper.experiments import fzp_cascade_comparison as single
from paper.experiments import fzp_cascade_nelem_sweep as sweep
from paper.sweeps import standard_params as defaults


def deterministic_optimizer(sim, opt, args, **kwargs):
    density = np.tile(np.linspace(0, 1, sim.Nx // 2), opt["Nelem"])
    model = SimpleNamespace(filter_density=lambda x: (x, None))
    return density, [torch.tensor(0.25), torch.tensor(0.5)], [], [], model


class FZPComparisonTests(unittest.TestCase):
    def test_spacing_precedence_in_both_clis(self):
        cases = (
            ([], (defaults.INTER_ELEM_DIST_DEFAULT, common.FZP_INTER_ELEM_DIST_DEFAULT)),
            (["--inter-elem-dist", "0.02"], (0.02, 0.02)),
            (["--inter-elem-dist", "0.02", "--opt-inter-elem-dist", "0.03"], (0.03, 0.02)),
            (["--inter-elem-dist", "0.02", "--fzp-inter-elem-dist", "0.004"], (0.02, 0.004)),
        )
        for module in (single, sweep):
            for options, expected in cases:
                with self.subTest(module=module.__name__, options=options), patch(
                    "sys.argv", [module.__file__, *options]
                ):
                    args = module._parse_args()
                    self.assertEqual(common.resolve_inter_elem_dists(args), expected)
        for invalid in (0, -1, float("nan"), float("inf")):
            args = argparse.Namespace(
                inter_elem_dist=invalid, opt_inter_elem_dist=None, fzp_inter_elem_dist=None,
            )
            with self.assertRaises(ValueError):
                common.resolve_inter_elem_dists(args)

    def test_drivers_agree_and_preserve_schemas(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "DIFFRACTIVE_CASCADES_QUIET": "1", "DIFFRACTIVE_CASCADES_TIMESTAMP_OUTPUTS": "0",
        }), patch.multiple(defaults, NX_DEFAULT=64, CROP_WIDTH_DEFAULT=32), patch.object(
            common, "run_torch_optimization", deterministic_optimizer
        ):
            for module, options in (
                (single, ["--nelem", "2"]),
                (sweep, ["--nelems", "1", "2", "--n-runs", "2"]),
            ):
                with patch("sys.argv", [module.__file__, "--device", "cpu", "--save-dir", directory, *options]):
                    module.main()
                params = np.load(Path(directory) / f"{module.SAVE_PREFIX}_params.npy", allow_pickle=True).item()
                self.assertEqual(params["efficiency_normalization"], "full_pupil")

            with np.load(Path(directory) / f"{single.SAVE_PREFIX}_results.npz") as one, np.load(
                Path(directory) / f"{sweep.SAVE_PREFIX}_results.npz"
            ) as many:
                self.assertEqual(many["opt_rhos"].shape, (2, 2, 64))
                self.assertEqual(many["opt_intensities"].shape, (2, 2, 64))
                np.testing.assert_array_equal(many["nelems"], [1, 2])
                np.testing.assert_array_equal(one["obj_list"], [0.25, 0.5])
                for run_id in (0, 1):
                    np.testing.assert_array_equal(one["rho_bar"], many["opt_rhos"][run_id, 1])
                    for singular, plural in (
                        ("opt_intensity", "opt_intensities"),
                        ("opt_efficiency", "opt_efficiencies"),
                        ("opt_width", "opt_widths"), ("opt_gain", "opt_gains"), ("opt_obj", "opt_objs"),
                    ):
                        np.testing.assert_allclose(one[singular], many[plural][run_id, 1])
                for singular, plural in (
                    ("fzp_cascade_rhos", "fzp_cascade_rhos"),
                    ("fzp_cascade_intensity", "fzp_cascade_intensities"),
                    ("fzp_cascade_efficiency", "fzp_cascade_efficiencies"),
                    ("fzp_cascade_width", "fzp_cascade_widths"),
                    ("fzp_cascade_gain", "fzp_cascade_gains"), ("fzp_cascade_obj", "fzp_cascade_objs"),
                ):
                    np.testing.assert_allclose(one[singular], many[plural][1])
                for key in ("fzp_focal_lengths", "fzp_radii_theory", "fzp_radii_used"):
                    np.testing.assert_allclose(one[key], many[key][1])


if __name__ == "__main__":
    unittest.main()
