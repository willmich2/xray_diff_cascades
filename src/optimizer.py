import torch # type: ignore
import torch.nn.functional as F # type: ignore
from .forwardmodels import forward_model_N_elements_mask
from . import console
import nlopt # type: ignore

_LOG = "optimizer"

class TopologyOptimizerTorch(torch.nn.Module):
    def __init__(self, n_elements, filter_radius, mesh_resolution, epsilon, constraint_fac, P,
                 constraint_method='indicator', morph_beta=50.0,
                 constraint_aggregation='max', morph_agg_beta=10.0, objective_function=forward_model_N_elements_mask):
        """
        Topology optimizer with geometric constraints for minimum feature size.
        
        Args:
            n_elements: Number of design variables
            filter_radius: Minimum feature width in pixels (min_feature_size / dx)
            mesh_resolution: Grid spacing for the density-filter kernel (pixel units)
            epsilon: Tolerance for constraint violation
            constraint_fac: Scaling factor for constraints
            P: P-norm exponent for constraint aggregation
            constraint_method: 'indicator' (original) or 'morphological' (direct feature size)
            morph_beta: Sharpness parameter for smooth morphological operations (higher = sharper)
            constraint_aggregation: How to combine per-element violations: 'max', 'smooth_max', or 'sum'.
                Use 'smooth_max' when n_opt_elements is large so all elements get constraint gradients.
            morph_agg_beta: Sharpness for smooth_max aggregation (higher = closer to true max).
        """
        super().__init__()
        self.n = n_elements
        self.r = filter_radius
        self.h = mesh_resolution
        
        # Design variables initialized to 0.5 (float64)
        initial_values = torch.rand(n_elements, dtype=torch.float64)
        # Clamp to ensure we stay within valid bounds [0,1] initially if needed, 
        # though bounds usually handle this.
        self.rho = torch.nn.Parameter(initial_values)
        
        self.beta = 1.0       
        self.eta_i = 0.5      
        self.eta_d = 0.25     
        self.eta_e = 0.75     
        self.c = (self.r / self.h)**4
        self.epsilon = epsilon
        self.constraint_fac = constraint_fac
        self.P = P
        self.objective_function = objective_function
        # Constraint method selection
        self.constraint_method = constraint_method
        self.morph_beta = morph_beta
        
        # Multi-element constraint aggregation (see compute_constraints)
        self.constraint_aggregation = constraint_aggregation
        self.morph_agg_beta = morph_agg_beta
        
        # Morphological SE width in pixels: this is the minimum feature size.
        # Round to the nearest odd integer so 1D opening/closing has a
        # well-defined center (e.g. 50 nm at dx=10 nm -> 5 pixels).
        n_mfs = max(1, int(round(self.r / self.h)))
        self.morph_kernel_size = max(3, n_mfs if n_mfs % 2 == 1 else n_mfs + 1)

        # Density filter is a short smoother only. A filter radius comparable
        # to the morph kernel stacks with it and inflates the realized floor.
        neighbor_count = max(1, self.morph_kernel_size // 4)
        grid_indices = torch.arange(-neighbor_count, neighbor_count + 1, dtype=torch.float64)
        x_coords = grid_indices * self.h
        kernel_weights = F.relu(self.r - torch.abs(x_coords))
        self.kernel = kernel_weights / kernel_weights.sum()
        
    def filter_density(self, x, compute_gradient=True):
        x_input = x.view(1, 1, -1)
        # Ensure kernel is on same device and dtype
        kernel = self.kernel.view(1, 1, -1).to(device=x.device, dtype=torch.float64)
        pad = self.kernel.size(0) // 2
        rho_tilde = F.conv1d(x_input, kernel, padding=pad).view(-1)
        
        if compute_gradient:
            # Central difference for gradient
            # FIX 2: Ensure gradient kernel is float64
            grad_kernel = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float64).view(1, 1, 3).to(x.device)
            grad_rho_tilde = F.conv1d(x_input, grad_kernel, padding=1).view(-1) / self.h
        else:
            grad_rho_tilde = None
        
        return rho_tilde, grad_rho_tilde

    def project_density(self, rho_tilde, threshold=0.5):
        beta_tensor = torch.tensor(self.beta)
        threshold_tensor = torch.tensor(threshold)
        num = torch.tanh(beta_tensor * threshold_tensor) + torch.tanh(beta_tensor * (rho_tilde - threshold_tensor))
        den = torch.tanh(beta_tensor * threshold_tensor) + torch.tanh(beta_tensor * (torch.tensor(1.0) - threshold_tensor)).view(1).repeat(num.shape[0])
        return num / den

    def compute_indicators(self, rho_bar, grad_rho_tilde):
        grad_sq = grad_rho_tilde**2
        exp_term = torch.exp(-self.c * grad_sq)
        I_s = rho_bar * exp_term
        I_v = (1 - rho_bar) * exp_term
        return I_s, I_v

    # =========================================================================
    # Smooth Morphological Operations for Direct Feature Size Enforcement
    # =========================================================================
    
    def smooth_erosion_1d(self, x, kernel_size=None, beta=None):
        """
        Smooth approximation of 1D erosion using soft-min.
        
        Erosion with a flat structuring element returns the minimum value
        within a sliding window. We approximate this with logsumexp:
            soft_min(x) = -logsumexp(-beta * x) / beta
        
        As beta -> inf, this approaches the true minimum.
        
        Args:
            x: 1D tensor of values in [0, 1]
            kernel_size: Size of structuring element (odd integer)
            beta: Sharpness parameter (higher = closer to true min)
            
        Returns:
            Eroded 1D tensor (same size as input)
        """
        if kernel_size is None:
            kernel_size = self.morph_kernel_size
        if beta is None:
            beta = self.morph_beta
            
        x_3d = x.view(1, 1, -1)
        pad = kernel_size // 2
        # Replicate padding preserves boundary values
        x_padded = F.pad(x_3d, (pad, pad), mode='replicate')
        
        # Extract sliding windows: shape (1, 1, L, kernel_size)
        x_unfolded = x_padded.unfold(2, kernel_size, 1)
        
        # Soft-min: -logsumexp(-beta * x) / beta
        # Adding a small constant for numerical stability
        eroded = -torch.logsumexp(-beta * x_unfolded, dim=-1) / beta
        
        return eroded.view(-1)
    
    def smooth_dilation_1d(self, x, kernel_size=None, beta=None):
        """
        Smooth approximation of 1D dilation using soft-max.
        
        Dilation with a flat structuring element returns the maximum value
        within a sliding window. We approximate this with logsumexp:
            soft_max(x) = logsumexp(beta * x) / beta
        
        As beta -> inf, this approaches the true maximum.
        
        Args:
            x: 1D tensor of values in [0, 1]
            kernel_size: Size of structuring element (odd integer)
            beta: Sharpness parameter (higher = closer to true max)
            
        Returns:
            Dilated 1D tensor (same size as input)
        """
        if kernel_size is None:
            kernel_size = self.morph_kernel_size
        if beta is None:
            beta = self.morph_beta
            
        x_3d = x.view(1, 1, -1)
        pad = kernel_size // 2
        x_padded = F.pad(x_3d, (pad, pad), mode='replicate')
        
        # Extract sliding windows: shape (1, 1, L, kernel_size)
        x_unfolded = x_padded.unfold(2, kernel_size, 1)
        
        # Soft-max: logsumexp(beta * x) / beta
        dilated = torch.logsumexp(beta * x_unfolded, dim=-1) / beta
        
        return dilated.view(-1)
    
    def smooth_opening(self, x, kernel_size=None, beta=None):
        """
        Smooth morphological opening: erosion followed by dilation.
        
        Opening removes solid (foreground) features smaller than the 
        structuring element while preserving larger features and boundaries.
        
        Args:
            x: 1D tensor representing the design (values in [0, 1])
            kernel_size: Size of structuring element
            beta: Sharpness parameter
            
        Returns:
            Opened design (small solid features removed)
        """
        eroded = self.smooth_erosion_1d(x, kernel_size, beta)
        opened = self.smooth_dilation_1d(eroded, kernel_size, beta)
        return opened
    
    def smooth_closing(self, x, kernel_size=None, beta=None):
        """
        Smooth morphological closing: dilation followed by erosion.
        
        Closing fills void (background) features smaller than the
        structuring element while preserving larger voids and boundaries.
        
        Args:
            x: 1D tensor representing the design (values in [0, 1])
            kernel_size: Size of structuring element
            beta: Sharpness parameter
            
        Returns:
            Closed design (small void features filled)
        """
        dilated = self.smooth_dilation_1d(x, kernel_size, beta)
        closed = self.smooth_erosion_1d(dilated, kernel_size, beta)
        return closed

    # =========================================================================
    # Indicator-based Constraints (Original Method)
    # =========================================================================

    def _compute_indicator_constraints_single_element(self, x):
        """
        Compute geometric constraints using indicator functions (original method).
        
        This method uses indicator functions to identify solid/void regions and
        penalizes violations based on threshold crossings. It's an indirect
        approach that may allow some small features through.
        """
        rho_tilde, grad_rho_tilde = self.filter_density(x)
        rho_bar = self.project_density(rho_tilde)
        I_s, I_v = self.compute_indicators(rho_bar, grad_rho_tilde)
        
        zero_tensor = torch.tensor(0.0, dtype=torch.float64, device=x.device)
        
        violation_s = torch.minimum(rho_tilde - self.eta_e, zero_tensor)**2
        violation_v = torch.minimum(self.eta_d - rho_tilde, zero_tensor)**2
                
        g_s = torch.norm(I_s * violation_s, p=self.P) - self.epsilon
        g_v = torch.norm(I_v * violation_v, p=self.P) - self.epsilon
        
        return g_s, g_v

    # =========================================================================
    # Morphological Constraints (New Direct Method)
    # =========================================================================
    
    def _compute_morphological_constraints_single_element(self, x):
        """
        Compute geometric constraints using morphological operations (direct method).
        
        This method directly enforces minimum feature sizes by comparing the
        design to its morphologically opened/closed version:
        
        - Opening removes solid features smaller than the structuring element
        - Closing removes void features smaller than the structuring element
        
        If the design differs from its opened version, small solid features exist.
        If the design differs from its closed version, small void features exist.
        
        The constraint is: ||design - morphed_design||_P <= epsilon
        
        Returns:
            g_s: Solid feature constraint (g_s <= 0 means no small solid features)
            g_v: Void feature constraint (g_v <= 0 means no small void features)
        """
        rho_tilde, _ = self.filter_density(x, compute_gradient=False)
        rho_bar = self.project_density(rho_tilde)
        
        # Opening removes small solid features
        # If rho_bar != opened, then small solid features exist
        opened = self.smooth_opening(rho_bar)
        
        # Closing fills small void features  
        # If rho_bar != closed, then small void features exist
        closed = self.smooth_closing(rho_bar)
        
        # Solid constraint: penalize where opening removed material
        # diff_solid is positive where small solid features were removed
        diff_solid = F.relu(rho_bar - opened)
        g_s = torch.norm(diff_solid, p=self.P) - self.epsilon
        
        # Void constraint: penalize where closing added material
        # diff_void is positive where small voids were filled
        diff_void = F.relu(closed - rho_bar)
        g_v = torch.norm(diff_void, p=self.P) - self.epsilon
        
        return g_s, g_v

    # =========================================================================
    # Unified Constraint Interface
    # =========================================================================

    def _compute_constraints_single_element(self, x):
        """Dispatch to the selected constraint method."""
        if self.constraint_method == 'morphological':
            return self._compute_morphological_constraints_single_element(x)
        else:  # 'indicator' (default)
            return self._compute_indicator_constraints_single_element(x)

    def compute_constraints(self, x, n_opt_elements=1):
        """
        Compute geometric constraints. When n_opt_elements > 1, parameters are split
        by element (matching forward_model_N_elements_mask) and constraints are
        evaluated per element. Violations are combined using constraint_aggregation:
        'max' (only worst element gets gradient), 'smooth_max' (all elements get
        gradient; use for many elements), or 'sum' (total violation).
        """
        if n_opt_elements <= 1:
            return self._compute_constraints_single_element(x)
        
        N = n_opt_elements
        if x.shape[0] % N != 0:
            raise ValueError(
                f"Parameter vector length ({x.shape[0]}) must be divisible by "
                f"number of elements ({N})"
            )
        x_part_size = x.shape[0] // N
        g_s_list = []
        g_v_list = []
        for i in range(N):
            x_part = x[i * x_part_size:(i + 1) * x_part_size]
            g_s_i, g_v_i = self._compute_constraints_single_element(x_part)
            g_s_list.append(g_s_i)
            g_v_list.append(g_v_i)
        
        g_s_stack = torch.stack(g_s_list)
        g_v_stack = torch.stack(g_v_list)
        
        agg = self.constraint_aggregation
        beta_agg = self.morph_agg_beta
        
        if agg == 'max':
            g_s = torch.max(g_s_stack)
            g_v = torch.max(g_v_stack)
        elif agg == 'smooth_max':
            # Smooth maximum: (1/beta)*logsumexp(beta * x) -> approximates max, gradients to all
            g_s = torch.logsumexp(beta_agg * g_s_stack, dim=0) / beta_agg
            g_v = torch.logsumexp(beta_agg * g_v_stack, dim=0) / beta_agg
        elif agg == 'sum':
            g_s = torch.sum(g_s_stack)
            g_v = torch.sum(g_v_stack)
        else:
            raise ValueError(
                f"constraint_aggregation must be 'max', 'smooth_max', or 'sum', got {agg!r}"
            )
        
        return g_s, g_v

    def compute_objective(self, x, sim_params, args):

        # 1. Filter
        rho_tilde, _ = self.filter_density(x, compute_gradient=False)
        
        # 2. Project
        rho_bar = self.project_density(rho_tilde)

        return self.objective_function(rho_bar, sim_params, *args)

    def compute_robust_objective(self, x, sim_params, args):

        # 1. Filter once
        rho_tilde, _ = self.filter_density(x)
        
        # 2. Generate Three Physical Realizations
        
        # A. Dilated (Thicker): Project at eta_d (0.25)
        # Any value > 0.25 becomes solid.
        rho_dilated = self.project_density(rho_tilde, threshold=self.eta_d)
        obj_dilated, intensity_dilated = forward_model_N_elements_mask(rho_dilated, sim_params, *args)
        
        # B. Nominal (Standard): Project at eta_i (0.5)
        rho_nominal = self.project_density(rho_tilde, threshold=self.eta_i)
        obj_nominal, intensity_nominal = forward_model_N_elements_mask(rho_nominal, sim_params, *args)
        
        # C. Eroded (Thinner): Project at eta_e (0.75)
        # Only values > 0.75 stay solid. Thin features disappear here!
        rho_eroded = self.project_density(rho_tilde, threshold=self.eta_e)
        obj_eroded, intensity_eroded = forward_model_N_elements_mask(rho_eroded, sim_params, *args)
        
        # 3. Min-Max Strategy
        # We want to minimize the LOSS, so the "Worst Case" is the Maximum Loss.
        # This forces the optimizer to ensure even the Eroded design matches the target.
        worst_case_obj = torch.min(torch.stack([obj_dilated, obj_nominal, obj_eroded]))
        worst_case_index = torch.argmin(torch.stack([obj_dilated, obj_nominal, obj_eroded]))
        worst_case_intensity = torch.stack([intensity_dilated, intensity_nominal, intensity_eroded])[worst_case_index]

        return worst_case_obj, worst_case_intensity

def create_tracking_objective(
    model,
    obj_list,
    intensity_list,
    extra_list,
    sim_params,
    forward_model_args,
    track_intermediates: bool = False,
):
    def tracking_objective(x, grad):
        # Ensure input is wrapped as float64
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        
        # Allow objective functions that return either:
        # - (obj, intensity)
        # - (obj, intensity, extra)
        # while remaining robust if only a scalar or a shorter tuple is returned.
        result = model.compute_objective(x_tensor, sim_params, forward_model_args)

        obj = None
        intensity = None
        extra = None

        if isinstance(result, tuple):
            if len(result) >= 1:
                obj = result[0]
            if len(result) >= 2:
                intensity = result[1]
            if len(result) >= 3:
                extra = result[2]
        else:
            obj = result
        
        if grad.size > 0:
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()
            obj.backward()
            grad[:] = x_tensor.grad.detach().numpy()
    
        obj_item = obj.item()
        obj_list.append(obj_item)
        if track_intermediates:
            # IMPORTANT: never store graph-attached CUDA tensors, otherwise each
            # objective call can retain a full autograd graph and leak GPU memory.
            if intensity is not None:
                if isinstance(intensity, torch.Tensor):
                    intensity_list.append(intensity.detach().cpu())
                else:
                    intensity_list.append(intensity)
            if extra is not None:
                if isinstance(extra, torch.Tensor):
                    extra_list.append(extra.detach().cpu())
                else:
                    extra_list.append(extra)
        
        return obj.item()

    return tracking_objective

def make_constraint_solid(model, n_opt_elements=1):
    def constraint_solid_wrapper(x, grad):
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        g_s, _ = model.compute_constraints(x_tensor, n_opt_elements=n_opt_elements)

        g_s_scaled = g_s * model.constraint_fac
        
        if grad.size > 0:
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()
            g_s.backward()
            grad[:] = x_tensor.grad.detach().numpy() * model.constraint_fac
        
        # Corrected: Return scalar, do not take 'result' arg
        return g_s_scaled.item()
    return constraint_solid_wrapper

def make_constraint_void(model, n_opt_elements=1):
    def constraint_void_wrapper(x, grad):
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        _, g_v = model.compute_constraints(x_tensor, n_opt_elements=n_opt_elements)

        g_v_scaled = g_v * model.constraint_fac
        
        if grad.size > 0:
            if x_tensor.grad is not None:
                x_tensor.grad.zero_()
            g_v.backward()
            grad[:] = x_tensor.grad.detach().numpy() * model.constraint_fac
            
        return g_v_scaled.item()
    return constraint_void_wrapper

def run_torch_optimization(sim_params, opt_params, args, objective_function=None):
    if objective_function is None:
        objective_function = forward_model_N_elements_mask
    n_elements = int(sim_params.Nx // 2 * opt_params["Nelem"])
    n_mfs = max(1, int(round(opt_params["min_feature_size"] / sim_params.dx)))
    R = float(n_mfs)
    h = 1.0
    epsilon = opt_params["epsilon"] 
    tolerance = opt_params["tolerance"]
    param_tolerance = opt_params["param_tolerance"]
    max_eval = opt_params["max_eval"]
    constraint_fac = opt_params["constraint_fac"]
    P = opt_params["P"]
    
    # Get constraint method parameters (with defaults for backward compatibility)
    constraint_method = opt_params.get("constraint_method", "indicator")
    morph_beta = opt_params.get("morph_beta", 50.0)
    constraint_aggregation = opt_params.get("constraint_aggregation", "max")
    morph_agg_beta = opt_params.get("morph_agg_beta", 10.0)
    track_intermediates = opt_params.get("track_intermediates", False)
    
    model = TopologyOptimizerTorch(
        n_elements,
        filter_radius=R,
        mesh_resolution=h,
        epsilon=epsilon,
        constraint_fac=constraint_fac,
        P=P,
        constraint_method=constraint_method,
        morph_beta=morph_beta,
        constraint_aggregation=constraint_aggregation,
        morph_agg_beta=morph_agg_beta,
        objective_function=objective_function,
    )

    # Number of optical elements (use opt_params; for multi_z, args[2] is z_distances_set with K entries)
    n_opt_elements = opt_params["Nelem"]

    obj_list = []
    intensity_list = []
    extra_list = []

    beta_values = opt_params.get("beta_schedule", [1, 2, 4, 8, 16, 32])
    min_beta = opt_params["min_beta"]
    current_x = model.rho.detach().numpy().copy()

    console.info(
        _LOG,
        (
            f"starting topology optimization: Nelem={n_opt_elements}, "
            f"n_design_vars={model.n}, min_feature_px={R:.3g}, "
            f"morph_kernel={model.morph_kernel_size}, "
            f"beta_schedule={beta_values}, min_beta={min_beta}, max_eval={max_eval}"
        ),
    )

    for i, beta in enumerate(beta_values):
        stage = f"{i + 1}/{len(beta_values)}"
        constraints_on = beta >= min_beta
        console.info(
            _LOG,
            f"beta stage {stage}: beta={beta}, constraints={'on' if constraints_on else 'off'}",
        )
        model.beta = beta
        
        opt = nlopt.opt(nlopt.LD_MMA, model.n)
        opt.set_lower_bounds(0.0)
        opt.set_upper_bounds(1.0)

        objective = create_tracking_objective(
            model,
            obj_list,
            intensity_list,
            extra_list,
            sim_params,
            args,
            track_intermediates=track_intermediates,
        )
        
        opt.set_max_objective(objective)
        
        if beta >= min_beta: 
            opt.add_inequality_constraint(make_constraint_solid(model, n_opt_elements), tolerance)
            opt.add_inequality_constraint(make_constraint_void(model, n_opt_elements), tolerance)
        
        opt.set_xtol_rel(param_tolerance)
        opt.set_maxeval(max_eval)
        
        try:
            current_x = opt.optimize(current_x)
            max_f = opt.last_optimum_value()
            console.info(_LOG, f"beta stage {stage} finished: objective={max_f:.6g}")
        except nlopt.RoundoffLimited:
            console.warn(_LOG, f"beta stage {stage}: NLopt roundoff limited (continuing)")
        except Exception as e:
            # Do not silently continue on CUDA OOM; free cache and fail fast.
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            console.warn(_LOG, f"beta stage {stage}: optimization error ({type(e).__name__}): {e}")

    final_obj = obj_list[-1] if obj_list else float("nan")
    console.info(
        _LOG,
        f"optimization complete: {len(obj_list)} objective evaluations, final_objective={final_obj:.6g}",
    )
    return current_x, obj_list, intensity_list, extra_list, model