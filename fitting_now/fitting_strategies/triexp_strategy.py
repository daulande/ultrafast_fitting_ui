# fitting_strategies/triexp_with_t0_strategy.py
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import lstsq
from typing import List, Tuple, Optional, Dict
import traceback

from .base_fitting_strategy import FittingStrategy

class TriexpWithT0Strategy(FittingStrategy):
    """
    Strategy for a triexponential decay with a t0 parameter, without convolution.
    Model: y(t) = Σ[i=1,3] ai*exp(-(t-t0)/τi) + c  (for t >= t0)
    """
    def __init__(self):
        self.debug = False

    def _print_debug(self, msg):
        if self.debug:
            print(f"[STRATEGY-TRIEXP-T0-DEBUG] {msg}")

    @property
    def name(self) -> str:
        return "Triexponential with t0"

    @property
    def num_parameters(self) -> int:
        return 8

    def get_parameter_names(self) -> List[str]:
        return ["a1", "tau1", "a2", "tau2", "a3", "tau3", "c", "t0_model"]

    def model_function(self, t: np.ndarray, *params) -> np.ndarray:
        a1, tau1, a2, tau2, a3, tau3, c, t0_model = params
        _tau1, _tau2, _tau3 = tau1, tau2, tau3
        if not np.isfinite(_tau1) or _tau1 <= 1e-9: _tau1 = 1e-9
        if not np.isfinite(_tau2) or _tau2 <= 1e-9: _tau2 = 1e-9
        if not np.isfinite(_tau3) or _tau3 <= 1e-9: _tau3 = 1e-9

        decay_t = t - t0_model
        result = np.full_like(t, c, dtype=np.float64)
        mask = decay_t >= 0
        
        t_masked = decay_t[mask]
        term1 = a1 * np.exp(-np.clip(t_masked / _tau1, -700, 700))
        term2 = a2 * np.exp(-np.clip(t_masked / _tau2, -700, 700))
        term3 = a3 * np.exp(-np.clip(t_masked / _tau3, -700, 700))
        
        result[mask] += term1 + term2 + term3
        
        if not np.all(np.isfinite(result)):
            result = np.nan_to_num(result, nan=c)
        return result

    def estimate_initial_parameters(self,
                                    time_original_slice: np.ndarray,
                                    data_slice_processed: np.ndarray,
                                    data_slice_smooth_for_est: np.ndarray,
                                    t_context_zero_gui: Optional[float],
                                    fixed_params: Optional[Dict[int, float]] = None) -> List[List[float]]:
        
        self._print_debug("Estimating triexp_with_t0 params.")
        fixed_params = fixed_params or {}
        peeling_params_are_fixed = any(key in fixed_params for key in [0, 1, 2, 3, 4, 5])

        # --- Advanced Path (Prony's Method) ---
        if t_context_zero_gui is not None and not peeling_params_are_fixed:
            self._print_debug("Attempting advanced estimation path (Prony for 3 components).")
            t0_guess_adv = fixed_params.get(7, t_context_zero_gui)
            
            valid_indices = time_original_slice >= t0_guess_adv
            t_eff = time_original_slice[valid_indices]
            y_eff = data_slice_smooth_for_est[valid_indices]

            if len(y_eff) >= 30:
                c_guess_adv = fixed_params.get(6, np.mean(y_eff[-max(10, len(y_eff)//10):]))
                y_decay_eff = y_eff - c_guess_adv
                
                advanced_guesses = self._estimate_decay_params_via_prony_tri(
                    t_eff - t0_guess_adv, y_decay_eff, t0_guess_adv, c_guess_adv, fixed_params
                )
                if advanced_guesses:
                    self._print_debug("Advanced estimator path SUCCEEDED.")
                    return advanced_guesses

        # --- Heuristic Path ---
        self._print_debug("Using heuristic estimation path.")
        fixed_a1, fixed_tau1, fixed_a2, fixed_tau2, fixed_a3, fixed_tau3, fixed_c, fixed_t0 = (fixed_params.get(i) for i in range(8))

        t0_model_guess = fixed_t0 if fixed_t0 is not None else (t_context_zero_gui or time_original_slice[0])
        c_guess_heuristic = fixed_c if fixed_c is not None else np.mean(data_slice_smooth_for_est[-max(5, len(data_slice_smooth_for_est)//10):])
        
        y_after_t0 = data_slice_smooth_for_est[time_original_slice >= t0_model_guess]
        peak_val = y_after_t0[np.argmax(np.abs(y_after_t0 - c_guess_heuristic))] if len(y_after_t0) > 0 else 0
        remaining_amp = peak_val - c_guess_heuristic

        a1_guess = fixed_a1 if fixed_a1 is not None else remaining_amp * 0.4
        a2_guess = fixed_a2 if fixed_a2 is not None else remaining_amp * 0.4
        a3_guess = fixed_a3 if fixed_a3 is not None else remaining_amp * 0.2

        t_range_decay = time_original_slice[-1] - t0_model_guess
        tau1_guess = fixed_tau1 if fixed_tau1 is not None else max(1e-9, t_range_decay * 0.05)
        tau2_guess = fixed_tau2 if fixed_tau2 is not None else max(1e-9, t_range_decay * 0.2)
        tau3_guess = fixed_tau3 if fixed_tau3 is not None else max(1e-9, t_range_decay * 0.8)

        base_params = [a1_guess, tau1_guess, a2_guess, tau2_guess, a3_guess, tau3_guess, c_guess_heuristic, t0_model_guess]
        return [base_params]

    def get_bounds(self, t_data_slice: np.ndarray, y_data_slice: np.ndarray, for_global_opt: bool = False) -> List[Tuple[float, float]]:
        y_min, y_max = np.min(y_data_slice), np.max(y_data_slice)
        y_range = max(y_max - y_min, 1e-9)
        t_min, t_max = np.min(t_data_slice), np.max(t_data_slice)
        t_range = max(t_max - t_min, 1.0)
        
        amp_bound = 10.0 * y_range
        tau_min_val = max(t_range * 0.0001, 1e-8)
        tau_max_val = t_range * 20.0
        
        return [
            (-amp_bound, amp_bound),      # a1
            (tau_min_val, tau_max_val),   # tau1
            (-amp_bound, amp_bound),      # a2
            (tau_min_val, tau_max_val),   # tau2
            (-amp_bound, amp_bound),      # a3
            (tau_min_val, tau_max_val),   # tau3
            (y_min - y_range, y_max + y_range), # c
            (t_min - 0.2 * t_range, t_max) # t0_model
        ]

    def check_parameter_validity(self, params: List[float], t_data_for_context: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        if len(params) != self.num_parameters: return False, "Incorrect number of parameters."
        if not all(np.isfinite(p) for p in params): return False, "Parameters contain non-finite values."
        _, tau1, _, tau2, _, tau3, _, _ = params
        if tau1 <= 1e-9 or tau2 <= 1e-9 or tau3 <= 1e-9: return False, "Time constants must be positive."
        return True, ""

    def _estimate_decay_params_via_prony_tri(self, t_rel, y_decay, t0_known, c_known, fixed_params):
        try:
            p = 3 # Number of exponential components
            if len(t_rel) < 2 * (2 * p): return []
            
            dt = np.min(np.diff(t_rel))
            if dt < 1e-12: return []
            num_samples = int(np.floor(t_rel[-1] / dt))
            if num_samples < 2 * (2 * p): return []
            t_uniform = np.linspace(0, dt * (num_samples - 1), num_samples)
            y_uniform = np.interp(t_uniform, t_rel, y_decay)

            if len(y_uniform) <= 2 * p: return []
            M = np.zeros((len(y_uniform) - 2 * p, p))
            Y = -y_uniform[p : len(y_uniform) - p]
            for i in range(p):
                M[:, i] = y_uniform[p - 1 - i : len(y_uniform) - p - 1 - i]
            
            poly_coeffs_rev, _, _, _ = lstsq(M, Y, rcond=None)
            poly_coeffs = np.concatenate(([1], poly_coeffs_rev))
            roots = np.roots(poly_coeffs)
            valid_roots = [r.real for r in roots if 0 < r.real < 1.0 and abs(r.imag) < 1e-9]

            if len(valid_roots) != p: return []
            taus = sorted([-dt / np.log(r) for r in valid_roots])
            tau1_est, tau2_est, tau3_est = taus[0], taus[1], taus[2]

            basis1 = np.exp(-t_rel / tau1_est)
            basis2 = np.exp(-t_rel / tau2_est)
            basis3 = np.exp(-t_rel / tau3_est)
            basis_matrix = np.vstack([basis1, basis2, basis3]).T
            amps, _, _, _ = lstsq(basis_matrix, y_decay, rcond=None)
            a1_est, a2_est, a3_est = amps[0], amps[1], amps[2]

            final_guess = [
                fixed_params.get(0, a1_est), fixed_params.get(1, tau1_est),
                fixed_params.get(2, a2_est), fixed_params.get(3, tau2_est),
                fixed_params.get(4, a3_est), fixed_params.get(5, tau3_est),
                c_known, t0_known
            ]
            return [final_guess]
        except Exception:
            return []