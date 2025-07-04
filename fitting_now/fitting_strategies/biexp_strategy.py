# fitting_strategies/biexp_strategy.py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import lstsq
from typing import List, Tuple, Optional, Dict
import traceback

# 从您的项目结构中正确导入基类
from .base_fitting_strategy import FittingStrategy

class BiexpStrategy(FittingStrategy):
    """
    策略实现：双指数衰减模型（无物理约束版本）。
    Model: f(t) = a1 * exp(-(t - t0) / tau1) + a2 * exp(-(t - t0) / tau2) + c
    这个版本允许函数在 t < t0 时向左自然延伸，而不会被截断为平线。
    """
    def __init__(self, debug=False):
        self.debug = debug

    def _print_debug(self, msg):
        if self.debug:
            print(f"[STRATEGY-BIEXP-DEBUG] {msg}")

    @property
    def name(self) -> str:
        return "Biexponential Decay (Extended)"

    @property
    def num_parameters(self) -> int:
        return 6

    def get_parameter_names(self) -> List[str]:
        return ["a1", "tau1", "a2", "tau2", "c", "t0"]

    def model_function(self, t_original: np.ndarray, *params) -> np.ndarray:
        """
        双指数衰减模型函数。
        **核心修改：移除了 np.where 条件，使函数在 t < t0 时自然延伸。**
        """
        a1, tau1, a2, tau2, c, t0 = params
        
        _tau1 = max(tau1, 1e-12)
        _tau2 = max(tau2, 1e-12)

        decay_t = t_original - t0
        
        # 原来的 np.where 条件被移除
        decay_part1 = a1 * np.exp(-decay_t / _tau1)
        decay_part2 = a2 * np.exp(-decay_t / _tau2)
        
        result = decay_part1 + decay_part2 + c
        
        if not np.all(np.isfinite(result)):
            finite_vals = result[np.isfinite(result)]
            max_val = np.max(finite_vals) if len(finite_vals) > 0 else 0
            result = np.nan_to_num(result, nan=c, posinf=max_val, neginf=c)
        return result

    def estimate_initial_parameters(self,
                                    time_original_slice: np.ndarray,
                                    data_slice_processed: np.ndarray,
                                    data_slice_smooth_for_est: np.ndarray,
                                    t_context_zero_gui: Optional[float],
                                    fixed_params: Optional[Dict[int, float]] = None) -> List[List[float]]:
        # 初始参数估计算法保持不变
        self._print_debug(f"Estimating biexp params. t_len: {len(time_original_slice)}, y_len: {len(data_slice_smooth_for_est)}")
        self._print_debug(f"  Inputs: t_context_zero_gui={t_context_zero_gui}, fixed_params={fixed_params}")
        
        fixed_params = fixed_params or {}
        
        peeling_params_are_fixed = any(key in fixed_params for key in [0, 1, 2, 3])
        if peeling_params_are_fixed:
            self._print_debug("  Decay-related param is fixed. Bypassing advanced estimation.")
        
        if t_context_zero_gui is not None and not peeling_params_are_fixed:
            self._print_debug("  Attempting advanced estimation path.")
            t0_guess_adv = fixed_params.get(5, t_context_zero_gui)
            valid_indices = time_original_slice >= t0_guess_adv
            t_eff_prime = time_original_slice[valid_indices] - t0_guess_adv
            y_eff_orig = data_slice_smooth_for_est[valid_indices]

            if len(y_eff_orig) >= 20:
                c_guess_adv = fixed_params.get(4, np.mean(y_eff_orig[-max(5, len(y_eff_orig) // 10):]))
                y_decay_eff_adv = y_eff_orig - c_guess_adv
                self._print_debug(f"    Advanced path: c_guess={c_guess_adv:.4f}")

                advanced_guesses = self._estimate_advanced_prony(
                    t_eff_prime, y_decay_eff_adv, t0_guess_adv, c_guess_adv
                )

                if advanced_guesses:
                    self._print_debug(f"  Advanced estimator path SUCCEEDED, returning {len(advanced_guesses)} candidates.")
                    base_guess = advanced_guesses[0]
                    a1, t1, a2, t2, c, t0 = base_guess
                    perturbed_guess = [a1*0.8, t1*0.8, a2*1.2, t2*1.2, c, t0]
                    advanced_guesses.append(perturbed_guess)
                    return advanced_guesses
                else:
                    self._print_debug("  Advanced estimator path did not yield results. Falling to broader heuristic.")

        self._print_debug(f"  Using broader heuristic estimation path.")
        if 5 in fixed_params:
             t0_model_guess = fixed_params[5]
        else:
            if t_context_zero_gui is not None:
                t0_model_guess = t_context_zero_gui
            else:
                try:
                    dy_smooth = gaussian_filter1d(np.gradient(data_slice_smooth_for_est), 2)
                    peak_dy_idx = np.argmax(np.abs(dy_smooth))
                    idx_t0_guess = max(0, peak_dy_idx - int(0.05 * len(time_original_slice)))
                    t0_model_guess = time_original_slice[idx_t0_guess]
                except Exception:
                    t0_model_guess = time_original_slice[0]

        c_guess_heuristic = fixed_params.get(4, np.mean(data_slice_smooth_for_est[-max(5, len(data_slice_smooth_for_est)//10):]))
        y_decay = data_slice_smooth_for_est - c_guess_heuristic
        y_decay_after_t0 = y_decay[time_original_slice >= t0_model_guess]
        peak_amp = np.max(np.abs(y_decay_after_t0)) if len(y_decay_after_t0) > 0 else np.max(np.abs(y_decay))
        
        a1_guess = fixed_params.get(0, peak_amp * 0.6)
        a2_guess = fixed_params.get(2, peak_amp * 0.4)

        t_range_decay = time_original_slice[-1] - t0_model_guess
        if t_range_decay <= 1e-9: t_range_decay = (time_original_slice[-1] - time_original_slice[0])
        
        tau1_guess = fixed_params.get(1, max(1e-9, t_range_decay * 0.1))
        tau2_guess = fixed_params.get(3, max(1e-9, t_range_decay * 0.5))

        if abs(tau1_guess - tau2_guess) < 1e-12: tau2_guess *= 2.0
        
        base_params = [a1_guess, tau1_guess, a2_guess, tau2_guess, c_guess_heuristic, t0_model_guess]
        
        params_list = [base_params]
        p_var1 = [p * 0.7 if i in [0,1] else p * 1.3 if i in [2,3] else p for i, p in enumerate(base_params)]
        p_var2 = [p * 1.3 if i in [0,1] else p * 0.7 if i in [2,3] else p for i, p in enumerate(base_params)]
        params_list.extend([p_var1, p_var2])
        
        return params_list

    def _estimate_advanced_prony(self, t_eff, y_decay_eff, t0_known, c_guess):
        self._print_debug(f"  Running advanced estimation via Prony's method...")
        if len(t_eff) < 20: 
            self._print_debug("  Prony failed: Not enough effective data points.")
            return []

        try:
            dt = np.mean(np.diff(t_eff))
            if dt <= 1e-12:
                 self._print_debug("  Prony failed: Time step too small or data invalid.")
                 return []
            num_samples = int(np.floor((t_eff[-1] - t_eff[0]) / dt))
            if num_samples < 20:
                self._print_debug("  Prony failed: Not enough samples for resampling.")
                return []
            t_uniform = np.linspace(t_eff[0], t_eff[0] + dt * (num_samples - 1), num_samples)
            y_uniform = np.interp(t_uniform, t_eff, y_decay_eff)

            p = 2
            if len(y_uniform) <= 2 * p: 
                self._print_debug("  Prony failed: Not enough uniform data points to build matrix.")
                return []
            
            Y_vec = -y_uniform[p:2*p]
            hankel_mat = np.array([[y_uniform[i+j] for j in range(p)] for i in range(p)])
            
            try:
                poly_coeffs_rev = np.linalg.solve(hankel_mat, Y_vec)
            except np.linalg.LinAlgError:
                poly_coeffs_rev, _, _, _ = lstsq(hankel_mat, Y_vec, rcond=None)

            poly_coeffs = np.concatenate(([1], poly_coeffs_rev[::-1]))
            
            roots = np.roots(poly_coeffs)
            valid_roots = [r.real for r in roots if 0 < r.real <= 1.0 and abs(r.imag) < 1e-9]
            if len(valid_roots) != 2:
                self._print_debug(f"  Prony failed: Found {len(valid_roots)} valid roots, expected 2. Roots: {roots}")
                return []
            
            taus = sorted([-dt / np.log(r) for r in valid_roots])
            tau1_est, tau2_est = taus[0], taus[1]

            if not (1e-12 < tau1_est < 1e4 and 1e-12 < tau2_est < 1e4 and (tau2_est - tau1_est) > 1e-12):
                self._print_debug(f"  Prony failed: Unphysical tau values estimated. taus={taus}")
                return []

            basis_1 = np.exp(-t_eff / tau1_est)
            basis_2 = np.exp(-t_eff / tau2_est)
            basis_matrix = np.vstack([basis_1, basis_2]).T
            
            amplitudes, _, _, _ = lstsq(basis_matrix, y_decay_eff, rcond=None)
            a1_est, a2_est = amplitudes[0], amplitudes[1]
            
            final_guess = [a1_est, tau1_est, a2_est, tau2_est, c_guess, t0_known]
            
            is_valid, reason = self.check_parameter_validity(final_guess, t_eff + t0_known)
            if not is_valid:
                self._print_debug(f"  Prony result failed validity check: {reason}")
                return []
            
            self._print_debug(f"    Prony estimated params (a1,t1,a2,t2): {final_guess[0]:.2f}, {final_guess[1]:.2e}, {final_guess[2]:.2f}, {final_guess[3]:.2e}")
            return [final_guess]

        except np.linalg.LinAlgError as e:
            self._print_debug(f"  Prony estimation failed with linear algebra error: {e}")
            return []
        except Exception as e_gen:
            self._print_debug(f"  An unexpected error occurred in Prony estimation: {e_gen}")
            self._print_debug(traceback.format_exc())
            return []

    def get_bounds(self, t_data_slice: np.ndarray, y_data_slice: np.ndarray, for_global_opt: bool = False) -> List[Tuple[float, float]]:
        y_min, y_max = (np.min(y_data_slice) if len(y_data_slice)>0 else 0), (np.max(y_data_slice) if len(y_data_slice)>0 else 1)
        y_range = y_max - y_min if y_max > y_min else max(abs(y_max), abs(y_min), 1e-9)
        t_min, t_max = (np.min(t_data_slice) if len(t_data_slice)>0 else 0), (np.max(t_data_slice) if len(t_data_slice)>0 else 1)
        t_range = t_max - t_min if t_max > t_min else 1.0

        amp_bound_abs = (3.0 if for_global_opt else 10.0) * y_range
        tau_min_val = max(t_range * 0.0001, 1e-9)
        tau_max_val = t_range * (10.0 if for_global_opt else 50.0)
        c_abs_max = max(abs(y_min), abs(y_max), y_range) * (1.5 if for_global_opt else 3.0)
        
        t0_expansion = (0.3 if for_global_opt else 0.8) * t_range
        t0_min_bound = t_min - t0_expansion
        t0_max_bound = t_max + t0_expansion

        return [
            (-amp_bound_abs, amp_bound_abs),  # a1
            (tau_min_val, tau_max_val),       # tau1
            (-amp_bound_abs, amp_bound_abs),  # a2
            (tau_min_val, tau_max_val),       # tau2
            (-c_abs_max, c_abs_max),          # c
            (t0_min_bound, t0_max_bound)      # t0
        ]

    def check_parameter_validity(self, params: List[float], t_data_for_context: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        if len(params) != self.num_parameters: 
            return False, f"参数数量错误, 期望 {self.num_parameters}, 得到 {len(params)}"
        if not all(np.isfinite(p) for p in params): 
            return False, f"参数包含非有限值: {params}"
        
        a1, tau1, a2, tau2, c, t0 = params
        
        if not (tau1 > 1e-12): return False, f"tau1 无效: {tau1}"
        if not (tau2 > 1e-12): return False, f"tau2 无效: {tau2}"
        if abs(tau1 - tau2) / max(tau1, tau2) < 0.01: return False, f"tau1 和 tau2 ({tau1:.2e}) 过于接近"
        
        if t_data_for_context is not None and len(t_data_for_context) > 0:
            t_min, t_max = np.min(t_data_for_context), np.max(t_data_for_context)
            t_range = t_max - t_min if t_max > t_min else 1.0
            if not ((t_min - t_range) <= t0 <= (t_max + t_range)):
                return False, f"t0 ({t0:.2f}) 超出数据范围"
        
        return True, ""
