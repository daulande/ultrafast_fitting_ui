# fitting_strategies/erf_conv_monoexp_strategy.py
import numpy as np
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import lstsq
from typing import List, Tuple, Optional, Dict
import traceback

from .base_fitting_strategy import FittingStrategy

class ErfConvMonoexpStrategy(FittingStrategy):
    """
    策略实现：用户定义的单指数衰减与高斯IRF卷积模型。
    """
    def __init__(self):
        self.debug = False

    def _print_debug(self, msg):
        if self.debug:
            print(f"[STRATEGY-ERF-MONO-DEBUG] {msg}")

    @property
    def name(self) -> str:
        return "Erf Convolved Monoexponential"

    @property
    def num_parameters(self) -> int:
        return 5

    def get_parameter_names(self) -> List[str]:
        return ["a1", "tau1", "c", "w", "t0_model"]

    def model_function(self, t: np.ndarray, *params) -> np.ndarray:
        """
        模型函数。
        t: 原始时间轴数据。
        *params: a1, tau1, c, w, t0_model
        """
        a1, tau1, c, w, t0_model = params

        _tau1, _w = tau1, w
        if not np.isfinite(_tau1) or _tau1 <= 1e-9: _tau1 = 1e-9
        if not np.isfinite(_w) or _w <= 1e-9: _w = 1e-9

        decay_t = t - t0_model
        
        # Term 1
        exp_decay1 = np.exp(-np.clip(decay_t / _tau1, -700, 700))
        arg_erf1 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau1)
        term1 = a1 * exp_decay1 * (1 + erf(np.clip(arg_erf1, -30, 30)))
        term1 = np.where(decay_t < -5 * _w, 0, term1)

        # Constant offset term
        arg_erf_c = 1.81 * (decay_t / _w)
        term_c = c * (1 + erf(np.clip(arg_erf_c, -30, 30)))
        term_c = np.where(decay_t < -5 * _w, 0, term_c)

        result = term1 + term_c
        
        if not np.all(np.isfinite(result)):
            finite_vals = result[np.isfinite(result)]
            max_val = np.max(finite_vals) if len(finite_vals) > 0 else 0
            min_val = np.min(finite_vals) if len(finite_vals) > 0 else 0
            result = np.nan_to_num(result, nan=0.0, posinf=max_val, neginf=min_val)
        return result

    def estimate_initial_parameters(self,
                                    time_original_slice: np.ndarray,
                                    data_slice_processed: np.ndarray,
                                    data_slice_smooth_for_est: np.ndarray,
                                    t_context_zero_gui: Optional[float],
                                    fixed_params: Optional[Dict[int, float]] = None) -> List[List[float]]:
        
        self._print_debug(f"Estimating erf_conv_monoexp params...")
        
        fixed_params = fixed_params or {}
        peeling_params_are_fixed = any(key in fixed_params for key in [0, 1])
        
        # --- 高级估算路径 (对数线性拟合) ---
        if t_context_zero_gui is not None and not peeling_params_are_fixed:
            self._print_debug(f"  Attempting advanced estimation path (log-linear fit).")
            
            fixed_c = fixed_params.get(2)
            fixed_w = fixed_params.get(3)
            fixed_t0 = fixed_params.get(4)

            w_guess_adv = fixed_w if fixed_w is not None else 1e-6
            t0_guess_adv = fixed_t0 if fixed_t0 is not None else t_context_zero_gui
            
            t_prime = time_original_slice - t0_guess_adv
            valid_indices = t_prime >= 0
            
            t_eff_prime = t_prime[valid_indices]
            y_eff_orig = data_slice_smooth_for_est[valid_indices]

            if len(y_eff_orig) >= 10:
                if fixed_c is not None:
                    c_guess_adv = fixed_c
                    C_baseline_guess_adv = c_guess_adv * 2.0
                else:
                    num_tail_points = max(5, len(y_eff_orig) // 10)
                    C_baseline_guess_adv = np.mean(y_eff_orig[-num_tail_points:])
                    c_guess_adv = C_baseline_guess_adv / 2.0
                
                y_decay_eff_adv = y_eff_orig - C_baseline_guess_adv

                advanced_guess = self._estimate_decay_params_loglinear(
                    t_eff_prime, y_decay_eff_adv, t0_guess_adv, w_guess_adv, c_guess_adv, fixed_params
                )

                if advanced_guess:
                    self._print_debug(f"  Advanced estimator (log-linear) path SUCCEEDED.")
                    return [advanced_guess]
                else:
                    self._print_debug("  Advanced estimator (log-linear) path did not yield results. Falling to broader heuristic.")

        # --- 通用启发式估算路径 (作为备用或默认) ---
        self._print_debug(f"  Using broader heuristic estimation path.")
        
        fixed_a1, fixed_tau1, fixed_c, fixed_w, fixed_t0 = (fixed_params.get(i) for i in range(5))

        if fixed_t0 is not None: t0_model_guess = fixed_t0
        else:
            if t_context_zero_gui is not None: t0_model_guess = t_context_zero_gui
            else:
                try:
                    dy_smooth = np.gradient(data_slice_smooth_for_est)
                    peak_dy_idx = np.argmax(np.abs(dy_smooth))
                    idx_t0_guess = max(0, peak_dy_idx - int(0.05 * len(time_original_slice)))
                    t0_model_guess = time_original_slice[idx_t0_guess]
                except Exception: t0_model_guess = time_original_slice[0]
        
        w_guess_heuristic = fixed_w if fixed_w is not None else (time_original_slice[-1] - time_original_slice[0]) * 0.02
        
        if fixed_c is not None:
            c_guess_heuristic = fixed_c
            y_tail_val = c_guess_heuristic * 2.0
        else:
            y_tail_val = np.mean(data_slice_smooth_for_est[-max(5, len(data_slice_smooth_for_est)//10):])
            c_guess_heuristic = y_tail_val / 2.0

        y_after_t0 = data_slice_smooth_for_est[time_original_slice >= t0_model_guess]
        if len(y_after_t0) == 0: y_after_t0 = data_slice_smooth_for_est
        
        peak_val = y_after_t0[np.argmax(np.abs(y_after_t0 - y_tail_val))]
        remaining_amp = peak_val - y_tail_val
        
        a1_guess = fixed_a1 if fixed_a1 is not None else remaining_amp / 2.0
        
        t_range_decay = time_original_slice[-1] - (t0_model_guess + w_guess_heuristic)
        if t_range_decay <= 1e-9: t_range_decay = (time_original_slice[-1] - time_original_slice[0])
        
        tau1_guess = fixed_tau1 if fixed_tau1 is not None else max(1e-9, t_range_decay * 0.3)

        base_params = [a1_guess, tau1_guess, c_guess_heuristic, w_guess_heuristic, t0_model_guess]
        return [base_params]

    def get_bounds(self, t_data_slice: np.ndarray, y_data_slice: np.ndarray, for_global_opt: bool = False) -> List[Tuple[float, float]]:
        y_min, y_max = (np.min(y_data_slice) if len(y_data_slice)>0 else 0), (np.max(y_data_slice) if len(y_data_slice)>0 else 1)
        y_range = y_max - y_min if y_max > y_min else max(abs(y_max), abs(y_min), 1e-9)
        t_min, t_max = (np.min(t_data_slice) if len(t_data_slice)>0 else 0), (np.max(t_data_slice) if len(t_data_slice)>0 else 1)
        t_range = t_max - t_min if t_max > t_min else 1.0

        amp_bound_abs = (3.0 if for_global_opt else 10.0) * y_range
        tau_min_val = max(t_range * 0.00005, 1e-8)
        tau_max_val = t_range * (10.0 if for_global_opt else 50.0)
        c_abs_max = max(abs(y_min), abs(y_max), y_range) * (1.5 if for_global_opt else 3.0)
        w_min_val = max(t_range * 0.00001, 1e-8)
        w_max_val = t_range * (0.5 if for_global_opt else 1.0)
        t0_expansion = (0.3 if for_global_opt else 0.8) * t_range
        t0_min_bound = t_min - t0_expansion
        t0_max_bound = t_max + t0_expansion

        return [
            (-amp_bound_abs, amp_bound_abs),  # a1
            (tau_min_val, tau_max_val),       # tau1
            (-c_abs_max, c_abs_max),          # c
            (w_min_val, w_max_val),           # w
            (t0_min_bound, t0_max_bound)      # t0_model
        ]

    def check_parameter_validity(self, params: List[float], t_data_for_context: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        if len(params) != self.num_parameters:
            return False, f"参数数量错误, 期望 {self.num_parameters}, 得到 {len(params)}"
        if not all(np.isfinite(p) for p in params):
            return False, f"参数包含非有限值: {params}"
        
        _, tau1, _, w, t0_model = params
        if not (tau1 > 1e-9): return False, f"tau1 无效: {tau1}"
        if not (w > 1e-9): return False, f"w 无效: {w}"
        
        if t_data_for_context is not None and len(t_data_for_context) > 0:
            t_min, t_max = np.min(t_data_for_context), np.max(t_data_for_context)
            t_range = t_max - t_min if t_max > t_min else 1.0
            if not ((t_min - t_range) <= t0_model <= (t_max + t_range)):
                return False, f"t0_model ({t0_model:.2f}) 超出数据范围"

        return True, ""

    def _estimate_decay_params_loglinear(self, t_prime_eff, y_decay_eff, t0_model_known, w_known_small, c_guess, fixed_params):
        """
        通过对数-线性拟合估算单指数衰减参数。
        y = a * exp(-t/tau)  =>  log(y) = log(a) - (1/tau) * t
        """
        self._print_debug(f"  Running log-linear fit for 1 component...")
        
        # 确保衰减数据为正，以便取对数
        y_abs_max = np.max(np.abs(y_decay_eff))
        if y_abs_max < 1e-9: return None
        
        y_sign = np.sign(y_decay_eff[np.argmax(np.abs(y_decay_eff))])
        y_positive = y_decay_eff * y_sign
        
        # 只对有意义的数据点进行拟合
        valid_fit_indices = y_positive > (y_abs_max * 0.05)
        if np.sum(valid_fit_indices) < 5: return None
        
        t_fit = t_prime_eff[valid_fit_indices]
        y_fit = y_positive[valid_fit_indices]
        
        try:
            # 执行对数-线性拟合
            coeffs = np.polyfit(t_fit, np.log(y_fit), 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            if slope >= 0: return None # 斜率必须为负
            
            tau1_est = -1.0 / slope
            a_raw_est = np.exp(intercept) * y_sign
            
            # 最终模型参数的幅度约需要减半
            a1_est = a_raw_est / 2.0
            
            final_guess = [
                fixed_params.get(0, a1_est),
                fixed_params.get(1, tau1_est),
                c_guess, w_known_small, t0_model_known
            ]
            
            is_valid, reason = self.check_parameter_validity(final_guess, t_prime_eff)
            if not is_valid:
                self._print_debug(f"  Log-linear fit result failed validity check: {reason}")
                return None
            
            self._print_debug(f"    Log-linear estimated params (a1,t1): {final_guess[0]:.2f}, {final_guess[1]:.2e}")
            return final_guess

        except Exception as e:
            self._print_debug(f"  Log-linear estimation failed: {e}")
            return None