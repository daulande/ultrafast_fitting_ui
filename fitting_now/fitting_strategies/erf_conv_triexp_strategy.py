# fitting_strategies/erf_conv_triexp_strategy.py
import numpy as np
from scipy.special import erf
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import lstsq
from typing import List, Tuple, Optional, Dict
import traceback

from .base_fitting_strategy import FittingStrategy

class ErfConvTriexpStrategy(FittingStrategy):
    """
    策略实现：用户定义的三指数衰减与高斯IRF卷积模型。
    """
    def __init__(self):
        self.debug = False

    def _print_debug(self, msg):
        if self.debug:
            print(f"[STRATEGY-ERF-TRI-DEBUG] {msg}")

    @property
    def name(self) -> str:
        return "Erf Convolved Triexponential"

    @property
    def num_parameters(self) -> int:
        return 9

    def get_parameter_names(self) -> List[str]:
        return ["a1", "tau1", "a2", "tau2", "a3", "tau3", "c", "w", "t0_model"]

    def model_function(self, t: np.ndarray, *params) -> np.ndarray:
        """
        模型函数。
        t: 原始时间轴数据。
        *params: a1, tau1, a2, tau2, a3, tau3, c, w, t0_model
        """
        a1, tau1, a2, tau2, a3, tau3, c, w, t0_model = params

        _tau1, _tau2, _tau3, _w = tau1, tau2, tau3, w
        if not np.isfinite(_tau1) or _tau1 <= 1e-9: _tau1 = 1e-9
        if not np.isfinite(_tau2) or _tau2 <= 1e-9: _tau2 = 1e-9
        if not np.isfinite(_tau3) or _tau3 <= 1e-9: _tau3 = 1e-9
        if not np.isfinite(_w) or _w <= 1e-9: _w = 1e-9

        decay_t = t - t0_model
        
        # Term 1
        exp_decay1 = np.exp(-np.clip(decay_t / _tau1, -700, 700))
        arg_erf1 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau1)
        term1 = a1 * exp_decay1 * (1 + erf(np.clip(arg_erf1, -30, 30)))
        term1 = np.where(decay_t < -5 * _w, 0, term1)

        # Term 2
        exp_decay2 = np.exp(-np.clip(decay_t / _tau2, -700, 700))
        arg_erf2 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau2)
        term2 = a2 * exp_decay2 * (1 + erf(np.clip(arg_erf2, -30, 30)))
        term2 = np.where(decay_t < -5 * _w, 0, term2)
        
        # Term 3
        exp_decay3 = np.exp(-np.clip(decay_t / _tau3, -700, 700))
        arg_erf3 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau3)
        term3 = a3 * exp_decay3 * (1 + erf(np.clip(arg_erf3, -30, 30)))
        term3 = np.where(decay_t < -5 * _w, 0, term3)

        # Constant offset term
        arg_erf_c = 1.81 * (decay_t / _w)
        term_c = c * (1 + erf(np.clip(arg_erf_c, -30, 30)))
        term_c = np.where(decay_t < -5 * _w, 0, term_c)

        result = term1 + term2 + term3 + term_c
        
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
        
        self._print_debug(f"Estimating erf_conv_triexp params. t_original len: {len(time_original_slice)}, y len: {len(data_slice_smooth_for_est)}")
        
        fixed_params = fixed_params or {}
        peeling_params_are_fixed = any(key in fixed_params for key in [0, 1, 2, 3, 4, 5])
        
        # --- 高级估算路径 (Prony 方法) ---
        if t_context_zero_gui is not None and not peeling_params_are_fixed:
            self._print_debug(f"  Attempting advanced estimation path using Prony's method for 3 components.")
            
            fixed_c = fixed_params.get(6)
            fixed_w = fixed_params.get(7)
            fixed_t0 = fixed_params.get(8)

            w_guess_adv = fixed_w if fixed_w is not None else 1e-6
            t0_guess_adv = fixed_t0 if fixed_t0 is not None else t_context_zero_gui
            
            t_prime = time_original_slice - t0_guess_adv
            valid_indices = t_prime >= -5 * w_guess_adv
            if not np.any(valid_indices): valid_indices = t_prime >= 0
            
            t_eff_prime = t_prime[valid_indices]
            y_eff_orig = data_slice_smooth_for_est[valid_indices]

            if len(y_eff_orig) >= 30: # 需要更多点来估算3个分量
                if fixed_c is not None:
                    c_guess_adv = fixed_c
                    C_baseline_guess_adv = c_guess_adv * 2.0
                else:
                    num_tail_points = max(10, len(y_eff_orig) // 10)
                    C_baseline_guess_adv = np.mean(y_eff_orig[-num_tail_points:])
                    c_guess_adv = C_baseline_guess_adv / 2.0
                
                y_decay_eff_adv = y_eff_orig - C_baseline_guess_adv

                advanced_guesses = self._estimate_decay_params_via_prony_tri(
                    t_eff_prime, y_decay_eff_adv, t0_guess_adv, w_guess_adv, c_guess_adv, fixed_params
                )

                if advanced_guesses:
                    self._print_debug(f"  Advanced estimator (Prony) path SUCCEEDED, returning {len(advanced_guesses)} candidates.")
                    return advanced_guesses
                else:
                    self._print_debug("  Advanced estimator (Prony) path did not yield results. Falling to broader heuristic.")

        # --- 通用启发式估算路径 (作为备用或默认) ---
        self._print_debug(f"  Using broader heuristic estimation path.")
        
        # 获取固定参数，注意索引变化
        fixed_a1, fixed_tau1, fixed_a2, fixed_tau2, fixed_a3, fixed_tau3, fixed_c, fixed_w, fixed_t0 = (
            fixed_params.get(i) for i in range(9))

        if fixed_t0 is not None: t0_model_guess = fixed_t0
        else:
            if t_context_zero_gui is not None: t0_model_guess = t_context_zero_gui
            else:
                try:
                    dy_smooth = gaussian_filter1d(np.gradient(gaussian_filter1d(data_slice_smooth_for_est, max(1,len(data_slice_smooth_for_est)//50))), max(1,len(data_slice_smooth_for_est)//50))
                    peak_dy_idx = np.argmax(np.abs(dy_smooth))
                    idx_t0_guess = max(0, peak_dy_idx - int(0.1 * len(time_original_slice)))
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
        peak_val = y_after_t0[np.argmax(np.abs(y_after_t0 - y_tail_val))] if len(y_after_t0) > 0 else 0
        remaining_amp = peak_val - y_tail_val
        
        # 分配3个幅度
        a1_guess = fixed_a1 if fixed_a1 is not None else remaining_amp * 0.4 / 2.0
        a2_guess = fixed_a2 if fixed_a2 is not None else remaining_amp * 0.4 / 2.0
        a3_guess = fixed_a3 if fixed_a3 is not None else remaining_amp * 0.2 / 2.0

        t_range_decay = time_original_slice[-1] - (t0_model_guess + w_guess_heuristic)
        if t_range_decay <= 1e-9: t_range_decay = (time_original_slice[-1] - time_original_slice[0]) / 2.0
        if t_range_decay <= 1e-9: t_range_decay = 1.0

        # 分配3个时间常数，并确保它们不同
        tau1_guess = fixed_tau1 if fixed_tau1 is not None else max(1e-9, t_range_decay * 0.05)
        tau2_guess = fixed_tau2 if fixed_tau2 is not None else max(1e-9, t_range_decay * 0.2)
        tau3_guess = fixed_tau3 if fixed_tau3 is not None else max(1e-9, t_range_decay * 0.8)
        
        taus = sorted(list(set([tau1_guess, tau2_guess, tau3_guess])))
        while len(taus) < 3: taus.append(taus[-1] * 2.0)
        tau1_guess, tau2_guess, tau3_guess = taus[0], taus[1], taus[2]

        base_params = [a1_guess, tau1_guess, a2_guess, tau2_guess, a3_guess, tau3_guess, c_guess_heuristic, w_guess_heuristic, t0_model_guess]
        
        params_list = [base_params]
        # 可以添加更多参数扰动来增加成功率
        return params_list[:10]

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
            (-amp_bound_abs, amp_bound_abs),  # a2
            (tau_min_val, tau_max_val),       # tau2
            (-amp_bound_abs, amp_bound_abs),  # a3
            (tau_min_val, tau_max_val),       # tau3
            (-c_abs_max, c_abs_max),          # c
            (w_min_val, w_max_val),           # w
            (t0_min_bound, t0_max_bound)      # t0_model
        ]

    def check_parameter_validity(self, params: List[float], t_data_for_context: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        if len(params) != self.num_parameters:
            return False, f"参数数量错误, 期望 {self.num_parameters}, 得到 {len(params)}"
        if not all(np.isfinite(p) for p in params):
            return False, f"参数包含非有限值: {params}"
        
        _, tau1, _, tau2, _, tau3, _, w, t0_model = params
        if not (tau1 > 1e-9): return False, f"tau1 无效: {tau1}"
        if not (tau2 > 1e-9): return False, f"tau2 无效: {tau2}"
        if not (tau3 > 1e-9): return False, f"tau3 无效: {tau3}"
        if not (w > 1e-9): return False, f"w 无效: {w}"
        
        if t_data_for_context is not None and len(t_data_for_context) > 0:
            t_min, t_max = np.min(t_data_for_context), np.max(t_data_for_context)
            t_range = t_max - t_min if t_max > t_min else 1.0
            if not ((t_min - t_range) <= t0_model <= (t_max + t_range)):
                return False, f"t0_model ({t0_model:.2f}) 超出数据范围"

        return True, ""

    def _estimate_decay_params_via_prony_tri(self, t_prime_eff, y_decay_eff, t0_model_known, w_known_small, c_guess, fixed_params):
        """
        使用Prony方法估算三指数衰减参数。
        """
        self._print_debug(f"  Running Prony estimation for 3 components...")
        num_components = 3
        if len(t_prime_eff) < 2 * (2 * num_components): return []

        try:
            # 步骤 1: 重采样
            dt = np.min(np.diff(t_prime_eff))
            if dt < 1e-12: return []
            num_samples = int(np.floor((t_prime_eff[-1] - t_prime_eff[0]) / dt))
            if num_samples < 2 * (2 * num_components): return []
            t_uniform = np.linspace(t_prime_eff[0], t_prime_eff[0] + dt * (num_samples - 1), num_samples)
            y_uniform = np.interp(t_uniform, t_prime_eff, y_decay_eff)

            # 步骤 2: 求解特征多项式
            p = num_components
            if len(y_uniform) <= 2 * p: return []
            
            M = np.zeros((len(y_uniform) - 2 * p, p))
            Y = -y_uniform[p : len(y_uniform) - p]
            for i in range(p):
                M[:, i] = y_uniform[p - 1 - i : len(y_uniform) - p - 1 - i]
            
            poly_coeffs_rev, _, _, _ = lstsq(M, Y, rcond=None)
            poly_coeffs = np.concatenate(([1], poly_coeffs_rev))

            # 步骤 3: 求解根并计算 tau
            roots = np.roots(poly_coeffs)
            valid_roots = [r.real for r in roots if 0 < r.real < 1.0 and np.abs(r.imag) < 1e-9]
            if len(valid_roots) != num_components:
                self._print_debug(f"  Prony failed: Found {len(valid_roots)} valid roots, expected {num_components}. Roots: {roots}")
                return []
            
            taus = sorted([-dt / np.log(r) for r in valid_roots])
            tau1_est, tau2_est, tau3_est = taus[0], taus[1], taus[2]

            # 步骤 4: 求解幅度
            basis_1 = np.exp(-t_prime_eff / tau1_est)
            basis_2 = np.exp(-t_prime_eff / tau2_est)
            basis_3 = np.exp(-t_prime_eff / tau3_est)
            basis_matrix = np.vstack([basis_1, basis_2, basis_3]).T
            
            amplitudes, _, _, _ = lstsq(basis_matrix, y_decay_eff, rcond=None)
            a1_est, a2_est, a3_est = amplitudes[0], amplitudes[1], amplitudes[2]
            
            # 最终模型参数的幅度约需要减半
            final_guess = [
                a1_est / 2.0, tau1_est, 
                a2_est / 2.0, tau2_est, 
                a3_est / 2.0, tau3_est, 
                c_guess, w_known_small, t0_model_known
            ]
            
            # 检查固定参数
            final_guess[0] = fixed_params.get(0, final_guess[0])
            final_guess[1] = fixed_params.get(1, final_guess[1])
            final_guess[2] = fixed_params.get(2, final_guess[2])
            final_guess[3] = fixed_params.get(3, final_guess[3])
            final_guess[4] = fixed_params.get(4, final_guess[4])
            final_guess[5] = fixed_params.get(5, final_guess[5])

            is_valid, reason = self.check_parameter_validity(final_guess, t_prime_eff)
            if not is_valid:
                self._print_debug(f"  Prony result failed validity check: {reason}")
                return []
            
            self._print_debug(f"    Prony estimated params (a1,t1,a2,t2,a3,t3): {final_guess[0]:.2f}, {final_guess[1]:.2e}, {final_guess[2]:.2f}, {final_guess[3]:.2e}, {final_guess[4]:.2f}, {final_guess[5]:.2e}")
            return [final_guess]

        except np.linalg.LinAlgError as e:
            self._print_debug(f"  Prony estimation failed with linear algebra error: {e}")
            return []
        except Exception as e_gen:
            self._print_debug(f"  An unexpected error occurred in Prony estimation: {e_gen}")
            # traceback.print_exc()
            return []