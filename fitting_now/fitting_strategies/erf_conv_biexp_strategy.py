# fitting_strategies/erf_conv_biexp_strategy.py
import numpy as np
from scipy.special import erf
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
# 新增导入
from numpy.linalg import lstsq
from typing import List, Tuple, Optional, Dict
import traceback

from .base_fitting_strategy import FittingStrategy

class ErfConvBiexpStrategy(FittingStrategy):
    """
    策略实现：用户定义的双指数衰减与高斯IRF卷积模型。
    """
    def __init__(self):
        self.debug = False

    def _print_debug(self, msg):
        if self.debug:
            print(f"[STRATEGY-ERF-DEBUG] {msg}")

    # ... (name, num_parameters, get_parameter_names, model_function 保持不变) ...
    @property
    def name(self) -> str:
        return "Erf Convolved Biexponential"

    @property
    def num_parameters(self) -> int:
        return 7

    def get_parameter_names(self) -> List[str]:
        return ["a1", "tau1", "a2", "tau2", "c", "w", "t0_model"]

    def model_function(self, t: np.ndarray, *params) -> np.ndarray:
        a1, tau1, a2, tau2, c, w, t0_model = params
        _tau1, _tau2, _w = tau1, tau2, w
        if not np.isfinite(_tau1) or _tau1 <= 1e-9: _tau1 = 1e-9
        if not np.isfinite(_tau2) or _tau2 <= 1e-9: _tau2 = 1e-9
        if not np.isfinite(_w) or _w <= 1e-9: _w = 1e-9
        decay_t = t - t0_model
        exp_decay1 = np.exp(-np.clip(decay_t / _tau1, -700, 700))
        exp_decay2 = np.exp(-np.clip(decay_t / _tau2, -700, 700))
        arg_erf1 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau1)
        term1 = a1 * exp_decay1 * (1 + erf(np.clip(arg_erf1, -30, 30)))
        term1 = np.where(decay_t < -5 * _w, 0, term1)
        arg_erf2 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau2)
        term2 = a2 * exp_decay2 * (1 + erf(np.clip(arg_erf2, -30, 30)))
        term2 = np.where(decay_t < -5 * _w, 0, term2)
        arg_erf_c = 1.81 * (decay_t / _w)
        term_c = c * (1 + erf(np.clip(arg_erf_c, -30, 30)))
        term_c = np.where(decay_t < -5 * _w, 0, term_c)
        result = term1 + term2 + term_c
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
        
        self._print_debug(f"Estimating erf_conv_biexp params. t_original len: {len(time_original_slice)}, y len: {len(data_slice_smooth_for_est)}")
        self._print_debug(f"  Inputs: t_context_zero_gui={t_context_zero_gui}, fixed_params={fixed_params}")
        
        fixed_params = fixed_params or {}
        fixed_a1, fixed_tau1, fixed_a2, fixed_tau2, fixed_c, fixed_w, fixed_t0 = (
            fixed_params.get(i) for i in range(7))

        peeling_params_are_fixed = any(key in fixed_params for key in [0, 1, 2, 3])
        if peeling_params_are_fixed:
            self._print_debug("  Decay-related param is fixed. Bypassing advanced estimation.")

        if t_context_zero_gui is not None and not peeling_params_are_fixed:
            self._print_debug(f"  Attempting advanced estimation path.")
            
            w_guess_adv = fixed_w if fixed_w is not None else 1e-6
            t0_guess_adv = fixed_t0 if fixed_t0 is not None else t_context_zero_gui
            
            t_prime = time_original_slice - t0_guess_adv
            valid_indices = t_prime >= -5 * w_guess_adv
            if not np.any(valid_indices): valid_indices = t_prime >= 0
            
            t_eff_prime = t_prime[valid_indices]
            y_eff_orig = data_slice_smooth_for_est[valid_indices]

            if len(y_eff_orig) >= 20:
                if fixed_c is not None:
                    c_guess_adv = fixed_c
                    C_baseline_guess_adv = c_guess_adv * 2.0
                else:
                    num_tail_points = max(5, len(y_eff_orig) // 10)
                    C_baseline_guess_adv = np.mean(y_eff_orig[-num_tail_points:])
                    c_guess_adv = C_baseline_guess_adv / 2.0
                
                y_decay_eff_adv = y_eff_orig - C_baseline_guess_adv
                self._print_debug(f"    Advanced path: C_baseline_guess={C_baseline_guess_adv:.4f}, c_guess={c_guess_adv:.4f}")

                is_likely_same_sign = self._analyze_decay_sign(t_eff_prime, y_eff_orig, y_decay_eff_adv)
                
                advanced_guesses = []
                # 调用统一的估算函数
                if not is_likely_same_sign:
                    self._print_debug("    Attempting OPPOSITE-sign advanced estimation (using Prony)...")
                    advanced_guesses = self._estimate_opposite_sign_advanced(
                        t_eff_prime, y_decay_eff_adv, t0_guess_adv, w_guess_adv, c_guess_adv
                    )
                else:
                    self._print_debug("    Attempting SAME-sign advanced estimation (using Prony)...")
                    advanced_guesses = self._estimate_same_sign_advanced(
                        t_eff_prime, y_decay_eff_adv, t0_guess_adv, w_guess_adv, c_guess_adv
                    )

                if advanced_guesses:
                    self._print_debug(f"  Advanced estimator path SUCCEEDED, returning {len(advanced_guesses)} candidates.")
                    # 增加一些扰动，为拟合器提供更多选择
                    base_guess = advanced_guesses[0]
                    a1, t1, a2, t2, c, w, t0 = base_guess
                    perturbed_guess = [a1*0.8, t1*0.8, a2*1.2, t2*1.2, c, w, t0]
                    advanced_guesses.append(perturbed_guess)
                    return advanced_guesses
                else:
                    self._print_debug("  Advanced estimator path did not yield results. Falling to broader heuristic.")

        # --- 通用启发式估算路径 (作为备用或默认) ---
        # ... (这部分代码保持不变) ...
        self._print_debug(f"  Using broader heuristic estimation path.")
        is_likely_same_sign = True
        if fixed_t0 is not None: t0_model_guess = fixed_t0
        else:
            if t_context_zero_gui is not None: t0_model_guess = t_context_zero_gui
            else:
                idx_t0_guess = 0
                if len(time_original_slice) > 10:
                    try:
                        dy_smooth = gaussian_filter1d(np.gradient(gaussian_filter1d(data_slice_smooth_for_est, max(1,len(data_slice_smooth_for_est)//50))), max(1,len(data_slice_smooth_for_est)//50))
                        peak_dy_idx = np.argmax(np.abs(dy_smooth))
                        idx_t0_guess = max(0, peak_dy_idx - int(0.1 * len(time_original_slice)))
                    except Exception: idx_t0_guess = 0
                t0_model_guess = time_original_slice[idx_t0_guess]
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
        if fixed_a1 is not None:
            a1_guess = fixed_a1
            a2_guess = fixed_a2 if fixed_a2 is not None else (remaining_amp - a1_guess)
        elif fixed_a2 is not None:
            a2_guess = fixed_a2
            a1_guess = remaining_amp - a2_guess
        else:
            if not is_likely_same_sign:
                a1_sign = np.sign(remaining_amp) if remaining_amp !=0 else -1.0
                a2_sign = -a1_sign
                a1_guess = a1_sign * abs(remaining_amp) * 0.6 / 2.0
                a2_guess = a2_sign * abs(remaining_amp) * 0.4 / 2.0
            else:
                a1_guess = remaining_amp * 0.5 / 2.0
                a2_guess = remaining_amp * 0.5 / 2.0
        t_range_decay = time_original_slice[-1] - (t0_model_guess + w_guess_heuristic)
        if t_range_decay <= 1e-9: t_range_decay = (time_original_slice[-1] - time_original_slice[0]) / 2.0
        if t_range_decay <= 1e-9: t_range_decay = 1.0
        tau1_guess = fixed_tau1 if fixed_tau1 is not None else max(1e-9, t_range_decay * 0.1)
        tau2_guess = fixed_tau2 if fixed_tau2 is not None else max(1e-9, t_range_decay * 0.5)
        if fixed_tau1 is None and fixed_tau2 is None and tau2_guess <= tau1_guess: tau2_guess = tau1_guess * 2.0 + 1e-9
        elif fixed_tau1 is None and fixed_tau2 is not None and tau1_guess >= tau2_guess: tau1_guess = tau2_guess / 2.0
        elif fixed_tau2 is None and fixed_tau1 is not None and tau2_guess <= tau1_guess: tau2_guess = tau1_guess * 2.0
        base_params = [a1_guess, tau1_guess, a2_guess, tau2_guess, c_guess_heuristic, w_guess_heuristic, t0_model_guess]
        params_list = [base_params]
        a1_s, t1_s, a2_s, t2_s, c_s, w_s, t0_s = base_params
        p_var1 = [fixed_a1 if fixed_a1 is not None else a1_s*0.7, fixed_tau1 if fixed_tau1 is not None else t1_s*0.7, fixed_a2 if fixed_a2 is not None else a2_s*1.3, fixed_tau2 if fixed_tau2 is not None else t2_s*1.3, fixed_c if fixed_c is not None else c_s*0.9, fixed_w if fixed_w is not None else w_s*0.8, fixed_t0 if fixed_t0 is not None else t0_s+0.05*w_s]
        p_var2 = [fixed_a1 if fixed_a1 is not None else a1_s*1.3, fixed_tau1 if fixed_tau1 is not None else t1_s*1.3, fixed_a2 if fixed_a2 is not None else a2_s*0.7, fixed_tau2 if fixed_tau2 is not None else t2_s*0.7, fixed_c if fixed_c is not None else c_s*1.1, fixed_w if fixed_w is not None else w_s*1.2, fixed_t0 if fixed_t0 is not None else t0_s-0.05*w_s]
        params_list.extend([p_var1, p_var2])
        return params_list[:10]


    # ... (get_bounds, check_parameter_validity, _analyze_decay_sign 保持不变) ...
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
        return [(-amp_bound_abs, amp_bound_abs), (tau_min_val, tau_max_val), (-amp_bound_abs, amp_bound_abs), (tau_min_val, tau_max_val), (-c_abs_max, c_abs_max), (w_min_val, w_max_val), (t0_min_bound, t0_max_bound)]

    def check_parameter_validity(self, params: List[float], t_data_for_context: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        if len(params) != self.num_parameters: return False, f"参数数量错误, 期望 {self.num_parameters}, 得到 {len(params)}"
        if not all(np.isfinite(p) for p in params): return False, f"参数包含非有限值: {params}"
        _, tau1, _, tau2, _, w, t0_model = params
        if not (tau1 > 1e-9): return False, f"tau1 无效: {tau1}"
        if not (tau2 > 1e-9): return False, f"tau2 无效: {tau2}"
        if not (w > 1e-9): return False, f"w 无效: {w}"
        if t_data_for_context is not None and len(t_data_for_context) > 0:
            t_min, t_max = np.min(t_data_for_context), np.max(t_data_for_context)
            t_range = t_max - t_min if t_max > t_min else 1.0
            if not ((t_min - t_range) <= t0_model <= (t_max + t_range)): return False, f"t0_model ({t0_model:.2f}) 超出数据范围"
        return True, ""

    def _analyze_decay_sign(self, t_eff_prime, y_eff_orig, y_decay_eff_adv) -> bool:
        is_same_sign = True
        if len(t_eff_prime) > 10:
            dy_dt = np.gradient(y_eff_orig, t_eff_prime)
            try:
                win = min(max(7, len(dy_dt) // 7), len(dy_dt) - 1)
                win = win if win % 2 != 0 else win - 1
                poly = min(3, win - 1) if win > 3 else 1
                dy_dt_smooth = savgol_filter(dy_dt, win, poly) if win > poly > 0 else dy_dt
                min_d, max_d = np.min(dy_dt_smooth), np.max(dy_dt_smooth)
                max_abs_d = max(abs(min_d), abs(max_d))
                if max_abs_d > 1e-7 and min_d < -0.2 * max_abs_d and max_d > 0.2 * max_abs_d: is_same_sign = False
            except Exception: pass
        if is_same_sign and len(y_decay_eff_adv) > 0:
            max_abs_decay = np.max(np.abs(y_decay_eff_adv))
            if max_abs_decay > 1e-9:
                pos_peaks = y_decay_eff_adv[y_decay_eff_adv > 0.15 * max_abs_decay]
                neg_peaks = y_decay_eff_adv[y_decay_eff_adv < -0.15 * max_abs_decay]
                if len(pos_peaks) > 0 and len(neg_peaks) > 0: is_same_sign = False
        self._print_debug(f"    Sign Analysis Result: is_likely_same_sign = {is_same_sign}")
        return is_same_sign


    def _estimate_same_sign_advanced(self, t_prime_eff, y_decay_eff, t0_model_known, w_known_small, c_guess):
        self._print_debug(f"  Running advanced estimation via Prony's method...")
        if len(t_prime_eff) < 20: return []

        try:
            # 步骤 1: 将数据重采样至均匀时间步长
            dt = np.min(np.diff(t_prime_eff))
            if dt < 1e-12:
                self._print_debug("  Prony failed: Time step too small or data invalid.")
                return []
            num_samples = int(np.floor((t_prime_eff[-1] - t_prime_eff[0]) / dt))
            if num_samples < 20:
                self._print_debug("  Prony failed: Not enough samples for resampling.")
                return []
            t_uniform = np.linspace(t_prime_eff[0], t_prime_eff[0] + dt * (num_samples - 1), num_samples)
            y_uniform = np.interp(t_uniform, t_prime_eff, y_decay_eff)

            # 步骤 2: 求解特征多项式系数
            p = 2  # 2个指数分量
            if len(y_uniform) <= 2 * p: return [] # 确保有足够的数据点构建矩阵
            
            M = np.zeros((len(y_uniform) - 2 * p, p))
            Y = -y_uniform[p : len(y_uniform) - p]
            for i in range(p):
                M[:, i] = y_uniform[p - 1 - i : len(y_uniform) - p - 1 - i]
            
            poly_coeffs_rev, _, _, _ = lstsq(M, Y, rcond=None)
            poly_coeffs = np.concatenate(([1], poly_coeffs_rev))

            # 步骤 3: 求解根并计算 tau
            roots = np.roots(poly_coeffs)
            valid_roots = [r.real for r in roots if 0 < r.real < 1.0 and np.abs(r.imag) < 1e-9]
            if len(valid_roots) != 2:
                self._print_debug(f"  Prony failed: Found {len(valid_roots)} valid roots, expected 2. Roots: {roots}")
                return []
            
            taus = sorted([-dt / np.log(r) for r in valid_roots])
            tau1_est, tau2_est = taus[0], taus[1]

            if not (1e-12 < tau1_est < 1e3 and 1e-12 < tau2_est < 1e3 and (tau2_est - tau1_est) > 1e-12):
                self._print_debug(f"  Prony failed: Unphysical tau values estimated. taus={taus}")
                return []

            # 步骤 4: 求解幅度 a1, a2
            basis_1 = np.exp(-t_prime_eff / tau1_est)
            basis_2 = np.exp(-t_prime_eff / tau2_est)
            basis_matrix = np.vstack([basis_1, basis_2]).T
            
            amplitudes, _, _, _ = lstsq(basis_matrix, y_decay_eff, rcond=None)
            a1_est, a2_est = amplitudes[0], amplitudes[1]
            
            # 最终模型参数的幅度约需要减半
            final_guess = [a1_est / 2.0, tau1_est, a2_est / 2.0, tau2_est, c_guess, w_known_small, t0_model_known]
            
            is_valid, reason = self.check_parameter_validity(final_guess, t_prime_eff)
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
            # traceback.print_exc()
            return []


    def _estimate_opposite_sign_advanced(self, t_prime_eff, y_decay_eff, t0_model_known, w_known_small, c_guess):
        """
        对于 Prony 方法，同号和异号衰减的处理方式是相同的。
        因此，这个函数直接调用 'same_sign' 的实现即可。
        """
        self._print_debug("  Redirecting opposite-sign estimation to the unified Prony's method.")
        return self._estimate_same_sign_advanced(t_prime_eff, y_decay_eff, t0_model_known, w_known_small, c_guess)