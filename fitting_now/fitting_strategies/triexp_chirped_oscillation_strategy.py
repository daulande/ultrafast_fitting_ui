# fitting_strategies/triexp_chirped_oscillation_strategy.py
import numpy as np
from scipy.special import erf
from scipy.signal import savgol_filter, hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from numpy.linalg import lstsq
from typing import List, Tuple, Optional, Dict
import warnings

from .base_fitting_strategy import FittingStrategy

class TriexpChirpedOscillationStrategy(FittingStrategy):
    """
    策略实现：三指数衰减加啁啾振荡衰减模型（带卷积）
    y(t) = Σ[i=1,3] ai*exp(-(t-t0)/taui)*(1+erf(...)) + C*(1+erf(...)) + A_ph*exp(-(t-t0)/tau_ph)*cos(2π(f0*(t-t0) + 0.5*beta*(t-t0)²) + phi)
    """
    def __init__(self):
        self.debug = False

    def _print_debug(self, msg):
        if self.debug:
            print(f"[STRATEGY-TRIEXP-CHIRP-DEBUG] {msg}")

    @property
    def name(self) -> str:
        return "Triexponential with Chirped Oscillation"

    @property
    def num_parameters(self) -> int:
        return 14  # a1, tau1, a2, tau2, a3, tau3, C, w, t0, A_ph, tau_ph, f0, beta, phi

    def get_parameter_names(self) -> List[str]:
        return ["a1", "tau1", "a2", "tau2", "a3", "tau3", "C", "w", "t0", "A_ph", "tau_ph", "f0", "beta", "phi"]

    def model_function(self, t: np.ndarray, *params) -> np.ndarray:
        """三指数衰减加振荡衰减模型函数（带卷积）"""
        a1, tau1, a2, tau2, a3, tau3, C, w, t0, A_ph, tau_ph, f0, beta, phi = params
        
        # 防止数值溢出
        _tau1 = max(tau1, 1e-9)
        _tau2 = max(tau2, 1e-9)
        _tau3 = max(tau3, 1e-9)
        _tau_ph = max(tau_ph, 1e-9)
        _w = max(w, 1e-9)
        
        # 相对于t0的时间
        decay_t = t - t0
        
        # 三个卷积的指数衰减项
        exp_decay1 = np.exp(-np.clip(decay_t / _tau1, -700, 700))
        arg_erf1 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau1)
        term1 = a1 * exp_decay1 * (1 + erf(np.clip(arg_erf1, -30, 30)))
        term1 = np.where(decay_t < -5 * _w, 0, term1)
        
        exp_decay2 = np.exp(-np.clip(decay_t / _tau2, -700, 700))
        arg_erf2 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau2)
        term2 = a2 * exp_decay2 * (1 + erf(np.clip(arg_erf2, -30, 30)))
        term2 = np.where(decay_t < -5 * _w, 0, term2)
        
        exp_decay3 = np.exp(-np.clip(decay_t / _tau3, -700, 700))
        arg_erf3 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau3)
        term3 = a3 * exp_decay3 * (1 + erf(np.clip(arg_erf3, -30, 30)))
        term3 = np.where(decay_t < -5 * _w, 0, term3)
        
        # 常数项卷积
        arg_erf_c = 1.81 * (decay_t / _w)
        term_c = C * (1 + erf(np.clip(arg_erf_c, -30, 30)))
        term_c = np.where(decay_t < -5 * _w, 0, term_c)
        
        # 振荡衰减项
        phase = 2 * np.pi * (f0 * decay_t + 0.5 * beta * decay_t**2) + phi
        osc_decay = A_ph * np.exp(-np.clip(decay_t / _tau_ph, -700, 700)) * np.cos(phase)
        osc_decay = np.where(decay_t < 0, 0, osc_decay)
        
        result = term1 + term2 + term3 + term_c + osc_decay
        
        # 处理非有限值
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
        
        self._print_debug(f"Estimating triexp_chirped_osc params. t len: {len(time_original_slice)}, y len: {len(data_slice_smooth_for_est)}")
        
        fixed_params = fixed_params or {}
        
        # 如果有足够数据点，尝试高级估算
        if len(time_original_slice) >= 80 and t_context_zero_gui is not None:
            try:
                advanced_guesses = self._advanced_estimation(
                    time_original_slice, data_slice_processed, data_slice_smooth_for_est, 
                    t_context_zero_gui, fixed_params
                )
                if advanced_guesses:
                    self._print_debug(f"Advanced estimation succeeded with {len(advanced_guesses)} guesses")
                    return advanced_guesses
            except Exception as e:
                self._print_debug(f"Advanced estimation failed: {e}")
        
        # 回退到简单启发式估算
        return self._heuristic_estimation(
            time_original_slice, data_slice_processed, data_slice_smooth_for_est, 
            t_context_zero_gui, fixed_params
        )

    def _advanced_estimation(self, t, y_raw, y_smooth, t0_guess, fixed_params):
        """高级参数估算：使用信号处理技术分离各成分"""
        guesses = []
        
        # 估算或使用固定的 w 和 t0
        w_est = fixed_params.get(7, (t[-1] - t[0]) * 0.02)
        t0_est = fixed_params.get(8, t0_guess if t0_guess is not None else t[0])
        
        # Step 1: 估算常数背景 C
        tail_size = max(10, len(y_smooth) // 10)
        C_full = fixed_params.get(6, np.mean(y_smooth[-tail_size:]))
        C_est = C_full / 2.0  # 卷积后的常数项会变成2倍
        
        self._print_debug(f"  Step 1: C estimated as {C_est:.4f}, w={w_est:.4f}, t0={t0_est:.4f}")
        
        # 只分析t0之后的数据
        t_after_t0 = t - t0_est
        valid_idx = t_after_t0 >= -5 * w_est
        if np.sum(valid_idx) < 30:
            valid_idx = t_after_t0 >= 0
        
        t_valid = t[valid_idx]
        y_valid = y_smooth[valid_idx]
        t_prime = t_valid - t0_est
        
        y_no_bg = y_valid - C_full
        
        # Step 2: 使用二阶导数增强振荡成分
        try:
            # 计算平滑的二阶导数
            dy_dt = np.gradient(y_no_bg, t_valid)
            d2y_dt2 = np.gradient(dy_dt, t_valid)
            
            # 平滑处理
            window_size = max(5, len(t_valid) // 20)
            if window_size % 2 == 0:
                window_size += 1
            d2y_smooth = savgol_filter(d2y_dt2, window_size, 3)
            
            # Step 3: 希尔伯特变换提取振荡包络
            analytic_signal = hilbert(d2y_smooth)
            envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            
            # 找到包络的显著部分
            env_threshold = 0.1 * np.max(envelope)
            significant_indices = envelope > env_threshold
            
            if np.sum(significant_indices) > 20:
                # 从包络估算 tau_ph 和 A_ph
                t_sig = t_prime[significant_indices]
                env_sig = envelope[significant_indices]
                
                # 对数变换拟合指数衰减
                log_env = np.log(env_sig + 1e-10)
                poly_coeffs = np.polyfit(t_sig, log_env, 1)
                tau_ph_est = -1 / poly_coeffs[0] if poly_coeffs[0] < 0 else (t[-1] - t[0]) / 3
                A_ph_est = np.exp(poly_coeffs[1]) / (2 * np.pi * 50)**2 / 2.0
                
                # Step 4: 瞬时频率分析
                inst_freq = np.gradient(instantaneous_phase[significant_indices], t_sig) / (2 * np.pi)
                
                # 拟合线性啁啾
                chirp_coeffs = np.polyfit(t_sig, inst_freq, 1)
                beta_est = chirp_coeffs[0]
                f0_est = chirp_coeffs[1]
                
                # 初始相位
                phi_est = instantaneous_phase[significant_indices][0] - 2 * np.pi * (f0_est * t_sig[0] + 0.5 * beta_est * t_sig[0]**2)
                phi_est = np.mod(phi_est, 2 * np.pi)
                
                self._print_debug(f"  Oscillation params: f0={f0_est:.2f}, beta={beta_est:.4f}, tau_ph={tau_ph_est:.4f}")
            else:
                A_ph_est, tau_ph_est = 0.0, (t[-1] - t[0]) / 3
                f0_est, beta_est, phi_est = 10.0, 0.0, 0.0
                
        except Exception as e:
            self._print_debug(f"  Oscillation extraction failed: {e}")
            A_ph_est, tau_ph_est = 0.0, (t[-1] - t[0]) / 3
            f0_est, beta_est, phi_est = 10.0, 0.0, 0.0
        
        # Step 5: 估算三指数衰减参数
        try:
            # 从原始信号中减去估算的振荡成分
            osc_est = A_ph_est * np.exp(-t_prime / tau_ph_est) * np.cos(2 * np.pi * (f0_est * t_prime + 0.5 * beta_est * t_prime**2) + phi_est)
            y_exp_only = y_no_bg - osc_est
            
            # 使用 Prony 方法或简单的三指数拟合
            a1_est, tau1_est, a2_est, tau2_est, a3_est, tau3_est = self._estimate_triexp_prony(t_valid, y_exp_only)
            
            # 对于卷积模型，幅度需要减半
            a1_est /= 2.0
            a2_est /= 2.0
            a3_est /= 2.0
            
        except Exception as e:
            self._print_debug(f"  Triexp estimation failed: {e}")
            # 使用简单估算
            peak_val = np.max(np.abs(y_no_bg))
            a1_est = peak_val * 0.15
            a2_est = peak_val * 0.15
            a3_est = peak_val * 0.1
            tau1_est = (t[-1] - t[0]) * 0.05
            tau2_est = (t[-1] - t[0]) * 0.2
            tau3_est = (t[-1] - t[0]) * 0.8
        
        # 应用固定参数
        params = [
            fixed_params.get(0, a1_est),
            fixed_params.get(1, tau1_est),
            fixed_params.get(2, a2_est),
            fixed_params.get(3, tau2_est),
            fixed_params.get(4, a3_est),
            fixed_params.get(5, tau3_est),
            fixed_params.get(6, C_est),
            fixed_params.get(7, w_est),
            fixed_params.get(8, t0_est),
            fixed_params.get(9, A_ph_est),
            fixed_params.get(10, tau_ph_est),
            fixed_params.get(11, f0_est),
            fixed_params.get(12, beta_est),
            fixed_params.get(13, phi_est)
        ]
        
        guesses.append(params)
        
        # 添加一些变化
        if all(i not in fixed_params for i in [0, 1, 2, 3, 4, 5]):
            # 变化1：调整衰减时间顺序
            params_var1 = params.copy()
            params_var1[1] *= 0.5  # tau1
            params_var1[3] *= 0.8  # tau2
            params_var1[5] *= 1.2  # tau3
            guesses.append(params_var1)
        
        return guesses

    def _estimate_triexp_prony(self, t, y):
        """使用 Prony 方法估算三指数参数"""
        try:
            # 确保均匀采样
            dt = np.min(np.diff(t))
            if dt < 1e-12:
                raise ValueError("Time step too small")
            
            num_samples = int((t[-1] - t[0]) / dt)
            if num_samples < 15:
                raise ValueError("Not enough samples")
            
            t_uniform = np.linspace(t[0], t[0] + dt * (num_samples - 1), num_samples)
            y_uniform = np.interp(t_uniform, t, y)
            
            # Prony 方法
            p = 3  # 三个指数
            if len(y_uniform) <= 2 * p:
                raise ValueError("Not enough data points")
            
            M = np.zeros((len(y_uniform) - 2 * p, p))
            Y = -y_uniform[p : len(y_uniform) - p]
            
            for i in range(p):
                M[:, i] = y_uniform[p - 1 - i : len(y_uniform) - p - 1 - i]
            
            coeffs, _, _, _ = lstsq(M, Y, rcond=None)
            poly_coeffs = np.concatenate(([1], coeffs))
            
            roots = np.roots(poly_coeffs)
            valid_roots = [r.real for r in roots if 0 < r.real < 1.0 and np.abs(r.imag) < 1e-9]
            
            if len(valid_roots) != 3:
                self._print_debug(f"  Prony: Found {len(valid_roots)} valid roots instead of 3")
                # 尝试放宽条件
                valid_roots = [r.real for r in roots if 0 < r.real < 1.0 and np.abs(r.imag) < 0.1]
                if len(valid_roots) < 3:
                    raise ValueError(f"Invalid roots: {roots}")
            
            taus = sorted([-dt / np.log(r) for r in valid_roots[:3]])
            tau1, tau2, tau3 = taus[0], taus[1], taus[2]
            
            # 估算幅度
            basis_1 = np.exp(-t / tau1)
            basis_2 = np.exp(-t / tau2)
            basis_3 = np.exp(-t / tau3)
            basis_matrix = np.vstack([basis_1, basis_2, basis_3]).T
            
            amplitudes, _, _, _ = lstsq(basis_matrix, y, rcond=None)
            a1, a2, a3 = amplitudes[0], amplitudes[1], amplitudes[2]
            
            return a1, tau1, a2, tau2, a3, tau3
            
        except Exception:
            # 回退到简单估算
            peak = np.max(np.abs(y))
            t_range = t[-1] - t[0]
            return (peak * 0.3, t_range * 0.05, 
                   peak * 0.3, t_range * 0.2,
                   peak * 0.2, t_range * 0.8)

    def _heuristic_estimation(self, t, y_raw, y_smooth, t0_guess, fixed_params):
        """简单启发式参数估算"""
        # 基本统计
        y_min, y_max = np.min(y_smooth), np.max(y_smooth)
        t_range = t[-1] - t[0]
        
        # IRF宽度和时间零点
        w_est = fixed_params.get(7, t_range * 0.02)
        t0_est = fixed_params.get(8, t0_guess if t0_guess is not None else t[0])
        
        # 常数背景
        tail_size = max(5, len(y_smooth) // 10)
        C_full = fixed_params.get(6, np.mean(y_smooth[-tail_size:]))
        C_est = C_full / 2.0
        
        # 振荡参数的简单估算
        y_no_bg = y_smooth - C_full
        peak_val = np.max(np.abs(y_no_bg))
        
        # 三指数衰减参数（考虑卷积效应）
        a1_est = fixed_params.get(0, peak_val * 0.1)
        tau1_est = fixed_params.get(1, t_range * 0.05)
        a2_est = fixed_params.get(2, peak_val * 0.1)
        tau2_est = fixed_params.get(3, t_range * 0.2)
        a3_est = fixed_params.get(4, peak_val * 0.1)
        tau3_est = fixed_params.get(5, t_range * 0.8)
        
        # 振荡参数
        A_ph_est = fixed_params.get(9, peak_val * 0.1)
        tau_ph_est = fixed_params.get(10, t_range * 0.3)
        f0_est = fixed_params.get(11, 10.0 / t_range)
        beta_est = fixed_params.get(12, 0.0)
        phi_est = fixed_params.get(13, 0.0)
        
        base_params = [a1_est, tau1_est, a2_est, tau2_est, a3_est, tau3_est, 
                      C_est, w_est, t0_est, A_ph_est, tau_ph_est, f0_est, beta_est, phi_est]
        
        # 生成多个变化
        params_list = [base_params]
        
        # 变化1：不同的衰减时间比例
        var1 = base_params.copy()
        if 1 not in fixed_params:
            var1[1] *= 0.3
        if 3 not in fixed_params:
            var1[3] *= 0.7
        if 5 not in fixed_params:
            var1[5] *= 1.5
        params_list.append(var1)
        
        # 变化2：不同的振幅分配
        var2 = base_params.copy()
        if 0 not in fixed_params:
            var2[0] *= 2.0
        if 2 not in fixed_params:
            var2[2] *= 0.5
        if 4 not in fixed_params:
            var2[4] *= 0.5
        params_list.append(var2)
        
        # 变化3：不同的振荡参数
        var3 = base_params.copy()
        if 11 not in fixed_params:
            var3[11] *= 2.0
        if 12 not in fixed_params:
            var3[12] = 1.0 / (t_range**2)
        params_list.append(var3)
        
        return params_list[:5]

    def get_bounds(self, t_data_slice: np.ndarray, y_data_slice: np.ndarray, 
                   for_global_opt: bool = False) -> List[Tuple[float, float]]:
        """获取参数边界"""
        # 数据范围
        y_min, y_max = np.min(y_data_slice), np.max(y_data_slice)
        y_range = max(y_max - y_min, abs(y_max), abs(y_min), 1e-9)
        t_min, t_max = np.min(t_data_slice), np.max(t_data_slice)
        t_range = max(t_max - t_min, 1.0)
        
        # 幅度边界
        amp_factor = 5.0 if for_global_opt else 10.0
        amp_bound = amp_factor * y_range
        
        # 时间常数边界
        tau_min = max(t_range * 0.001, 1e-8)
        tau_max = t_range * (5.0 if for_global_opt else 20.0)
        
        # IRF宽度边界
        w_min = max(t_range * 0.00001, 1e-8)
        w_max = t_range * (0.5 if for_global_opt else 1.0)
        
        # t0边界
        t0_expansion = (0.3 if for_global_opt else 0.8) * t_range
        t0_min = t_min - t0_expansion
        t0_max = t_max + t0_expansion
        
        # 频率边界
        f_min = 1.0 / t_range
        f_max = 0.5 / np.mean(np.diff(t_data_slice)) if len(t_data_slice) > 1 else 1000.0
        
        # 啁啾率边界
        beta_max = f_max / t_range
        
        bounds = [
            (-amp_bound, amp_bound),   # a1
            (tau_min, tau_max),        # tau1
            (-amp_bound, amp_bound),   # a2
            (tau_min, tau_max),        # tau2
            (-amp_bound, amp_bound),   # a3
            (tau_min, tau_max),        # tau3
            (-amp_bound, amp_bound),   # C
            (w_min, w_max),            # w
            (t0_min, t0_max),          # t0
            (-amp_bound, amp_bound),   # A_ph
            (tau_min, tau_max),        # tau_ph
            (f_min, f_max),            # f0
            (-beta_max, beta_max),     # beta
            (0, 2 * np.pi)            # phi
        ]
        
        return bounds

    def check_parameter_validity(self, params: List[float], 
                                 t_data_for_context: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """检查参数有效性"""
        if len(params) != self.num_parameters:
            return False, f"参数数量错误, 期望 {self.num_parameters}, 得到 {len(params)}"
        
        if not all(np.isfinite(p) for p in params):
            return False, f"参数包含非有限值: {params}"
        
        a1, tau1, a2, tau2, a3, tau3, C, w, t0, A_ph, tau_ph, f0, beta, phi = params
        
        # 检查时间常数
        for i, (tau, name) in enumerate([(tau1, "tau1"), (tau2, "tau2"), (tau3, "tau3"), (tau_ph, "tau_ph")]):
            if tau <= 1e-9:
                return False, f"{name} 无效: {tau}"
        
        if w <= 1e-9:
            return False, f"w 无效: {w}"
        
        # 检查频率
        if f0 <= 0:
            return False, f"f0 必须为正: {f0}"
        
        # 检查相位
        if not (0 <= phi <= 2 * np.pi):
            return False, f"phi 应在 [0, 2π] 范围内: {phi}"
        
        # 检查时间常数的顺序（建议性，不是强制的）
        if tau1 > tau2 or tau2 > tau3:
            self._print_debug(f"Warning: tau values not in ascending order: tau1={tau1:.3f}, tau2={tau2:.3f}, tau3={tau3:.3f}")
        
        # 如果提供了时间数据，检查参数是否合理
        if t_data_for_context is not None and len(t_data_for_context) > 1:
            t_min, t_max = np.min(t_data_for_context), np.max(t_data_for_context)
            t_range = t_max - t_min
            
            # 检查t0是否在合理范围内
            if not ((t_min - t_range) <= t0 <= (t_max + t_range)):
                return False, f"t0 ({t0:.2f}) 超出数据范围"
            
            # 检查频率是否合理
            dt_mean = np.mean(np.diff(t_data_for_context))
            nyquist_freq = 0.5 / dt_mean
            if f0 > nyquist_freq:
                return False, f"f0 ({f0:.2f}) 超过 Nyquist 频率 ({nyquist_freq:.2f})"
            
            # 检查总的频率变化
            max_freq = f0 + abs(beta) * t_range
            if max_freq > nyquist_freq:
                return False, f"最大瞬时频率 ({max_freq:.2f}) 超过 Nyquist 频率"
        
        return True, ""