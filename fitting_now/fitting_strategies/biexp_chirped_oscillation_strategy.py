# fitting_strategies/biexp_chirped_oscillation_strategy_improved.py
"""
改进版的双指数加啁啾振荡拟合策略
使用Matrix Pencil方法进行智能参数估计
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
from typing import List, Tuple, Optional, Dict
import warnings

from .base_fitting_strategy import FittingStrategy
# --- CORRECTED IMPORT ---
# The non-existent 'extract_parameters_for_fitting' has been removed.
from .matrix_pencil import MatrixPencilDecomposition


class BiexpChirpedOscillationStrategyImproved(FittingStrategy):
    """
    改进的策略实现：使用Matrix Pencil方法智能提取初始参数
    
    模型: y(t) = a1*exp(-t/tau1) + a2*exp(-t/tau2) + C + 
                 A_ph*exp(-t/tau_ph)*cos(2π(f0*t + 0.5*beta*t²) + phi)
    
    相比原版本的改进：
    1. 使用Matrix Pencil自动识别信号成分，无需二阶导数
    2. 能够处理极低频振荡（1-2个周期）
    3. 对噪声更加鲁棒
    4. 自动适应有/无振荡的情况
    """
    
    def __init__(self):
        self.debug = False
        # 初始化Matrix Pencil分解器（在需要时创建）
        self._mp_decomposer = None

    def _print_debug(self, msg):
        if self.debug:
            print(f"[STRATEGY-MP-DEBUG] {msg}")

    @property
    def name(self) -> str:
        return "Biexponential with Chirped Oscillation (Matrix Pencil Enhanced)"

    @property
    def num_parameters(self) -> int:
        return 12  # 参数数量不变

    def get_parameter_names(self) -> List[str]:
        return ["a1", "tau1", "a2", "tau2", "C", "w", "t0", "A_ph", "tau_ph", "f0", "beta", "phi"]

    def model_function(self, t: np.ndarray, *params) -> np.ndarray:
        """模型函数保持不变"""
        a1, tau1, a2, tau2, C, w, t0, A_ph, tau_ph, f0, beta, phi = params
        
        # 防止数值溢出
        _tau1 = max(tau1, 1e-9)
        _tau2 = max(tau2, 1e-9)
        _tau_ph = max(tau_ph, 1e-9)
        _w = max(w, 1e-9)
        
        # 相对于t0的时间
        decay_t = t - t0
        
        # 卷积的指数衰减项
        exp_decay1 = np.exp(-np.clip(decay_t / _tau1, -700, 700))
        arg_erf1 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau1)
        term1 = a1 * exp_decay1 * (1 + erf(np.clip(arg_erf1, -30, 30)))
        term1 = np.where(decay_t < -5 * _w, 0, term1)
        
        exp_decay2 = np.exp(-np.clip(decay_t / _tau2, -700, 700))
        arg_erf2 = 1.81 * (decay_t / _w) - _w / (3.62 * _tau2)
        term2 = a2 * exp_decay2 * (1 + erf(np.clip(arg_erf2, -30, 30)))
        term2 = np.where(decay_t < -5 * _w, 0, term2)
        
        # 常数项卷积
        arg_erf_c = 1.81 * (decay_t / _w)
        term_c = C * (1 + erf(np.clip(arg_erf_c, -30, 30)))
        term_c = np.where(decay_t < -5 * _w, 0, term_c)
        
        # 振荡衰减项
        phase = 2 * np.pi * (f0 * decay_t + 0.5 * beta * decay_t**2) + phi
        osc_decay = A_ph * np.exp(-np.clip(decay_t / _tau_ph, -700, 700)) * np.cos(phase)
        osc_decay = np.where(decay_t < 0, 0, osc_decay)
        
        result = term1 + term2 + term_c + osc_decay
        
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
        """
        使用Matrix Pencil方法估计初始参数
        
        这是核心改进：使用信号分解而不是导数分析
        """
        self._print_debug(f"开始Matrix Pencil参数估计. 数据长度: {len(time_original_slice)}")
        
        fixed_params = fixed_params or {}
        
        # 如果数据点太少，使用简单方法
        if len(time_original_slice) < 20:
            self._print_debug("数据点过少，使用启发式方法")
            return self._heuristic_estimation(
                time_original_slice, data_slice_processed, 
                data_slice_smooth_for_est, t_context_zero_gui, fixed_params
            )
        
        try:
            # 使用Matrix Pencil进行高级估算
            mp_guesses = self._matrix_pencil_estimation(
                time_original_slice, data_slice_processed, 
                data_slice_smooth_for_est, t_context_zero_gui, fixed_params
            )
            
            if mp_guesses:
                self._print_debug(f"Matrix Pencil成功，生成了{len(mp_guesses)}个初始猜测")
                return mp_guesses
            else:
                self._print_debug("Matrix Pencil未找到有效参数，回退到启发式方法")
                
        except Exception as e:
            self._print_debug(f"Matrix Pencil失败: {e}，使用启发式方法")
        
        # 回退到简单方法
        return self._heuristic_estimation(
            time_original_slice, data_slice_processed, 
            data_slice_smooth_for_est, t_context_zero_gui, fixed_params
        )

    def _matrix_pencil_estimation(self, t, y_raw, y_smooth, t0_guess, fixed_params):
        """
        使用Matrix Pencil方法进行参数估计
        
        这是改进的核心：直接从信号中提取极点，
        然后转换为物理参数
        """
        # 步骤1：准备Matrix Pencil分析
        # 计算采样间隔
        dt = np.mean(np.diff(t))
        
        # 创建Matrix Pencil分解器（如果还没有）
        if self._mp_decomposer is None or abs(self._mp_decomposer.dt - dt) > 1e-10:
            self._mp_decomposer = MatrixPencilDecomposition(dt, debug=self.debug)
        
        # 步骤2：估算或使用固定的w和t0
        w_est = fixed_params.get(5, (t[-1] - t[0]) * 0.02)
        t0_est = fixed_params.get(6, t0_guess if t0_guess is not None else t[0])
        
        # 步骤3：准备数据
        # 考虑到卷积效应，我们需要分析t0之后的数据
        t_relative = t - t0_est
        valid_idx = t_relative >= -2 * w_est  # 稍微包含一些t0之前的数据
        
        if np.sum(valid_idx) < 20:
            # 数据太少，扩大范围
            valid_idx = np.ones_like(t, dtype=bool)
        
        t_valid = t[valid_idx]
        y_valid = y_smooth[valid_idx]  # 使用平滑数据以减少噪声影响
        
        # 步骤4：执行Matrix Pencil分解
        # 估计期望的极点数：
        # - 2个实极点（双指数）
        # - 1个直流极点
        # - 2个复极点（振荡）
        # 总共5个，但我们稍微高估以捕获所有成分
        expected_poles = 6
        
        self._print_debug("执行Matrix Pencil分解...")
        decomp_result = self._mp_decomposer.decompose(
            y_valid, 
            expected_poles=expected_poles,
            pencil_parameter=0.4,
            noise_threshold=0.01
        )
        
        # 步骤5：从分解结果提取参数
        guesses = []
        
        # 基础参数提取
        mp_params = self._extract_parameters_from_mp(
            decomp_result, dt, t_valid, y_valid, 
            w_est, t0_est, fixed_params
        )
        
        if mp_params:
            guesses.append(mp_params)
            
            # 生成变体以提供多个初始猜测
            guesses.extend(self._generate_parameter_variants(mp_params, fixed_params))
        
        return guesses

    def _extract_parameters_from_mp(self, decomp_result, dt, t, y, w_est, t0_est, fixed_params):
        """
        从Matrix Pencil分解结果中提取拟合参数
        
        这个函数将极点和幅度转换为我们模型的物理参数
        """
        # 初始化参数字典
        params = {
            'a1': 0.0, 'tau1': (t[-1] - t[0]) * 0.1,
            'a2': 0.0, 'tau2': (t[-1] - t[0]) * 0.5,
            'C': 0.0,
            'A_ph': 0.0, 'tau_ph': (t[-1] - t[0]) * 0.3,
            'f0': 10.0, 'beta': 0.0, 'phi': 0.0
        }
        
        # 提取直流成分
        if decomp_result['dc_mode']:
            dc_idx = decomp_result['poles'].index(decomp_result['dc_mode']['pole'])
            # 考虑卷积效应，实际常数是提取值的一半
            params['C'] = decomp_result['amplitudes'][dc_idx].real / 2.0
        else:
            # 如果没有检测到直流，使用数据尾部估算
            params['C'] = np.mean(y[-max(5, len(y)//10):]) / 2.0
        
        # 提取指数衰减参数
        exp_modes = sorted(decomp_result['exponential_modes'], 
                          key=lambda x: x['time_constant'])
        
        if len(exp_modes) >= 1:
            # 第一个（最快的）指数衰减
            amp_idx = decomp_result['poles'].index(exp_modes[0]['pole'])
            # 考虑卷积效应
            params['a1'] = decomp_result['amplitudes'][amp_idx].real / 2.0
            params['tau1'] = exp_modes[0]['time_constant']
        
        if len(exp_modes) >= 2:
            # 第二个（较慢的）指数衰减
            amp_idx = decomp_result['poles'].index(exp_modes[1]['pole'])
            params['a2'] = decomp_result['amplitudes'][amp_idx].real / 2.0
            params['tau2'] = exp_modes[1]['time_constant']
        elif len(exp_modes) == 1:
            # 只有一个指数，分配给慢衰减
            params['a2'] = params['a1']
            params['tau2'] = params['tau1']
            params['a1'] = 0.0
            params['tau1'] = (t[-1] - t[0]) * 0.05
        
        # 提取振荡参数
        osc_modes = decomp_result['oscillatory_modes']
        if osc_modes:
            # 选择幅度最大的振荡模态
            main_osc = None
            max_amp = 0
            
            for mode in osc_modes:
                pole_idx = decomp_result['poles'].index(mode['pole'])
                amp = abs(decomp_result['amplitudes'][pole_idx])
                if amp > max_amp:
                    max_amp = amp
                    main_osc = mode
            
            if main_osc:
                pole_idx = decomp_result['poles'].index(main_osc['pole'])
                complex_amp = decomp_result['amplitudes'][pole_idx]
                
                # 从复幅度提取实际参数
                # 注意：Matrix Pencil给出的是单边谱，需要乘2
                params['A_ph'] = 2 * abs(complex_amp)
                params['tau_ph'] = main_osc['time_constant']
                params['f0'] = main_osc['frequency']
                params['phi'] = np.angle(complex_amp)
                
                # 估算chirp（这需要额外的分析）
                params['beta'] = self._estimate_chirp_from_residual(
                    t, y, params, w_est, t0_est
                )
        
        # 添加固定参数
        params['w'] = w_est
        params['t0'] = t0_est
        
        # 转换为列表格式，应用固定参数
        param_list = []
        param_names = self.get_parameter_names()
        for i, name in enumerate(param_names):
            if i in fixed_params:
                param_list.append(fixed_params[i])
            else:
                param_list.append(params[name])
        
        # 验证参数合理性
        valid, msg = self.check_parameter_validity(param_list, t)
        if not valid:
            self._print_debug(f"提取的参数无效: {msg}")
            return None
        
        return param_list

    def _estimate_chirp_from_residual(self, t, y, params, w_est, t0_est):
        """
        通过残差分析估算chirp参数
        
        由于chirp效应很小，Matrix Pencil会给出平均频率，
        我们可以通过分析残差来估算频率的线性变化
        """
        try:
            # 构造不含chirp的模型
            y_no_chirp = self._construct_model_without_chirp(t, params, w_est, t0_est)
            
            # 计算残差
            residual = y - y_no_chirp
            
            # 如果振荡幅度很小，chirp估算不可靠
            if params['A_ph'] < 0.01 * np.std(residual):
                return 0.0
            
            # 简单估算：假设残差主要由chirp引起
            # 使用相位的二阶导数估算
            t_rel = t - t0_est
            valid = t_rel > 0
            
            if np.sum(valid) < 10:
                return 0.0
            
            # 估算瞬时相位变化
            # 这是一个简化的估算，实际应用中可能需要更复杂的方法
            phase_error = np.arctan2(residual[valid], params['A_ph'] * np.exp(-t_rel[valid] / params['tau_ph']))
            
            # 线性拟合相位误差
            if len(phase_error) > 20:
                # 去除相位跳变
                phase_unwrapped = np.unwrap(phase_error)
                # 拟合二次项
                p = np.polyfit(t_rel[valid], phase_unwrapped / (2 * np.pi), 2)
                beta_est = 2 * p[0]  # 二次项系数的2倍
                
                # 限制chirp在合理范围内
                max_beta = params['f0'] / (t[-1] - t[0])
                beta_est = np.clip(beta_est, -max_beta, max_beta)
                
                return beta_est
            
        except Exception as e:
            self._print_debug(f"Chirp估算失败: {e}")
        
        return 0.0

    def _construct_model_without_chirp(self, t, params, w_est, t0_est):
        """构造不含chirp的模型，用于chirp估算"""
        t_rel = t - t0_est
        
        # 双指数部分（含卷积）
        y_model = np.zeros_like(t)
        
        # 使用简化的卷积（高斯近似）
        for a, tau in [(params['a1'], params['tau1']), (params['a2'], params['tau2'])]:
            if tau > 0:
                exp_decay = np.exp(-np.clip(t_rel / tau, -700, 700))
                # 简化的阶跃响应
                step = 0.5 * (1 + np.tanh(t_rel / w_est))
                y_model += a * exp_decay * step
        
        # 常数项
        step = 0.5 * (1 + np.tanh(t_rel / w_est))
        y_model += params['C'] * step
        
        # 振荡项（无chirp）
        if params['A_ph'] > 0 and params['tau_ph'] > 0:
            osc = params['A_ph'] * np.exp(-np.clip(t_rel / params['tau_ph'], -700, 700)) * \
                  np.cos(2 * np.pi * params['f0'] * t_rel + params['phi'])
            osc = np.where(t_rel < 0, 0, osc)
            y_model += osc
        
        return y_model

    def _generate_parameter_variants(self, base_params, fixed_params):
        """
        生成参数变体以提供多个初始猜测
        
        这提高了优化找到全局最优的概率
        """
        variants = []
        
        # 变体1：交换快慢衰减时间
        if 0 not in fixed_params and 2 not in fixed_params:
            var1 = base_params.copy()
            var1[0], var1[2] = var1[2], var1[0]  # 交换a1, a2
            var1[1], var1[3] = var1[3], var1[1]  # 交换tau1, tau2
            variants.append(var1)
        
        # 变体2：调整振荡频率
        if 9 not in fixed_params:
            var2 = base_params.copy()
            var2[9] *= 1.5  # 增加50%频率
            variants.append(var2)
            
            var3 = base_params.copy()
            var3[9] *= 0.7  # 减少30%频率
            variants.append(var3)
        
        # 变体3：添加小的chirp
        if 10 not in fixed_params and base_params[10] == 0:
            var4 = base_params.copy()
            # chirp约为频率的1/1000
            var4[10] = base_params[9] * 0.001
            variants.append(var4)
        
        # 限制变体数量
        return variants[:3]

    def _heuristic_estimation(self, t, y_raw, y_smooth, t0_guess, fixed_params):
        """简单启发式参数估算（作为后备方案）"""
        # 这部分保持与原版本相同
        y_min, y_max = np.min(y_smooth), np.max(y_smooth)
        t_range = t[-1] - t[0]
        
        w_est = fixed_params.get(5, t_range * 0.02)
        t0_est = fixed_params.get(6, t0_guess if t0_guess is not None else t[0])
        
        tail_size = max(5, len(y_smooth) // 10)
        C_full = fixed_params.get(4, np.mean(y_smooth[-tail_size:]))
        C_est = C_full / 2.0
        
        y_no_bg = y_smooth - C_full
        peak_val = np.max(np.abs(y_no_bg))
        
        a1_est = fixed_params.get(0, peak_val * 0.15)
        tau1_est = fixed_params.get(1, t_range * 0.1)
        a2_est = fixed_params.get(2, peak_val * 0.15)
        tau2_est = fixed_params.get(3, t_range * 0.5)
        
        A_ph_est = fixed_params.get(7, peak_val * 0.1)
        tau_ph_est = fixed_params.get(8, t_range * 0.3)
        f0_est = fixed_params.get(9, 10.0 / t_range)
        beta_est = fixed_params.get(10, 0.0)
        phi_est = fixed_params.get(11, 0.0)
        
        base_params = [a1_est, tau1_est, a2_est, tau2_est, C_est, w_est, t0_est,
                      A_ph_est, tau_ph_est, f0_est, beta_est, phi_est]
        
        return [base_params]

    def get_bounds(self, t_data_slice: np.ndarray, y_data_slice: np.ndarray, 
                   for_global_opt: bool = False) -> List[Tuple[float, float]]:
        """参数边界保持不变"""
        y_min, y_max = np.min(y_data_slice), np.max(y_data_slice)
        y_range = max(y_max - y_min, abs(y_max), abs(y_min), 1e-9)
        t_min, t_max = np.min(t_data_slice), np.max(t_data_slice)
        t_range = max(t_max - t_min, 1.0)
        
        amp_factor = 5.0 if for_global_opt else 10.0
        amp_bound = amp_factor * y_range
        
        tau_min = max(t_range * 0.001, 1e-8)
        tau_max = t_range * (5.0 if for_global_opt else 20.0)
        
        w_min = max(t_range * 0.00001, 1e-8)
        w_max = t_range * (0.5 if for_global_opt else 1.0)
        
        t0_expansion = (0.3 if for_global_opt else 0.8) * t_range
        t0_min = t_min - t0_expansion
        t0_max = t_max + t0_expansion
        
        f_min = 1.0 / t_range
        f_max = 0.5 / np.mean(np.diff(t_data_slice)) if len(t_data_slice) > 1 else 1000.0
        
        beta_max = f_max / t_range
        
        bounds = [
            (-amp_bound, amp_bound),   # a1
            (tau_min, tau_max),        # tau1
            (-amp_bound, amp_bound),   # a2
            (tau_min, tau_max),        # tau2
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
        """检查参数有效性（保持不变）"""
        if len(params) != self.num_parameters:
            return False, f"参数数量错误"
        
        if not all(np.isfinite(p) for p in params):
            return False, f"参数包含非有限值"
        
        a1, tau1, a2, tau2, C, w, t0, A_ph, tau_ph, f0, beta, phi = params
        
        if tau1 <= 1e-9 or tau2 <= 1e-9 or tau_ph <= 1e-9 or w <= 1e-9:
            return False, f"时间常数无效"
        
        if f0 <= 0:
            return False, f"频率必须为正"
        
        if not (0 <= phi <= 2 * np.pi):
            return False, f"相位超出范围"
        
        if t_data_for_context is not None and len(t_data_for_context) > 1:
            t_min, t_max = np.min(t_data_for_context), np.max(t_data_for_context)
            t_range = t_max - t_min
            
            if not ((t_min - t_range) <= t0 <= (t_max + t_range)):
                return False, f"t0超出数据范围"
            
            dt_mean = np.mean(np.diff(t_data_for_context))
            nyquist_freq = 0.5 / dt_mean
            if f0 > nyquist_freq:
                return False, f"频率超过Nyquist频率"
            
            max_freq = f0 + abs(beta) * t_range
            if max_freq > nyquist_freq:
                return False, f"最大瞬时频率超过Nyquist频率"
        
        return True, ""


# 使用示例和测试
if __name__ == "__main__":
    # 创建测试数据
    import matplotlib.pyplot as plt
    
    # 生成类似你图中的信号
    t = np.linspace(-10, 250, 1000)
    
    # 参数设置
    a1, tau1 = 0.05, 15
    a2, tau2 = 0.15, 80
    C = -0.3
    w, t0 = 2.0, 0.0
    A_ph, tau_ph = 0.03, 60
    f0, beta = 8.0, 0.01  # 8 GHz, 轻微chirp
    phi = np.pi/4
    
    # 创建策略实例并生成信号
    strategy = BiexpChirpedOscillationStrategyImproved()
    strategy.debug = True
    
    params_true = [a1, tau1, a2, tau2, C, w, t0, A_ph, tau_ph, f0, beta, phi]
    y_true = strategy.model_function(t, *params_true)
    
    # 添加噪声
    np.random.seed(42)
    noise_level = 0.002
    y_noisy = y_true + noise_level * np.random.randn(len(t))
    
    # 测试参数估计
    print("=== 测试Matrix Pencil增强的参数估计 ===\n")
    
    # 模拟平滑数据
    from scipy.ndimage import gaussian_filter1d
    y_smooth = gaussian_filter1d(y_noisy, sigma=3)
    
    # 估计初始参数
    initial_guesses = strategy.estimate_initial_parameters(
        t, y_noisy, y_smooth, t0_guess=0.0, fixed_params={5: w, 6: t0}
    )
    
    print(f"\n生成了 {len(initial_guesses)} 个初始猜测")
    
    # 显示第一个猜测与真实值的比较
    if initial_guesses:
        guess = initial_guesses[0]
        param_names = strategy.get_parameter_names()
        
        print("\n参数比较（真实值 vs 估计值）:")
        print("-" * 50)
        for i, name in enumerate(param_names):
            true_val = params_true[i]
            est_val = guess[i]
            if true_val != 0:
                error = abs((est_val - true_val) / true_val) * 100
                print(f"{name:8s}: {true_val:10.4f} vs {est_val:10.4f} (误差: {error:6.1f}%)")
            else:
                print(f"{name:8s}: {true_val:10.4f} vs {est_val:10.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, y_noisy, 'b.', alpha=0.3, markersize=2, label='含噪声数据')
    plt.plot(t, y_true, 'r-', linewidth=2, label='真实信号')
    if initial_guesses:
        y_est = strategy.model_function(t, *initial_guesses[0])
        plt.plot(t, y_est, 'g--', linewidth=2, label='Matrix Pencil估计')
    plt.xlabel('时间 (ps)')
    plt.ylabel('信号')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-10, 250)
    
    plt.subplot(2, 1, 2)
    if initial_guesses:
        residual = y_true - y_est
        plt.plot(t, residual, 'k-', linewidth=1)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('时间 (ps)')
        plt.ylabel('残差')
        plt.grid(True, alpha=0.3)
        plt.xlim(-10, 250)
        
        rms_error = np.sqrt(np.mean(residual**2))
        plt.title(f'残差 (RMS = {rms_error:.4f})')
    
    plt.tight_layout()
    plt.show()