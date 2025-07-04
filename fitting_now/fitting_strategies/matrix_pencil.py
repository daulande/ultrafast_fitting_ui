"""
自适应多尺度分解（AMSD）方法
专门设计用于处理具有极大时间尺度差异的信号

核心思想：
1. 使用对数时间变换来"压缩"时间尺度差异
2. 基于导数分析自动识别特征时间尺度
3. 迭代提取不同尺度的成分
4. 局部应用Matrix Pencil以保持数值稳定性
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict, Optional
import warnings

class MatrixPencilDecomposition:
    """
    自适应多尺度分解器
    
    这个方法能够处理时间尺度相差2-30倍的信号成分
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._mp_decomposer = None
        
    def _print_debug(self, msg: str):
        if self.debug:
            print(f"[AMSD] {msg}")
    
    def decompose_adaptive(self, t: np.ndarray, y: np.ndarray, 
                          t0_est: float, w_est: float) -> Dict:
        """
        主入口：执行自适应多尺度分解
        
        返回格式适合BiexpChirpedOscillationStrategy使用的参数
        """
        # 步骤1：数据预处理和特征时间尺度识别
        self._print_debug("步骤1：识别特征时间尺度...")
        time_scales = self._identify_time_scales(t, y, t0_est)
        self._print_debug(f"识别到的时间尺度: {time_scales}")
        
        # 步骤2：多尺度成分提取
        self._print_debug("步骤2：提取多尺度成分...")
        components = self._extract_multiscale_components(t, y, t0_est, time_scales)
        
        # 步骤3：识别振荡成分（可能很早就开始）
        self._print_debug("步骤3：检测早期振荡...")
        oscillation = self._detect_early_oscillation(t, y, t0_est, components)
        
        # 步骤4：整合为最终参数
        self._print_debug("步骤4：参数整合...")
        params = self._integrate_parameters(components, oscillation, t, y, t0_est, w_est)
        
        return params
    
    def _identify_time_scales(self, t: np.ndarray, y: np.ndarray, 
                             t0: float) -> List[float]:
        """
        自动识别信号中的特征时间尺度
        
        使用多种方法的组合：
        1. 导数分析找转折点
        2. 对数导数分析找时间常数
        3. 局部斜率变化检测
        """
        # 只分析t0之后的数据
        mask = t >= t0
        if np.sum(mask) < 10:
            return [5.0, 50.0]  # 默认值
        
        t_rel = t[mask] - t0
        y_data = y[mask]
        
        # 确保数据为正（为对数分析）
        y_offset = np.min(y_data) - 0.1 * np.abs(np.min(y_data))
        y_positive = y_data - y_offset
        y_positive = np.maximum(y_positive, 1e-10)
        
        time_scales = []
        
        # 方法1：对数导数分析
        # 对于纯指数衰减，d(ln y)/dt = -1/tau
        try:
            # 使用Savitzky-Golay滤波器计算平滑导数
            window_length = min(21, len(t_rel) // 4)
            if window_length % 2 == 0:
                window_length += 1
            
            if window_length >= 5:
                log_y = np.log(y_positive)
                # 一阶导数
                d_log_y = savgol_filter(log_y, window_length, 3, deriv=1, 
                                       delta=np.mean(np.diff(t_rel)))
                
                # 局部时间常数估计
                tau_local = -1.0 / (d_log_y + 1e-10)
                tau_local = tau_local[tau_local > 0]
                tau_local = tau_local[tau_local < 1000]  # 合理范围
                
                if len(tau_local) > 0:
                    # 使用聚类识别不同的时间常数
                    tau_clusters = self._cluster_time_constants(tau_local)
                    time_scales.extend(tau_clusters)
        except Exception as e:
            self._print_debug(f"对数导数分析失败: {e}")
        
        # 方法2：基于曲率的特征点检测
        try:
            # 计算信号的二阶导数（曲率相关）
            if len(t_rel) > 10:
                # 使用更稳健的数值微分
                dt = np.mean(np.diff(t_rel))
                smoothed = savgol_filter(y_data, 
                                       min(11, len(y_data)//2*2-1), 3)
                
                # 找到曲率最大的点
                d2y = np.gradient(np.gradient(smoothed, dt), dt)
                
                # 归一化曲率
                curvature = np.abs(d2y) / (1 + np.gradient(smoothed, dt)**2)**1.5
                
                # 找峰值
                peaks, properties = find_peaks(curvature, 
                                             height=np.max(curvature)*0.1,
                                             distance=5)
                
                # 这些峰值对应的时间可能是特征时间
                for peak_idx in peaks[:3]:  # 最多取3个
                    characteristic_time = t_rel[peak_idx]
                    if characteristic_time > 0:
                        time_scales.append(characteristic_time)
        except Exception as e:
            self._print_debug(f"曲率分析失败: {e}")
        
        # 方法3：多分辨率斜率分析
        # 在对数-对数空间中寻找斜率变化
        try:
            if len(t_rel) > 20:
                # 对数-对数变换
                log_t = np.log(t_rel + 0.1)  # 避免log(0)
                log_y_pos = np.log(y_positive)
                
                # 在不同尺度上计算局部斜率
                window_sizes = [5, 10, 20]
                for window in window_sizes:
                    if window < len(log_t):
                        slopes = []
                        times = []
                        
                        for i in range(len(log_t) - window):
                            # 局部线性拟合
                            idx = slice(i, i + window)
                            if len(log_t[idx]) > 3:
                                p = np.polyfit(log_t[idx], log_y_pos[idx], 1)
                                slopes.append(p[0])
                                times.append(t_rel[i + window//2])
                        
                        if slopes:
                            # 寻找斜率变化点
                            slopes = np.array(slopes)
                            d_slopes = np.abs(np.diff(slopes))
                            
                            # 找到显著的斜率变化
                            change_points = np.where(d_slopes > np.std(d_slopes))[0]
                            for cp in change_points[:2]:
                                if cp < len(times):
                                    time_scales.append(times[cp])
        except Exception as e:
            self._print_debug(f"多分辨率分析失败: {e}")
        
        # 去重和排序
        if time_scales:
            time_scales = list(set(time_scales))
            time_scales.sort()
            
            # 合并过于接近的时间尺度
            merged = [time_scales[0]]
            for ts in time_scales[1:]:
                if ts / merged[-1] > 1.5:  # 至少相差50%
                    merged.append(ts)
            
            return merged[:4]  # 最多返回4个时间尺度
        else:
            # 默认时间尺度
            t_range = t_rel[-1]
            return [t_range * 0.05, t_range * 0.2, t_range * 0.5]
    
    def _cluster_time_constants(self, tau_values: np.ndarray) -> List[float]:
        """
        对时间常数进行聚类，识别不同的特征尺度
        
        使用简单的对数空间聚类
        """
        if len(tau_values) == 0:
            return []
        
        # 在对数空间中聚类
        log_tau = np.log(tau_values)
        
        # 简单的1D聚类：寻找间隙
        sorted_log_tau = np.sort(log_tau)
        gaps = np.diff(sorted_log_tau)
        
        # 找到显著的间隙（大于平均间隙的2倍）
        if len(gaps) > 0:
            gap_threshold = 2 * np.mean(gaps)
            cluster_boundaries = np.where(gaps > gap_threshold)[0] + 1
            
            # 提取每个簇的代表值（中位数）
            clusters = []
            start = 0
            
            for boundary in cluster_boundaries:
                cluster_values = sorted_log_tau[start:boundary]
                if len(cluster_values) > 0:
                    clusters.append(np.exp(np.median(cluster_values)))
                start = boundary
            
            # 最后一个簇
            if start < len(sorted_log_tau):
                cluster_values = sorted_log_tau[start:]
                clusters.append(np.exp(np.median(cluster_values)))
            
            return clusters
        else:
            return [np.exp(np.median(log_tau))]
    
    def _extract_multiscale_components(self, t: np.ndarray, y: np.ndarray,
                                     t0: float, time_scales: List[float]) -> Dict:
        """
        迭代提取不同时间尺度的成分
        
        策略：从最快的成分开始，逐步提取并减去
        """
        components = {
            'fast': None,
            'medium': None, 
            'slow': None,
            'constant': 0.0
        }
        
        # 工作副本
        t_work = t[t >= t0] - t0
        y_work = y[t >= t0].copy()
        
        # 估计常数背景
        tail_size = max(5, len(y_work) // 5)
        components['constant'] = np.mean(y_work[-tail_size:])
        
        # 按时间尺度从小到大处理
        time_scales_sorted = sorted(time_scales)
        
        for i, tau_target in enumerate(time_scales_sorted[:3]):
            self._print_debug(f"提取时间尺度 ~{tau_target:.1f} 的成分...")
            
            # 选择合适的分析窗口
            # 窗口应该包含该时间尺度的几个特征长度
            window_end = min(5 * tau_target, t_work[-1])
            window_mask = t_work <= window_end
            
            if np.sum(window_mask) < 5:
                continue
            
            # 局部拟合单指数（考虑已提取的成分）
            try:
                # 定义局部拟合函数
                def local_exp(t_local, a, tau, c):
                    return a * np.exp(-t_local / tau) + c
                
                # 初始猜测
                y_window = y_work[window_mask]
                t_window = t_work[window_mask]
                
                a_guess = y_window[0] - components['constant']
                tau_guess = tau_target
                c_guess = components['constant']
                
                # 边界
                bounds = (
                    [-3*abs(a_guess), tau_guess*0.1, -abs(components['constant'])*2],
                    [3*abs(a_guess), tau_guess*10, abs(components['constant'])*2]
                )
                
                # 拟合
                popt, _ = curve_fit(local_exp, t_window, y_window,
                                   p0=[a_guess, tau_guess, c_guess],
                                   bounds=bounds,
                                   maxfev=1000)
                
                # 存储成分
                if i == 0 and tau_target < 20:
                    components['fast'] = {'amplitude': popt[0], 'tau': popt[1]}
                elif i == 1 or (i == 0 and components['fast'] is None):
                    components['medium'] = {'amplitude': popt[0], 'tau': popt[1]}
                else:
                    components['slow'] = {'amplitude': popt[0], 'tau': popt[1]}
                
                # 从信号中减去此成分（在整个范围上）
                y_work -= popt[0] * np.exp(-t_work / popt[1])
                
            except Exception as e:
                self._print_debug(f"成分提取失败: {e}")
        
        return components
    
    def _detect_early_oscillation(self, t: np.ndarray, y: np.ndarray,
                                 t0: float, components: Dict) -> Optional[Dict]:
        """
        检测可能很早就开始的振荡
        
        关键：即使振荡在8ps就开始，也要能检测到
        """
        # 构造去除指数成分后的残差
        t_work = t[t >= t0] - t0
        y_residual = y[t >= t0].copy()
        
        # 减去已识别的指数成分
        for comp_name in ['fast', 'medium', 'slow']:
            if components[comp_name]:
                amp = components[comp_name]['amplitude']
                tau = components[comp_name]['tau']
                y_residual -= amp * np.exp(-t_work / tau)
        
        # 减去常数
        y_residual -= components['constant']
        
        # 检测振荡的多种方法
        oscillation = None
        
        # 方法1：自相关分析
        if len(y_residual) > 20:
            # 计算自相关函数
            autocorr = np.correlate(y_residual, y_residual, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # 只取正延迟部分
            autocorr = autocorr / autocorr[0]  # 归一化
            
            # 寻找第一个显著的峰（排除零延迟）
            if len(autocorr) > 10:
                peaks, _ = find_peaks(autocorr[5:], height=0.1)
                if len(peaks) > 0:
                    # 第一个峰对应周期
                    period_samples = peaks[0] + 5
                    dt = np.mean(np.diff(t_work))
                    period = period_samples * dt
                    frequency = 1.0 / period
                    
                    self._print_debug(f"自相关检测到频率: {frequency:.2f}")
                    
                    # 估计振幅和衰减
                    # 使用包络检测
                    envelope = self._extract_envelope(y_residual)
                    if len(envelope) > 10:
                        # 拟合指数包络
                        try:
                            def exp_decay(t, A, tau):
                                return A * np.exp(-t / tau)
                            
                            popt, _ = curve_fit(exp_decay, t_work, envelope,
                                              p0=[np.max(envelope), 50])
                            
                            oscillation = {
                                'amplitude': popt[0],
                                'tau': popt[1],
                                'frequency': frequency,
                                'phase': 0.0  # 相位需要更复杂的分析
                            }
                        except:
                            pass
        
        # 方法2：FFT分析（作为备选）
        if oscillation is None and len(y_residual) > 50:
            # 零填充以提高频率分辨率
            n_fft = 2**int(np.ceil(np.log2(len(y_residual) * 4)))
            
            # 应用窗函数减少频谱泄露
            window = np.hanning(len(y_residual))
            y_windowed = y_residual * window
            
            # FFT
            fft_result = np.fft.fft(y_windowed, n=n_fft)
            freqs = np.fft.fftfreq(n_fft, d=np.mean(np.diff(t_work)))
            
            # 只看正频率
            positive_freq_idx = freqs > 0
            freqs_positive = freqs[positive_freq_idx]
            magnitude = np.abs(fft_result[positive_freq_idx])
            
            # 找主频率
            peak_idx = np.argmax(magnitude)
            if magnitude[peak_idx] > 0.01 * np.max(np.abs(y_residual)):
                main_freq = freqs_positive[peak_idx]
                
                # 简单的幅度估计
                amplitude = 2 * magnitude[peak_idx] / len(y_residual)
                
                oscillation = {
                    'amplitude': amplitude,
                    'tau': 50.0,  # 默认值，需要更好的估计
                    'frequency': main_freq,
                    'phase': 0.0
                }
        
        return oscillation
    
    def _extract_envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        提取振荡信号的包络
        
        使用Hilbert变换或峰值检测
        """
        # 简单方法：连接局部最大值
        from scipy.signal import find_peaks
        
        # 找到所有局部最大值
        peaks, _ = find_peaks(np.abs(signal))
        
        if len(peaks) > 3:
            # 插值生成包络
            envelope_interp = interp1d(peaks, np.abs(signal[peaks]),
                                     kind='linear', fill_value='extrapolate')
            envelope = envelope_interp(np.arange(len(signal)))
            return np.abs(envelope)
        else:
            # 备选：使用滑动最大值
            window = min(10, len(signal) // 4)
            envelope = np.array([np.max(np.abs(signal[max(0, i-window):i+window+1])) 
                               for i in range(len(signal))])
            return envelope
    
    def _integrate_parameters(self, components: Dict, oscillation: Optional[Dict],
                            t: np.ndarray, y: np.ndarray, 
                            t0: float, w: float) -> List[List[float]]:
        """
        将提取的成分整合为BiexpChirpedOscillationStrategy格式的参数
        """
        # 基础参数模板
        params = [0.0] * 12
        
        # 分配指数成分
        exp_components = []
        for comp_type in ['fast', 'medium', 'slow']:
            if components[comp_type]:
                exp_components.append(components[comp_type])
        
        # 确保至少有两个指数成分
        while len(exp_components) < 2:
            # 添加默认成分
            if len(exp_components) == 0:
                exp_components.append({'amplitude': 0.1, 'tau': 10.0})
            else:
                exp_components.append({'amplitude': 0.05, 'tau': 100.0})
        
        # 按tau排序并分配
        exp_components.sort(key=lambda x: x['tau'])
        
        params[0] = exp_components[0]['amplitude']  # a1
        params[1] = exp_components[0]['tau']        # tau1
        params[2] = exp_components[1]['amplitude']  # a2
        params[3] = exp_components[1]['tau']        # tau2
        
        # 常数项（考虑卷积效应，需要除以2）
        params[4] = components['constant'] / 2.0    # C
        
        # 固定参数
        params[5] = w                               # w
        params[6] = t0                              # t0
        
        # 振荡参数
        if oscillation:
            params[7] = oscillation['amplitude']     # A_ph
            params[8] = oscillation['tau']          # tau_ph
            params[9] = oscillation['frequency']    # f0
            params[10] = 0.0                        # beta (chirp)
            params[11] = oscillation.get('phase', 0.0)  # phi
        else:
            # 默认振荡参数
            y_range = np.max(y) - np.min(y)
            params[7] = 0.01 * y_range
            params[8] = 50.0
            params[9] = 10.0
            params[10] = 0.0
            params[11] = 0.0
        
        # 生成多个初始猜测
        guesses = [params]
        
        # 变体1：强调更快的成分
        if params[1] > 5:
            fast_variant = params.copy()
            fast_variant[1] = 2.0  # 非常快的tau1
            fast_variant[0] *= 1.5
            guesses.append(fast_variant)
        
        # 变体2：不同的振荡起始时间
        if oscillation and params[9] > 0:
            early_osc = params.copy()
            early_osc[7] *= 1.2  # 稍大的振幅
            early_osc[8] *= 0.8  # 稍快的衰减
            guesses.append(early_osc)
        
        return guesses


def integrate_amsd_with_strategy(strategy_instance):
    """
    将AMSD方法集成到现有的BiexpChirpedOscillationStrategy中
    
    这个函数修改策略实例，使其使用AMSD方法
    """
    # 创建AMSD实例
    amsd = AdaptiveMultiScaleDecomposition(debug=strategy_instance.debug)
    
    # 保存原始方法
    original_mp_estimation = strategy_instance._matrix_pencil_estimation
    
    # 定义新的估计方法
    def new_estimation_method(t, y_raw, y_smooth, t0_guess, fixed_params):
        """使用AMSD的新估计方法"""
        strategy_instance._print_debug("使用AMSD方法进行参数估计...")
        
        # 获取固定参数
        w_est = fixed_params.get(5, (t[-1] - t[0]) * 0.02)
        t0_est = fixed_params.get(6, t0_guess if t0_guess is not None else t[0])
        
        try:
            # 使用AMSD
            param_guesses = amsd.decompose_adaptive(t, y_smooth, t0_est, w_est)
            
            # 应用固定参数
            for guess in param_guesses:
                for idx, value in fixed_params.items():
                    guess[idx] = value
            
            return param_guesses
            
        except Exception as e:
            strategy_instance._print_debug(f"AMSD失败: {e}，回退到原方法")
            return original_mp_estimation(t, y_raw, y_smooth, t0_guess, fixed_params)
    
    # 替换方法
    strategy_instance._matrix_pencil_estimation = new_estimation_method
    
    return strategy_instance