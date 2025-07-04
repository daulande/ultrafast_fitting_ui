# fitting_core.py (Refactored)
"""
核心拟合算法模块。
这个文件包含通用的拟合流程控制。
具体的模型函数、初始值估算等由传入的“拟合策略”对象提供。
"""
import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import savgol_filter
import warnings
import traceback

# 从新框架导入策略基类（用于类型提示）
from fitting_strategies.base_fitting_strategy import FittingStrategy

warnings.filterwarnings('ignore', category=RuntimeWarning)

class ExponentialFitter:
    """
    通用指数拟合器类。
    依赖于一个“拟合策略”对象来执行特定模型的拟合。
    """
    def __init__(self, strategy: FittingStrategy):
        """
        初始化拟合器。
        参数:
            strategy (FittingStrategy): 一个实现了FittingStrategy接口的对象实例。
        """
        if not isinstance(strategy, FittingStrategy):
            raise TypeError("The provided strategy must be an instance of a class inheriting from FittingStrategy.")
        self.strategy = strategy
        self.debug = False

    def set_debug(self, debug=True):
        """设置调试模式"""
        self.debug = debug
        if hasattr(self.strategy, 'debug'): # 如果策略本身也有debug模式
            self.strategy.debug = debug

    def _print_debug(self, msg):
        """打印调试信息"""
        if self.debug:
            print(f"[FITTER-CORE-DEBUG] {msg}")

    def _preprocess_data_for_estimation(self, y_data):
        """通用的数据平滑预处理，用于初始值估算"""
        if len(y_data) > 20:
            window = min(15, len(y_data) // 3 * 2 + 1)
            window = max(3, window if window % 2 != 0 else window - 1)
            polyorder = min(3, window - 1)
            if window > polyorder > 0:
                try:
                    y_smooth = savgol_filter(y_data, window, polyorder)
                    self._print_debug(f"  Applied Savitzky-Golay filter for estimation: window={window}, polyorder={polyorder}")
                    return y_smooth
                except ValueError as sve:
                    self._print_debug(f"  Savitzky-Golay filter failed: {sve}. Using original y for estimation.")
                    return y_data
        self._print_debug("  Skipping smoothing for estimation (data points <= 20).")
        return y_data

    def fit(self, time_original_slice, data_slice, time_zero_gui_context=0.0, fitting_method_type='multi_start'):
        """
        执行拟合的主函数。
        
        参数:
            time_original_slice (np.ndarray): 用于拟合的原始时间数据 (可能已被UI切片)。
            data_slice (np.ndarray): 对应的y轴数据。
            time_zero_gui_context (float): 主GUI的T0值，作为上下文传递给策略。
            fitting_method_type (str): 'multi_start' 或 'global'。
        
        返回:
            (params, errors, r_squared)
        """
        self._print_debug(f"Fit called for strategy: '{self.strategy.name}', GUI T0 (context): {time_zero_gui_context}, Method: {fitting_method_type}")

        if len(time_original_slice) < self.strategy.num_parameters + 2:
            raise ValueError(f"Not enough data points ({len(time_original_slice)}) for {self.strategy.num_parameters} parameters.")

        y_smooth_for_est = self._preprocess_data_for_estimation(data_slice)

        initial_guesses = self.strategy.estimate_initial_parameters(
            time_original_slice, data_slice, y_smooth_for_est, time_zero_gui_context
        )

        if not initial_guesses:
            raise RuntimeError(f"Fitting failed for '{self.strategy.name}': The strategy provided no initial guesses.")

        valid_initial_guesses = [g for g in initial_guesses if self.strategy.check_parameter_validity(g, time_original_slice)[0]]
        if not valid_initial_guesses:
            self._print_debug(f"Warning: All {len(initial_guesses)} initial guesses were flagged as invalid. Attempting with the first raw guess anyway.")
            if initial_guesses:
                valid_initial_guesses = initial_guesses[:1]
            if not valid_initial_guesses:
                raise RuntimeError(f"Fitting failed for '{self.strategy.name}': No valid initial guesses could be formed.")

        initial_guesses_to_try = valid_initial_guesses[:15] # Limit attempts
        self._print_debug(f"Fitting with '{self.strategy.name}'. Using {len(initial_guesses_to_try)} valid initial guess(es). First: {initial_guesses_to_try[0]}")

        # --- 时间轴处理 ---
        # 推荐的折中方案：根据模型类型决定是否调整时间轴
        # 假设策略有一个属性 `uses_absolute_t0_model`
        # 更好的方式是，这个逻辑应该封装在策略内部，但为了快速实现，我们暂时放在这里
        # if getattr(self.strategy, 'uses_absolute_t0_model', False):
        if self.strategy.name == "Erf Convolved Biexponential": # 这是一个临时的硬编码检查
             x_data_for_curve_fit = time_original_slice
             self._print_debug("  Using original time axis for curve_fit (Erf model).")
        else:
             x_data_for_curve_fit = time_original_slice - (time_zero_gui_context if time_zero_gui_context is not None else 0)
             self._print_debug(f"  Using time axis adjusted by GUI T0 ({time_zero_gui_context}) for curve_fit (Simple model).")

        # --- 拟合流程 ---
        bounds_tuples = self.strategy.get_bounds(x_data_for_curve_fit, data_slice, for_global_opt=False)
        cf_bounds = ([b[0] for b in bounds_tuples], [b[1] for b in bounds_tuples])
        
        best_result = None
        if fitting_method_type == 'multi_start':
            best_result = self._run_multi_start(x_data_for_curve_fit, data_slice, initial_guesses_to_try, cf_bounds)
        elif fitting_method_type == 'global':
            global_bounds = self.strategy.get_bounds(x_data_for_curve_fit, data_slice, for_global_opt=True)
            best_result = self._run_global_then_local(x_data_for_curve_fit, data_slice, global_bounds, cf_bounds)

        if best_result is None:
            raise RuntimeError(f"Fitting failed for '{self.strategy.name}': Unable to find suitable parameters with any method/guess.")

        params, pcov, r_squared = best_result
        errors = np.full_like(params, np.nan)
        if pcov is not None:
            try:
                diag_pcov = np.diag(pcov)
                valid_diag = diag_pcov >= 0
                errors[valid_diag] = np.sqrt(diag_pcov[valid_diag])
            except Exception as e_err:
                self._print_debug(f"Warning: Could not compute errors from pcov: {e_err}")
        
        self._print_debug(f"Final best R² for '{self.strategy.name}' = {r_squared:.6f}. Params: {params}, Errors: {errors}")
        return params, errors, r_squared

    def _run_multi_start(self, t_fit, y_fit, initial_guesses, bounds):
        best_overall_result = None
        best_overall_r2 = -np.inf
        for i, guess in enumerate(initial_guesses):
            self._print_debug(f"  Multi-start: Trying guess {i+1}/{len(initial_guesses)}: {guess}")
            current_fit_result = self._fit_with_initial_guess(t_fit, y_fit, guess, bounds=bounds)
            if current_fit_result:
                popt, _, r2 = current_fit_result
                if r2 > best_overall_r2:
                    best_overall_result = current_fit_result
                    best_overall_r2 = r2
                    self._print_debug(f"  Multi-start: New best (guess {i+1}): R² = {r2:.6f} with params {popt}")
        return best_overall_result

    def _run_global_then_local(self, t_fit, y_fit, global_bounds, local_bounds):
        self._print_debug("Starting global optimization (Differential Evolution)...")
        global_params = self._global_optimization(t_fit, y_fit, global_bounds)
        if global_params is not None:
            self._print_debug(f"Global optimization found params: {global_params}. Refining with local fit...")
            return self._fit_with_initial_guess(t_fit, y_fit, global_params, bounds=local_bounds, method='trf')
        else:
            self._print_debug("Global optimization did not find suitable parameters.")
            return None

    def _fit_with_initial_guess(self, t_fit, y_fit, initial_params, bounds=None, method=None):
        is_p0_valid, reason = self.strategy.check_parameter_validity(initial_params, t_fit)
        if not is_p0_valid:
            self._print_debug(f"  Skipping fit: Initial params {initial_params} are invalid. Reason: {reason}")
            return None

        best_result = None
        best_r2 = -np.inf
        methods_to_try = [method] if method else ['trf', 'dogbox']
        
        for current_method in methods_to_try:
            actual_bounds = bounds
            if current_method == 'lm':
                # 'lm' does not support bounds.
                actual_bounds = (-np.inf, np.inf)

            try:
                popt, pcov = curve_fit(self.strategy.model_function, t_fit, y_fit, p0=initial_params,
                                       bounds=actual_bounds, method=current_method,
                                       maxfev=50000, ftol=1e-9, xtol=1e-9)
                
                is_popt_valid, reason_popt = self.strategy.check_parameter_validity(popt, t_fit)
                if not is_popt_valid:
                    self._print_debug(f"  Method '{current_method}' produced invalid parameters: {popt}. Reason: {reason_popt}")
                    continue

                y_pred = self.strategy.model_function(t_fit, *popt)
                ss_res = np.sum((y_fit - y_pred) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0

                if r2 > best_r2:
                    best_r2 = r2
                    best_result = (popt, pcov, r2)
                    
            except RuntimeError as e:
                self._print_debug(f"  curve_fit with method '{current_method}' failed: {e}")
            except Exception as e_gen:
                self._print_debug(f"  General error during fitting with '{current_method}': {e_gen}")
        
        if best_result:
            self._print_debug(f"  _fit_with_initial_guess success for p0 {initial_params}. Best R²={best_result[2]:.6f}")
        return best_result

    def _global_optimization(self, t_fit, y_fit, bounds_global):
        def objective(params):
            # Objective function for differential_evolution
            try:
                y_pred = self.strategy.model_function(t_fit, *params)
                if not np.all(np.isfinite(y_pred)): return 1e12
                cost = np.sum((y_fit - y_pred) ** 2)
                return cost if np.isfinite(cost) else 1e12
            except Exception:
                return 1e12

        result = differential_evolution(objective, bounds_global, strategy='best1bin', maxiter=1500,
                                        popsize=20, tol=1e-6, mutation=(0.5, 1.5), recombination=0.8,
                                        workers=-1, updating='deferred', polish=True)
        
        if result.success:
            is_valid, _ = self.strategy.check_parameter_validity(result.x, t_fit)
            if is_valid:
                self._print_debug(f"Global optimization successful. Cost: {result.fun:.4e}, Params: {result.x}")
                return result.x
            else:
                self._print_debug(f"Global optimization succeeded but produced invalid params: {result.x}")
        else:
            self._print_debug(f"Global optimization failed. Message: {result.message}")
        
        return None
