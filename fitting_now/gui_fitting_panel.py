# gui_fitting_panel.py
"""
A reusable panel containing all UI elements and logic for fitting a single dataset.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import traceback
from typing import Optional, Callable, Dict, Any, List, Tuple
from scipy.optimize import curve_fit

# Assume these modules are in the project path
from fitting_core import ExponentialFitter
from fitting_strategies.base_fitting_strategy import FittingStrategy
import fitting_dispatcher

class FittingPanel(ttk.Frame):
    """A self-contained frame for fitting a single temperature's data."""
    def __init__(self, parent, apply_callback: Callable,
                 current_temp_val: float, times_original: np.ndarray, data_original: np.ndarray,
                 time_zero_main_context: float, debug_mode: bool, preview_callback: Callable,
                 default_strategy_name: str, initial_fit_result: Optional[Dict] = None):
        super().__init__(parent, padding=10)
        
        self.apply_callback = apply_callback
        
        # --- Core Attributes ---
        self.strategy: Optional[FittingStrategy] = None
        self.times_original = times_original
        self.data_original = data_original
        self.current_temp_val = current_temp_val
        self.time_zero_main_context = time_zero_main_context
        self.debug_mode = debug_mode
        self.preview_callback = preview_callback
        
        self.fitter: Optional[ExponentialFitter] = None
        
        # --- Tkinter Variables ---
        self.param_vars_tk: List[tk.DoubleVar] = []
        self.fix_param_vars_tk: List[tk.BooleanVar] = []
        self.fit_range_start_var = tk.DoubleVar(value=round(self.times_original.min(), 4) if times_original.size > 0 else 0)
        self.fit_range_end_var = tk.DoubleVar(value=round(self.times_original.max(), 4) if times_original.size > 0 else 0)
        self.fitting_method_var = tk.StringVar(value='trf')
        self.r_squared_label_var = tk.StringVar(value="R²: -")
        self.chi_squared_label_var = tk.StringVar(value="\u03C7²_red: -")
        self.local_debug_var = tk.BooleanVar(value=self.debug_mode)
        self.strategy_name_var = tk.StringVar()

        # --- State Variables ---
        self.last_fit_result: Optional[Dict[str, Any]] = initial_fit_result

        self._create_widgets()
        
        # --- Initialization ---
        available_strategies = fitting_dispatcher.get_available_strategy_names()
        if available_strategies:
            initial_strategy = default_strategy_name
            # If an existing result is provided, use its strategy
            if initial_fit_result and initial_fit_result.get('strategy_name_used') in available_strategies:
                initial_strategy = initial_fit_result['strategy_name_used']
            
            self.strategy_name_var.set(initial_strategy)
            self._on_strategy_change()

            # If existing result is provided, load its data
            if initial_fit_result:
                self._load_from_fit_result(initial_fit_result)
        
    def _create_widgets(self):
        # The content will be placed in a scrollable frame
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=550) # Set a sensible initial width
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, padding=10)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"), "+")

        # --- Widgets inside the scrollable frame ---
        container = self.scrollable_frame

        model_frame = ttk.LabelFrame(container, text="拟合模型", padding=5)
        model_frame.pack(fill=tk.X, pady=5)
        self.strategy_combobox = ttk.Combobox(model_frame, textvariable=self.strategy_name_var,
                                              values=fitting_dispatcher.get_available_strategy_names(),
                                              state="readonly")
        self.strategy_combobox.pack(fill=tk.X, expand=True)
        self.strategy_combobox.bind("<<ComboboxSelected>>", self._on_strategy_change)

        self.params_outer_frame = ttk.LabelFrame(container, text="拟合参数", padding=5)
        self.params_outer_frame.pack(fill=tk.X, pady=5)
        self.params_frame = ttk.Frame(self.params_outer_frame)
        self.params_frame.pack(fill=tk.X, expand=True)

        fit_range_frame = ttk.LabelFrame(container, text="拟合范围 (绝对时间)", padding=5)
        fit_range_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fit_range_frame, text="起始:").grid(row=0, column=0)
        ttk.Entry(fit_range_frame, textvariable=self.fit_range_start_var, width=12).grid(row=0, column=1, padx=5)
        ttk.Label(fit_range_frame, text="结束:").grid(row=0, column=2)
        ttk.Entry(fit_range_frame, textvariable=self.fit_range_end_var, width=12).grid(row=0, column=3, padx=5)

        fit_method_frame = ttk.LabelFrame(container, text="拟合算法 (curve_fit)", padding=5)
        fit_method_frame.pack(fill=tk.X, pady=5)
        ttk.Combobox(fit_method_frame, textvariable=self.fitting_method_var,
                     values=['trf', 'dogbox', 'lm'], state="readonly", width=12).pack(side=tk.LEFT, padx=5)

        stats_frame = ttk.LabelFrame(container, text="统计与调试", padding=5)
        stats_frame.pack(fill=tk.X, pady=5)
        ttk.Label(stats_frame, textvariable=self.r_squared_label_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.chi_squared_label_var).pack(anchor=tk.W)
        ttk.Checkbutton(stats_frame, text="显示详细调试信息", variable=self.local_debug_var,
                        command=self._toggle_debug).pack(anchor=tk.W)

        action_frame = ttk.Frame(container, padding=(0, 10))
        action_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10,0))
        ttk.Button(action_frame, text="寻找初始值", command=self._find_initial_values).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="拟合", command=self._fit).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="应用", command=self._on_apply).pack(side=tk.RIGHT, padx=5)

    def _build_param_entries(self):
        for widget in self.params_frame.winfo_children(): widget.destroy()
        
        self.param_vars_tk.clear(); self.fix_param_vars_tk.clear()
        
        param_names = self.strategy.get_parameter_names()
        self.params_outer_frame.config(text=f"拟合参数 ({len(param_names)}个)")

        ttk.Label(self.params_frame, text="参数").grid(row=0, column=0, padx=2)
        ttk.Label(self.params_frame, text="值").grid(row=0, column=1, padx=2)
        ttk.Label(self.params_frame, text="固定?").grid(row=0, column=2, padx=2)

        for i, name in enumerate(param_names):
            ttk.Label(self.params_frame, text=name + ":").grid(row=i + 1, column=0, sticky=tk.W, pady=2)
            var = tk.DoubleVar(value=np.nan)
            entry = ttk.Entry(self.params_frame, textvariable=var, width=15)
            entry.grid(row=i + 1, column=1, padx=2, pady=2)
            entry.bind("<FocusOut>", lambda e: self._update_main_gui_preview())
            entry.bind("<Return>", lambda e: self._update_main_gui_preview())
            self.param_vars_tk.append(var)
            fix_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.params_frame, variable=fix_var).grid(row=i + 1, column=2, padx=2, pady=2)
            self.fix_param_vars_tk.append(fix_var)
    
    def _initialize_and_preview(self):
        self._build_param_entries()
        self._find_initial_values()

    def _on_strategy_change(self, event=None):
        new_strategy_name = self.strategy_name_var.get()
        new_strategy = fitting_dispatcher.get_strategy(new_strategy_name)
        if new_strategy:
            self.strategy = new_strategy
            self.fitter = ExponentialFitter(self.strategy)
            self.fitter.set_debug(self.local_debug_var.get())
            self.last_fit_result = None # Clear old result when strategy changes
            self._initialize_and_preview()
        else:
            messagebox.showerror("错误", f"无法加载策略: {new_strategy_name}", parent=self)
            
    def _toggle_debug(self):
        if self.fitter: self.fitter.set_debug(self.local_debug_var.get())

    def _find_initial_values(self):
        if not self.fitter: return
        start, end = self.fit_range_start_var.get(), self.fit_range_end_var.get()
        mask = (self.times_original >= start) & (self.times_original <= end)
        t_slice, y_slice = self.times_original[mask], self.data_original[mask]

        fixed_params = {}
        current_params, _ = self._get_current_params_from_entries()
        for i, fix_var in enumerate(self.fix_param_vars_tk):
            if fix_var.get() and not np.isnan(current_params[i]):
                fixed_params[i] = current_params[i]
        
        try:
            y_smooth_for_est = self.fitter._preprocess_data_for_estimation(y_slice)
            initial_guesses = self.fitter.strategy.estimate_initial_parameters(
                t_slice, y_slice, y_smooth_for_est, self.time_zero_main_context,
                fixed_params=fixed_params
            )
            if initial_guesses:
                best_guess = initial_guesses[0]
                for i, val in enumerate(best_guess):
                    if i < len(self.param_vars_tk) and not self.fix_param_vars_tk[i].get():
                        self.param_vars_tk[i].set(round(val, 6))
            else:
                messagebox.showwarning("警告", "未能估算出有效的初始参数。", parent=self)
        except Exception as e:
            messagebox.showerror("错误", f"寻找初始值时出错:\n{e}", parent=self)
            if self.local_debug_var.get():
                traceback.print_exc()
        
        self._update_main_gui_preview()

    def _fit(self):
        if not self.fitter or not self.strategy:
            messagebox.showerror("错误", "拟合器或策略未初始化。", parent=self)
            return

        start, end = self.fit_range_start_var.get(), self.fit_range_end_var.get()
        mask = (self.times_original >= start) & (self.times_original <= end)
        t_slice, y_slice = self.times_original[mask], self.data_original[mask]

        if len(t_slice) < self.strategy.num_parameters + 1:
            messagebox.showwarning("数据不足", "拟合范围内的数据点太少，无法进行拟合。", parent=self)
            return
            
        p_initial_full, p_valid = self._get_current_params_from_entries()
        if not p_valid:
            messagebox.showerror("错误", "所有参数都必须有有效的初始值才能拟合。", parent=self)
            return

        fixed_param_indices = [i for i, var in enumerate(self.fix_param_vars_tk) if var.get()]
        free_param_indices = [i for i, var in enumerate(self.fix_param_vars_tk) if not var.get()]
        
        if not free_param_indices:
            messagebox.showinfo("提示", "所有参数均已固定，无需拟合。", parent=self)
            return

        p_initial_free = [p_initial_full[i] for i in free_param_indices]
        
        bounds_full = self.strategy.get_bounds(t_slice, y_slice)
        lower_bounds_free = [bounds_full[i][0] for i in free_param_indices]
        upper_bounds_free = [bounds_full[i][1] for i in free_param_indices]
        bounds_free = (lower_bounds_free, upper_bounds_free)

        def model_wrapper(t, *p_free):
            p_full = np.array(p_initial_full)
            p_full[free_param_indices] = p_free
            return self.strategy.model_function(t, *p_full)

        try:
            popt_free, pcov_free = curve_fit(
                model_wrapper, t_slice, y_slice,
                p0=p_initial_free,
                bounds=bounds_free,
                method=self.fitting_method_var.get(),
                maxfev=50000, ftol=1e-9, xtol=1e-9
            )

            popt_full = np.array(p_initial_full)
            popt_full[free_param_indices] = popt_free

            errors_full = np.zeros_like(popt_full)
            if pcov_free is not None:
                diag_pcov_free = np.diag(pcov_free)
                errors_free = np.sqrt(np.where(diag_pcov_free >= 0, diag_pcov_free, 0))
                for i, idx in enumerate(free_param_indices):
                    errors_full[idx] = errors_free[i]

            for i, val in enumerate(popt_full):
                self.param_vars_tk[i].set(round(val, 6))

            y_pred = self.strategy.model_function(t_slice, *popt_full)
            ss_res = np.sum((y_slice - y_pred) ** 2)
            ss_tot = np.sum((y_slice - np.mean(y_slice)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
            dof = len(y_slice) - len(p_initial_free)
            chi2_red = ss_res / dof if dof > 0 else np.nan

            self.r_squared_label_var.set(f"R²: {r2:.5f}")
            self.chi_squared_label_var.set(f"\u03C7²_red: {chi2_red:.3e}")

            self.last_fit_result = {
                'success': True, 'params': popt_full.tolist(), 'errors': errors_full.tolist(), 'r_squared': r2,
                'chi_squared_reduced': chi2_red, 'strategy_name_used': self.strategy.name,
                'time_zero_at_fit': self.time_zero_main_context,
                'fit_range_abs_start': start, 'fit_range_abs_end': end,
                'fixed_params_mask': [f.get() for f in self.fix_param_vars_tk]
            }
            self._update_main_gui_preview()

        except RuntimeError as e:
            messagebox.showerror("拟合失败", f"curve_fit 未能收敛或发生运行时错误。\n请尝试调整初始值、范围或固定部分参数。\n\n错误信息: {e}", parent=self)
            self.last_fit_result = None
        except Exception as e:
            messagebox.showerror("拟合失败", f"拟合过程中发生未知错误:\n{e}", parent=self)
            if self.local_debug_var.get():
                traceback.print_exc()
            self.last_fit_result = None
    
    def _get_current_params_from_entries(self) -> Tuple[List[float], bool]:
        params, is_valid = [], True
        for var in self.param_vars_tk:
            try:
                val = var.get()
                if np.isnan(val): is_valid = False
                params.append(val)
            except (ValueError, tk.TclError):
                params.append(np.nan); is_valid = False
        return params, is_valid

    def _update_main_gui_preview(self):
        if self.preview_callback and self.strategy:
            params, is_valid = self._get_current_params_from_entries()
            # If the last fit was successful, preview that, otherwise preview current entries
            if self.last_fit_result and self.last_fit_result.get('success'):
                self.preview_callback(self.last_fit_result['params'], self.strategy.name)
            else:
                self.preview_callback(params if is_valid else None, self.strategy.name)
    
    def _on_apply(self):
        if self.last_fit_result:
            self.apply_callback(self.last_fit_result)
        else:
            messagebox.showwarning("无结果", "没有成功的拟合结果可应用。请先点击“拟合”。", parent=self)

    def _load_from_fit_result(self, fit_result: Dict):
        """Loads the state of the panel from a fit_result dictionary."""
        if not fit_result or not fit_result.get('success'):
            return

        # Set fit range
        self.fit_range_start_var.set(fit_result.get('fit_range_abs_start', self.times_original.min()))
        self.fit_range_end_var.set(fit_result.get('fit_range_abs_end', self.times_original.max()))

        # Set parameters and fixed status
        params = fit_result.get('params', [])
        fixed_mask = fit_result.get('fixed_params_mask', [])
        for i, param_val in enumerate(params):
            if i < len(self.param_vars_tk):
                self.param_vars_tk[i].set(round(param_val, 6))
        for i, is_fixed in enumerate(fixed_mask):
            if i < len(self.fix_param_vars_tk):
                self.fix_param_vars_tk[i].set(is_fixed)
        
        # Set stats
        r2 = fit_result.get('r_squared')
        chi2_red = fit_result.get('chi_squared_reduced')
        if r2 is not None:
            self.r_squared_label_var.set(f"R²: {r2:.5f}")
        if chi2_red is not None:
            self.chi_squared_label_var.set(f"\u03C7²_red: {chi2_red:.3e}")

        # Update preview
        self._update_main_gui_preview()

