# gui_modify_fit_dialog.py
"""
用于修改一个已存在的拟合结果的对话框。
(已重构，移除独立的样式定义，继承主程序样式)
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import traceback
from typing import Optional, Callable, Dict, Any, List, Tuple
from scipy.optimize import curve_fit

from fitting_core import ExponentialFitter
from fitting_strategies.base_fitting_strategy import FittingStrategy
import fitting_dispatcher

class ModifyFitDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, main_app_instance, existing_fit_result: Dict,
                 times_original: np.ndarray, data_original: np.ndarray,
                 preview_callback: Optional[Callable] = None,
                 finalized_callback: Optional[Callable] = None):
        super().__init__(parent)
        self.transient(parent)
        self.parent = parent
        self.main_app = main_app_instance

        self.existing_fit_result = existing_fit_result.copy()
        self.times_original = times_original
        self.data_original = data_original
        self.preview_callback = preview_callback
        self.finalized_callback = finalized_callback

        self.strategy: Optional[FittingStrategy] = None
        self.fitter: Optional[ExponentialFitter] = None
        
        self.param_vars_tk: List[tk.DoubleVar] = []
        self.fix_param_vars_tk: List[tk.BooleanVar] = []
        self.fit_range_start_var = tk.DoubleVar()
        self.fit_range_end_var = tk.DoubleVar()
        self.fitting_method_var = tk.StringVar(value='trf')
        self.r_squared_label_var = tk.StringVar()
        self.chi_squared_label_var = tk.StringVar()
        self.local_debug_var = tk.BooleanVar(value=self.main_app.debug_var.get())
        self.strategy_name_var = tk.StringVar()

        self.title(f"更改拟合 (ID: {self.existing_fit_result.get('id', 'N/A')[:8]}...)")
        self.geometry("700x750")
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # --- MODIFIED: Set background color directly. Styles are inherited. ---
        self.config(bg='#2d2d2d')

        self._create_widgets()
        self._initialize_from_existing_fit()
        
    # --- DELETED: apply_styles method is no longer needed ---

    def _create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # The canvas should inherit the bg color, but setting it explicitly is safer.
        self.canvas = tk.Canvas(main_frame, borderwidth=0, highlightthickness=0, bg='#2d2d2d')
        self.scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, padding=10)
        
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"), "+")

        model_frame = ttk.LabelFrame(self.scrollable_frame, text="拟合模型", padding=5)
        model_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(model_frame, textvariable=self.strategy_name_var, font=("Microsoft YaHei UI", 12, "bold")).pack(fill=tk.X, expand=True, padx=5, pady=5)

        self.params_outer_frame = ttk.LabelFrame(self.scrollable_frame, text="拟合参数", padding=5)
        self.params_outer_frame.pack(fill=tk.X, pady=5, padx=5)
        self.params_frame = ttk.Frame(self.params_outer_frame, padding=5)
        self.params_frame.pack(fill=tk.X, expand=True)

        fit_range_frame = ttk.LabelFrame(self.scrollable_frame, text="拟合范围 (绝对时间)", padding=5)
        fit_range_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(fit_range_frame, text="起始:").grid(row=0, column=0)
        ttk.Entry(fit_range_frame, textvariable=self.fit_range_start_var, width=12).grid(row=0, column=1, padx=5)
        ttk.Label(fit_range_frame, text="结束:").grid(row=0, column=2)
        ttk.Entry(fit_range_frame, textvariable=self.fit_range_end_var, width=12).grid(row=0, column=3, padx=5)

        fit_method_frame = ttk.LabelFrame(self.scrollable_frame, text="拟合算法 (curve_fit)", padding=5)
        fit_method_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Combobox(fit_method_frame, textvariable=self.fitting_method_var,
                     values=['trf', 'dogbox', 'lm'], state="readonly", width=12).pack(side=tk.LEFT, padx=5)

        stats_frame = ttk.LabelFrame(self.scrollable_frame, text="统计与调试", padding=5)
        stats_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Label(stats_frame, textvariable=self.r_squared_label_var).pack(anchor=tk.W)
        ttk.Label(stats_frame, textvariable=self.chi_squared_label_var).pack(anchor=tk.W)
        ttk.Checkbutton(stats_frame, text="显示详细调试信息", variable=self.local_debug_var,
                        command=self._toggle_debug).pack(anchor=tk.W)

        action_frame = ttk.Frame(self, padding=(0, 10))
        action_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(action_frame, text="重新拟合", command=self._fit).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="寻找初始值", command=self._find_initial_values).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="应用更改并关闭", command=self._on_apply_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(action_frame, text="取消", command=self._on_cancel).pack(side=tk.RIGHT)

    def _initialize_from_existing_fit(self):
        strategy_name = self.existing_fit_result.get('strategy_name_used')
        if not strategy_name:
            messagebox.showerror("错误", "该拟合结果缺少模型名称，无法修改。", parent=self)
            self._on_cancel()
            return

        self.strategy = fitting_dispatcher.get_strategy(strategy_name)
        if not self.strategy:
            messagebox.showerror("错误", f"无法加载模型: {strategy_name}", parent=self)
            self._on_cancel()
            return
        
        self.strategy_name_var.set(strategy_name)
        self.fitter = ExponentialFitter(self.strategy)
        self.fitter.set_debug(self.local_debug_var.get())

        self._build_param_entries()

        params = self.existing_fit_result.get('params', [])
        for i, val in enumerate(params):
            if i < len(self.param_vars_tk):
                self.param_vars_tk[i].set(round(val, 6))

        fixed_mask = self.existing_fit_result.get('fixed_params_mask', [])
        for i, is_fixed in enumerate(fixed_mask):
            if i < len(self.fix_param_vars_tk):
                self.fix_param_vars_tk[i].set(is_fixed)

        start = self.existing_fit_result.get('fit_range_abs_start', self.times_original.min() if self.times_original.size > 0 else 0)
        end = self.existing_fit_result.get('fit_range_abs_end', self.times_original.max() if self.times_original.size > 0 else 0)
        self.fit_range_start_var.set(round(start, 4))
        self.fit_range_end_var.set(round(end, 4))
        
        r2 = self.existing_fit_result.get('r_squared', 0)
        chi2 = self.existing_fit_result.get('chi_squared_reduced', np.nan)
        self.r_squared_label_var.set(f"R²: {r2:.5f}")
        self.chi_squared_label_var.set(f"\u03C7²_red: {chi2:.3e}")

        self._update_main_gui_preview()
        
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
                t_slice, y_slice, y_smooth_for_est, self.main_app.time_zero_var.get(),
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
            traceback.print_exc()
        
        self._update_main_gui_preview()

    def _fit(self):
        if not self.fitter or not self.strategy:
            messagebox.showerror("错误", "拟合器或策略未初始化。", parent=self)
            return

        start, end = self.fit_range_start_var.get(), self.fit_range_end_var.get()
        mask = (self.times_original >= start) & (self.times_original <= end)
        t_slice, y_slice = self.times_original[mask], self.data_original[mask]

        if len(t_slice) < self.strategy.num_parameters:
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
            self._update_stats_and_result(p_initial_full, np.zeros_like(p_initial_full), t_slice, y_slice, start, end)
            self._update_main_gui_preview()
            return

        p_initial_free = [p_initial_full[i] for i in free_param_indices]
        
        bounds_full = self.strategy.get_bounds(t_slice, y_slice)
        lower_bounds_free = [bounds_full[i][0] for i in free_param_indices]
        upper_bounds_free = [bounds_full[i][1] for i in free_param_indices]
        bounds_free = (lower_bounds_free, upper_bounds_free)

        def model_wrapper(t, *p_free):
            p_full = np.array(p_initial_full, dtype=float)
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

            popt_full = np.array(p_initial_full, dtype=float)
            popt_full[free_param_indices] = popt_free

            errors_full = np.zeros_like(popt_full)
            if pcov_free is not None:
                diag_pcov_free = np.diag(pcov_free)
                errors_free = np.sqrt(np.where(diag_pcov_free >= 0, diag_pcov_free, 0))
                for i, idx in enumerate(free_param_indices):
                    errors_full[idx] = errors_free[i]

            for i, val in enumerate(popt_full):
                self.param_vars_tk[i].set(round(val, 6))

            self._update_stats_and_result(popt_full, errors_full, t_slice, y_slice, start, end)
            self._update_main_gui_preview()

        except RuntimeError as e:
            messagebox.showerror("拟合失败", f"curve_fit 未能收敛或发生运行时错误。\n请尝试调整初始值、范围或固定部分参数。\n\n错误信息: {e}", parent=self)
        except Exception as e:
            messagebox.showerror("拟合失败", f"拟合过程中发生未知错误:\n{e}", parent=self)
            traceback.print_exc()
            
    def _update_stats_and_result(self, params, errors, t_slice, y_slice, start, end):
        y_pred = self.strategy.model_function(t_slice, *params)
        ss_res = np.sum((y_slice - y_pred) ** 2)
        ss_tot = np.sum((y_slice - np.mean(y_slice)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
        dof = len(y_slice) - len([v for v in self.fix_param_vars_tk if not v.get()])
        chi2_red = ss_res / dof if dof > 0 else np.nan

        self.r_squared_label_var.set(f"R²: {r2:.5f}")
        self.chi_squared_label_var.set(f"\u03C7²_red: {chi2_red:.3e}")

        self.existing_fit_result.update({
            'success': True, 'params': params, 'errors': errors,
            'r_squared': r2, 'chi_squared_reduced': chi2_red,
            'fit_range_abs_start': start, 'fit_range_abs_end': end,
            'fixed_params_mask': [f.get() for f in self.fix_param_vars_tk]
        })
    
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
            self.preview_callback(params if is_valid else None, self.strategy.name)
            
    def _toggle_debug(self):
        if self.fitter: self.fitter.set_debug(self.local_debug_var.get())

    def _on_apply_and_close(self):
        if not self.existing_fit_result.get('success'):
             messagebox.showwarning("无有效结果", "当前没有有效的拟合结果可以应用。请先成功拟合一次。", parent=self)
             return
             
        result = {'success': True, 'updated_fit_result': self.existing_fit_result}
        if self.finalized_callback:
            self.finalized_callback(result)
        self.destroy()

    def _on_cancel(self):
        if self.finalized_callback:
            self.finalized_callback({'success': False})
        self.destroy()
