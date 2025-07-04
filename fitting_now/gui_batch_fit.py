# gui_batch_fit.py
"""
新的批量拟合对话框，包含逐个参数拟合和批量范围拟合的选项卡。
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import traceback
import threading
from typing import Dict, Any, List, Callable, Optional

from gui_fitting_panel import FittingPanel
import fitting_dispatcher
from fitting_core import ExponentialFitter
from gui_modify_fit_dialog import ModifyFitDialog # To reuse parameter setting UI

class BatchFitDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, all_temperatures: np.ndarray,
                 times_original: np.ndarray, data_matrix: np.ndarray,
                 time_zero_main_context: float, debug_mode: bool, fit_results: Dict,
                 preview_callback: Optional[Callable] = None,
                 finalized_callback: Optional[Callable] = None,
                 switch_preview_context_callback: Optional[Callable] = None,
                 default_strategy_name: Optional[str] = None):
        super().__init__(parent)
        self.transient(parent)
        self.parent = parent
        self.title("批量拟合")
        self.geometry("1100x800")
        
        # Core data
        self.all_temperatures = all_temperatures
        self.times_original = times_original
        self.data_matrix = data_matrix
        self.time_zero_main_context = time_zero_main_context
        self.debug_mode = debug_mode
        self.fit_results_main_ref = fit_results # Reference to main GUI's results
        
        # Callbacks
        self.preview_callback = preview_callback
        self.finalized_callback = finalized_callback
        self.switch_preview_context_callback = switch_preview_context_callback
        
        self.default_strategy_name = default_strategy_name
        
        # Batch Job Management
        self.batch_jobs = []
        self.job_frames_container = None
        self._is_running_batch = False
        
        # Per-Temperature Fitting (Tab 1)
        self.panels: Dict[float, FittingPanel] = {}
        self.active_panel: Optional[FittingPanel] = None
        self.right_frame: Optional[ttk.Frame] = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._create_widgets()

        # Initialize first tab if data is present
        if self.all_temperatures.size > 0:
            self._create_all_panels_for_tab1()
            self.temp_listbox.selection_set(0)
            self._on_temp_select()

    def _create_widgets(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.notebook = ttk.Notebook(top_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Single Temperature Fitting
        self.tab1 = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.tab1, text="逐个指定参数拟合")
        self._create_per_temp_tab()

        # Tab 2: Batch Range Fitting
        self.tab2 = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab2, text="批量范围拟合 (新)")
        self._create_batch_range_tab()

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Separator(bottom_frame).pack(fill=tk.X, pady=5)
        self.close_button = ttk.Button(bottom_frame, text="关闭窗口", command=self._on_close)
        self.close_button.pack(side=tk.RIGHT)
        
    # --- TAB 1: PER-TEMPERATURE FITTING ---
    def _create_per_temp_tab(self):
        paned_window = ttk.PanedWindow(self.tab1, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned_window, padding=5)
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        paned_window.add(left_frame, weight=1)

        self.right_frame = ttk.Frame(paned_window, padding=5)
        self.right_frame.rowconfigure(0, weight=1)
        self.right_frame.columnconfigure(0, weight=1)
        paned_window.add(self.right_frame, weight=4)

        self.temp_listbox = tk.Listbox(left_frame, exportselection=False, relief="flat", borderwidth=0, highlightthickness=0, bg='#3c3f41', fg='#cccccc', selectbackground='#007ACC')
        self.temp_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.temp_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.temp_listbox.config(yscrollcommand=scrollbar.set)
        
        for temp in self.all_temperatures:
            self.temp_listbox.insert(tk.END, f"{temp:.1f} K")
            
        self.temp_listbox.bind('<<ListboxSelect>>', self._on_temp_select)

    def _create_all_panels_for_tab1(self):
        if not self.right_frame: return
        for i, temp in enumerate(self.all_temperatures):
            data_for_temp = self.data_matrix[:, i]
            initial_fit = next((fit for fit in self.fit_results_main_ref.get(temp, []) if fit.get('success')), None)
            apply_callback = lambda fit_res, t=temp: self._handle_panel_apply(t, fit_res)
            panel = FittingPanel(
                parent=self.right_frame, apply_callback=apply_callback, current_temp_val=temp,
                times_original=self.times_original, data_original=data_for_temp,
                time_zero_main_context=self.time_zero_main_context, debug_mode=self.debug_mode,
                preview_callback=self.preview_callback, default_strategy_name=self.default_strategy_name,
                initial_fit_result=initial_fit
            )
            self.panels[temp] = panel

    def _on_temp_select(self, event=None):
        selection = self.temp_listbox.curselection()
        if not selection: return
        selected_index = selection[0]
        selected_temp = self.all_temperatures[selected_index]
        if self.active_panel: self.active_panel.grid_forget()
        new_panel = self.panels.get(selected_temp)
        if new_panel:
            self.active_panel = new_panel
            self.active_panel.grid(row=0, column=0, sticky="nsew")
        if self.switch_preview_context_callback:
            self.switch_preview_context_callback(selected_index)

    def _handle_panel_apply(self, temp: float, fit_result: Dict):
        result_dict = {'action': 'fit_batch_single', 'temperature': temp, 'fit_result': fit_result}
        if self.finalized_callback:
            self.finalized_callback(result_dict)
        messagebox.showinfo("应用成功", f"温度 {temp:.1f} K 的拟合结果已应用到主程序。", parent=self)

    # --- TAB 2: BATCH RANGE FITTING ---
    def _create_batch_range_tab(self):
        # Main container
        main_batch_frame = ttk.Frame(self.tab2)
        main_batch_frame.pack(fill=tk.BOTH, expand=True)
        main_batch_frame.rowconfigure(1, weight=1)
        main_batch_frame.columnconfigure(0, weight=1)

        # Top control frame
        controls_frame = ttk.Frame(main_batch_frame)
        controls_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        ttk.Button(controls_frame, text="➕ 添加拟合任务", command=self._add_batch_job_frame).pack(side=tk.LEFT)
        
        self.run_batch_button = ttk.Button(controls_frame, text="▶️ 运行所有批量拟合", command=self._run_all_batch_fits)
        self.run_batch_button.pack(side=tk.RIGHT, padx=5)

        # Progress bar
        self.progress_label_var = tk.StringVar(value="准备就绪")
        ttk.Label(controls_frame, textvariable=self.progress_label_var, anchor='w').pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.progress_bar = ttk.Progressbar(controls_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Canvas for scrollable job frames
        canvas = tk.Canvas(main_batch_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_batch_frame, orient="vertical", command=canvas.yview)
        self.job_frames_container = ttk.Frame(canvas, padding=5)

        self.job_frames_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.job_frames_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=1, column=0, sticky='nsew')
        scrollbar.grid(row=1, column=1, sticky='ns')

    def _add_batch_job_frame(self):
        job_id = len(self.batch_jobs)
        job_config = {
            "id": job_id,
            "start_temp": tk.StringVar(value=f"{self.all_temperatures[0]:.1f}"),
            "end_temp": tk.StringVar(value=f"{self.all_temperatures[-1]:.1f}"),
            "strategy": tk.StringVar(value=self.default_strategy_name),
            "mode": tk.StringVar(value="统一拟合"),
            "iterative_direction": tk.StringVar(value="low_to_high"),
            "uniform_fixed_params": {}, # {param_index: value}
            "fit_range_start": tk.StringVar(value=f"{self.times_original.min():.4f}" if self.times_original.size > 0 else "0"),
            "fit_range_end": tk.StringVar(value=f"{self.times_original.max():.4f}" if self.times_original.size > 0 else "0"),
        }
        self.batch_jobs.append(job_config)

        # Create the visual frame for this job
        frame = ttk.LabelFrame(self.job_frames_container, text=f"任务 #{job_id + 1}", padding=10)
        frame.pack(fill=tk.X, expand=True, pady=5, padx=5)
        
        # UI elements for the job
        top_row = ttk.Frame(frame)
        top_row.pack(fill=tk.X, expand=True, pady=5)

        # Temp Range
        temp_range_frame = ttk.Frame(top_row)
        temp_range_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(temp_range_frame, text="温度范围:").pack(side=tk.LEFT)
        temp_vals = [f"{t:.1f}" for t in self.all_temperatures]
        ttk.Combobox(temp_range_frame, textvariable=job_config["start_temp"], values=temp_vals, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(temp_range_frame, text="K  至").pack(side=tk.LEFT)
        ttk.Combobox(temp_range_frame, textvariable=job_config["end_temp"], values=temp_vals, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(temp_range_frame, text="K").pack(side=tk.LEFT)

        # Strategy
        strategy_frame = ttk.Frame(top_row)
        strategy_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(strategy_frame, text="拟合模型:").pack(side=tk.LEFT)
        ttk.Combobox(strategy_frame, textvariable=job_config["strategy"], values=fitting_dispatcher.get_available_strategy_names(), state="readonly", width=25).pack(side=tk.LEFT, padx=2)

        # Mode
        mode_frame = ttk.Frame(top_row)
        mode_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(mode_frame, text="拟合模式:").pack(side=tk.LEFT)
        rb1 = ttk.Radiobutton(mode_frame, text="循环拟合", variable=job_config["mode"], value="循环拟合")
        rb2 = ttk.Radiobutton(mode_frame, text="统一拟合", variable=job_config["mode"], value="统一拟合")
        rb1.pack(side=tk.LEFT)
        rb2.pack(side=tk.LEFT)

        options_frame = ttk.Frame(frame, padding=5)
        options_frame.pack(fill=tk.X, expand=True, pady=5)
        
        ttk.Button(top_row, text="🗑️", command=lambda f=frame, jc=job_config: self._remove_batch_job(f, jc), width=3).pack(side=tk.RIGHT)
        
        job_config["mode"].trace_add("write", lambda *args, jc=job_config, of=options_frame: self._update_job_options_ui(jc, of))
        self._update_job_options_ui(job_config, options_frame)

    def _update_job_options_ui(self, job_config, options_frame):
        for widget in options_frame.winfo_children():
            widget.destroy()

        mode = job_config["mode"].get()
        if mode == "循环拟合":
            ttk.Label(options_frame, text="方向:", font=("", 10, "bold")).pack(side=tk.LEFT, padx=5)
            ttk.Radiobutton(options_frame, text="从低到高", variable=job_config["iterative_direction"], value="low_to_high").pack(side=tk.LEFT)
            ttk.Radiobutton(options_frame, text="从高到低", variable=job_config["iterative_direction"], value="high_to_low").pack(side=tk.LEFT)
            ttk.Label(options_frame, text="说明：使用前一温度的拟合结果(包括范围)作为下个温度的初始值。", style="secondary.TLabel").pack(side=tk.LEFT, padx=20)
        
        elif mode == "统一拟合":
            uniform_options_container = ttk.Frame(options_frame)
            uniform_options_container.pack(fill=tk.X, expand=True)

            # Row 1: Fixed Parameters
            fixed_params_frame = ttk.Frame(uniform_options_container)
            fixed_params_frame.pack(fill=tk.X, pady=2)
            
            fixed_params_label_var = tk.StringVar(value="未设置固定参数")
            def update_label():
                fixed_count = len(job_config['uniform_fixed_params'])
                if fixed_count > 0:
                    fixed_params_label_var.set(f"已固定 {fixed_count} 个参数")
                else:
                    fixed_params_label_var.set("未设置固定参数")
            update_label()

            ttk.Button(fixed_params_frame, text="设置统一固定参数...", command=lambda jc=job_config, ul=update_label: self._open_uniform_fixed_param_dialog(jc, ul)).pack(side=tk.LEFT, padx=5)
            ttk.Label(fixed_params_frame, textvariable=fixed_params_label_var).pack(side=tk.LEFT, padx=5)
            
            # Row 2: Fit Range
            fit_range_frame = ttk.Frame(uniform_options_container)
            fit_range_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(fit_range_frame, text="拟合范围 (绝对时间):").pack(side=tk.LEFT, padx=5)
            ttk.Label(fit_range_frame, text="起始:").pack(side=tk.LEFT)
            ttk.Entry(fit_range_frame, textvariable=job_config["fit_range_start"], width=12).pack(side=tk.LEFT, padx=(2, 8))
            ttk.Label(fit_range_frame, text="结束:").pack(side=tk.LEFT)
            ttk.Entry(fit_range_frame, textvariable=job_config["fit_range_end"], width=12).pack(side=tk.LEFT, padx=2)


    def _open_uniform_fixed_param_dialog(self, job_config, update_callback):
        d = tk.Toplevel(self)
        d.transient(self)
        d.title("设置统一固定参数")
        d.geometry("400x500")

        strategy_name = job_config['strategy'].get()
        strategy = fitting_dispatcher.get_strategy(strategy_name)
        if not strategy:
            messagebox.showerror("错误", f"无法加载模型: {strategy_name}", parent=d)
            return

        param_names = strategy.get_parameter_names()
        vars_check = [tk.BooleanVar(value=(i in job_config['uniform_fixed_params'])) for i in range(len(param_names))]
        vars_value = [tk.StringVar(value=(f"{job_config['uniform_fixed_params'][i]:.4e}" if i in job_config['uniform_fixed_params'] else '0.0')) for i in range(len(param_names))]

        frame = ttk.Frame(d, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="参数").grid(row=0, column=0)
        ttk.Label(frame, text="固定?").grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="值").grid(row=0, column=2)

        for i, name in enumerate(param_names):
            ttk.Label(frame, text=f"{name}:").grid(row=i+1, column=0, sticky='w', pady=2)
            ttk.Checkbutton(frame, variable=vars_check[i]).grid(row=i+1, column=1)
            ttk.Entry(frame, textvariable=vars_value[i], width=15).grid(row=i+1, column=2, padx=5)

        def on_apply():
            job_config['uniform_fixed_params'].clear()
            for i in range(len(param_names)):
                if vars_check[i].get():
                    try:
                        val = float(vars_value[i].get())
                        job_config['uniform_fixed_params'][i] = val
                    except ValueError:
                        messagebox.showwarning("无效值", f"参数 '{param_names[i]}' 的值无效，将不会被固定。", parent=d)
            update_callback()
            d.destroy()

        ttk.Button(d, text="应用并关闭", command=on_apply).pack(pady=10)
        self.wait_window(d)

    def _remove_batch_job(self, frame, job_config):
        if messagebox.askyesno("确认", "确定要移除这个拟合任务吗？", parent=self):
            frame.destroy()
            self.batch_jobs.remove(job_config)

    def _run_all_batch_fits(self):
        if self._is_running_batch:
            messagebox.showwarning("进行中", "一个批量拟合任务已在运行。", parent=self)
            return
        if not self.batch_jobs:
            messagebox.showinfo("无任务", "请先添加至少一个拟合任务。", parent=self)
            return

        if not messagebox.askyesno("确认", f"确定要运行 {len(self.batch_jobs)} 个批量拟合任务吗？\n这可能会覆盖现有的拟合结果。", parent=self):
            return

        self._is_running_batch = True
        self.run_batch_button.config(state=tk.DISABLED)
        self.close_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self._batch_fit_thread_worker, args=(self.batch_jobs[:],))
        thread.daemon = True
        thread.start()

    def _batch_fit_thread_worker(self, jobs_to_run):
        total_fits = 0
        all_new_results = {}
        all_failed_fits = []
        
        for job in jobs_to_run:
            try:
                start_temp = float(job['start_temp'].get())
                end_temp = float(job['end_temp'].get())
                indices = np.where((self.all_temperatures >= start_temp) & (self.all_temperatures <= end_temp))[0]
                total_fits += len(indices)
            except ValueError:
                pass 
        
        self.progress_bar['maximum'] = total_fits
        fits_done = 0
        
        for i, job_config in enumerate(jobs_to_run):
            self.after(0, self.progress_label_var.set, f"正在运行任务 {i+1}/{len(jobs_to_run)}...")
            
            try:
                start_temp = float(job_config['start_temp'].get())
                end_temp = float(job_config['end_temp'].get())
                strategy_name = job_config['strategy'].get()
                mode = job_config['mode'].get()
                
                strategy = fitting_dispatcher.get_strategy(strategy_name)
                if not strategy:
                    all_failed_fits.append(f"任务 {i+1}: 模型 '{strategy_name}' 未找到")
                    continue

                fitter = ExponentialFitter(strategy)
                fitter.set_debug(self.debug_mode)
                
                temp_indices = np.where((self.all_temperatures >= start_temp) & (self.all_temperatures <= end_temp))[0]
                if len(temp_indices) == 0: continue

                temps_for_job = self.all_temperatures[temp_indices]
                
                if mode == "循环拟合":
                    new_res, failed = self._execute_iterative_fit_job(job_config, temps_for_job, fitter, fits_done, total_fits)
                else:
                    new_res, failed = self._execute_uniform_fit_job(job_config, temps_for_job, fitter, fits_done, total_fits)

                all_new_results.update(new_res)
                all_failed_fits.extend(failed)
                fits_done += len(temps_for_job)

            except Exception as e:
                all_failed_fits.append(f"任务 {i+1} 发生严重错误: {e}")
                traceback.print_exc()

        self.after(0, self._finalize_batch_run, all_new_results, all_failed_fits)

    def _execute_iterative_fit_job(self, job_config, temps, fitter, fits_done_so_far, total_fits):
        direction = job_config['iterative_direction'].get()
        sorted_temps = sorted(temps, reverse=(direction == 'high_to_low'))
        
        new_results = {}
        failed_fits = []
        
        # Get starting parameters and crucially, the fit range
        start_temp = sorted_temps[0]
        
        existing_fits_for_start = [r for r in self.fit_results_main_ref.get(start_temp, []) if r.get('success')]
        if not existing_fits_for_start:
            failed_fits.append(f"{start_temp:.1f} K (循环拟合起始点，无预先拟合结果)")
            for k in range(len(sorted_temps)): self.after(0, self.progress_bar.step, 1)
            return {}, failed_fits

        start_fit = max(existing_fits_for_start, key=lambda r: r.get('r_squared', 0))
        
        # Inherit all crucial properties from the starting fit
        last_successful_params = start_fit['params']
        fixed_mask = start_fit['fixed_params_mask']
        start_time = start_fit.get('fit_range_abs_start', self.times_original.min())
        end_time = start_fit.get('fit_range_abs_end', self.times_original.max())

        for i, temp in enumerate(sorted_temps):
            current_fit_index = fits_done_so_far + i
            self.after(0, self.progress_label_var.set, f"循环拟合: {temp:.1f} K ({current_fit_index+1}/{total_fits})")
            
            if i == 0:
                new_results[temp] = start_fit
                self.after(0, self.progress_bar.step, 1)
                continue
            
            temp_idx_in_matrix = np.where(self.all_temperatures == temp)[0][0]
            full_data_slice = self.data_matrix[:, temp_idx_in_matrix]

            # Apply the inherited fit range
            time_mask = (self.times_original >= start_time) & (self.times_original <= end_time)
            t_slice = self.times_original[time_mask]
            y_slice = full_data_slice[time_mask]

            fit_success = False
            try_count = 0
            while not fit_success and try_count < 2:
                try:
                    initial_guess = last_successful_params if try_count == 0 else None
                    if try_count == 1:
                         y_smooth_est = fitter._preprocess_data_for_estimation(y_slice)
                         estimations = fitter.strategy.estimate_initial_parameters(
                             t_slice, y_slice, y_smooth_est, self.time_zero_main_context)
                         if estimations:
                            initial_guess = estimations[0]
                            for p_idx, is_fixed in enumerate(fixed_mask):
                                if is_fixed: initial_guess[p_idx] = last_successful_params[p_idx]
                         else:
                            raise RuntimeError("无法估算初始值(重试)")
                    
                    popt, errors, r2 = self._fit_with_fixed_params(fitter, t_slice, y_slice, initial_guess, fixed_mask)
                    
                    fit_res = {
                        'success': True, 'params': popt, 'errors': errors, 'r_squared': r2,
                        'strategy_name_used': fitter.strategy.name, 'time_zero_at_fit': self.time_zero_main_context,
                        'fit_range_abs_start': start_time,
                        'fit_range_abs_end': end_time,
                        'fixed_params_mask': fixed_mask
                    }
                    new_results[temp] = fit_res
                    last_successful_params = popt
                    fit_success = True
                
                except Exception as e:
                    try_count += 1
                    if try_count >= 2:
                        failed_fits.append(f"{temp:.1f} K (循环拟合失败: {e})")
            
            self.after(0, self.progress_bar.step, 1)
        return new_results, failed_fits

    def _execute_uniform_fit_job(self, job_config, temps, fitter, fits_done_so_far, total_fits):
        new_results = {}
        failed_fits = []
        
        fixed_params_config = job_config['uniform_fixed_params']
        num_params = fitter.strategy.num_parameters
        fixed_mask = [(i in fixed_params_config) for i in range(num_params)]
        try:
            start_time = float(job_config['fit_range_start'].get())
            end_time = float(job_config['fit_range_end'].get())
        except ValueError:
            failed_fits.append(f"整个任务失败: 无效的拟合时间范围。")
            for k in range(len(temps)): self.after(0, self.progress_bar.step, 1)
            return {}, failed_fits

        for i, temp in enumerate(temps):
            current_fit_index = fits_done_so_far + i
            self.after(0, self.progress_label_var.set, f"统一拟合: {temp:.1f} K ({current_fit_index+1}/{total_fits})")
            
            temp_idx_in_matrix = np.where(self.all_temperatures == temp)[0][0]
            full_data_slice = self.data_matrix[:, temp_idx_in_matrix]

            try:
                time_mask = (self.times_original >= start_time) & (self.times_original <= end_time)
                t_slice = self.times_original[time_mask]
                y_slice = full_data_slice[time_mask]

                if len(t_slice) < num_params + 1:
                    raise RuntimeError("拟合范围内数据点不足")

                y_smooth_est = fitter._preprocess_data_for_estimation(y_slice)
                initial_guesses = fitter.strategy.estimate_initial_parameters(
                    t_slice, y_slice, y_smooth_est, self.time_zero_main_context, fixed_params=fixed_params_config
                )
                if not initial_guesses:
                    raise RuntimeError("无法估算初始值")
                
                initial_guess = initial_guesses[0]

                for param_idx, fixed_val in fixed_params_config.items():
                    if 0 <= param_idx < len(initial_guess):
                        initial_guess[param_idx] = fixed_val

                popt, errors, r2 = self._fit_with_fixed_params(fitter, t_slice, y_slice, initial_guess, fixed_mask)

                fit_res = {
                    'success': True, 'params': popt, 'errors': errors, 'r_squared': r2,
                    'strategy_name_used': fitter.strategy.name, 'time_zero_at_fit': self.time_zero_main_context,
                    'fit_range_abs_start': start_time,
                    'fit_range_abs_end': end_time,
                    'fixed_params_mask': fixed_mask
                }
                new_results[temp] = fit_res

            except Exception as e:
                failed_fits.append(f"{temp:.1f} K (统一拟合失败: {e})")

            self.after(0, self.progress_bar.step, 1)
            
        return new_results, failed_fits
    
    def _fit_with_fixed_params(self, fitter: ExponentialFitter, t_data, y_data, initial_params, fixed_mask):
        from scipy.optimize import curve_fit
        
        free_param_indices = [i for i, is_fixed in enumerate(fixed_mask) if not is_fixed]
        if not free_param_indices:
            popt_full = initial_params
            errors_full = np.zeros_like(popt_full)
            y_pred = fitter.strategy.model_function(t_data, *popt_full)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
            return popt_full, errors_full, r2

        p_initial_free = [initial_params[i] for i in free_param_indices]
        bounds_full = fitter.strategy.get_bounds(t_data, y_data)
        lower_bounds_free = [bounds_full[i][0] for i in free_param_indices]
        upper_bounds_free = [bounds_full[i][1] for i in free_param_indices]
        bounds_free = (lower_bounds_free, upper_bounds_free)

        def model_wrapper(t, *p_free):
            p_full = np.array(initial_params, dtype=float)
            p_full[free_param_indices] = p_free
            return fitter.strategy.model_function(t, *p_full)

        popt_free, pcov_free = curve_fit(
            model_wrapper, t_data, y_data,
            p0=p_initial_free, bounds=bounds_free, method='trf',
            maxfev=50000, ftol=1e-9, xtol=1e-9
        )
        popt_full = np.array(initial_params, dtype=float)
        popt_full[free_param_indices] = popt_free

        errors_full = np.full_like(popt_full, np.nan)
        if pcov_free is not None:
            diag_pcov_free = np.diag(pcov_free)
            errors_free = np.sqrt(np.where(diag_pcov_free >= 0, diag_pcov_free, 0))
            for i, idx in enumerate(free_param_indices):
                errors_full[idx] = errors_free[i]

        y_pred = fitter.strategy.model_function(t_data, *popt_full)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
        
        return popt_full.tolist(), errors_full.tolist(), r2
    
    def _finalize_batch_run(self, new_results, failed_fits):
        self._is_running_batch = False
        self.run_batch_button.config(state=tk.NORMAL)
        self.close_button.config(state=tk.NORMAL)
        self.progress_label_var.set("批量拟合完成!")
        
        summary_message = f"批量拟合完成。\n\n成功: {len(new_results)} 个\n失败: {len(failed_fits)} 个"
        if failed_fits:
            summary_message += "\n\n失败详情 (前10条):\n" + "\n".join(failed_fits[:10])
        messagebox.showinfo("批量拟合结果", summary_message, parent=self)
        
        if new_results and self.finalized_callback:
            result_dict = {'action': 'fit_batch_multiple', 'new_results': new_results}
            self.finalized_callback(result_dict)

    def _on_close(self):
        if self._is_running_batch:
            if not messagebox.askyesno("确认", "批量拟合正在进行中，关闭窗口将中止该过程。确定要关闭吗？", parent=self):
                return
        
        if self.finalized_callback:
            self.finalized_callback({'action': 'cancel_batch'})
        self.destroy()
