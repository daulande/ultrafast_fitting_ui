# gui_main.py
"""
多指数拟合程序主界面 (已重构以支持多重拟合和结果修改)
"""
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkFont
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import os
from pathlib import Path 
import traceback
from typing import Optional, List, Dict, Any
import uuid # 用于生成唯一ID
import pickle # <-- 新增导入, 用于项目保存/加载
import subprocess # <-- 新增导入, 用于运行curve_main.py
from gui_export_selection_dialog import ExportSelectionDialog

# --- Matplotlib 中文显示设置 (已优化) ---
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Dengxian', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    print("Matplotlib 字体已设置为 'Microsoft YaHei' (若不可用则备选 'SimHei' 等).")
except Exception as e:
    print(f"警告: 设置 Matplotlib 中文字体失败: {e}")
# --- End Matplotlib 设置 ---


# --- Refactored Imports ---
try:
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from io_functions import read_data_file, export_parameters_summary, export_curves_data
    from fitting_core import ExponentialFitter
    import fitting_dispatcher
    from fitting_strategies.base_fitting_strategy import FittingStrategy
    from gui_current_fit import CurrentFitWindow
    from gui_batch_fit import BatchFitDialog
    from gui_modify_fit_dialog import ModifyFitDialog
except ImportError as e:
    messagebox.showerror("启动错误", f"无法导入必要的模块: {e}\n请确保所有项目文件都在正确的位置。")
    traceback.print_exc()
    sys.exit(1)
# --- End Refactored Imports ---


# ------------------- 集中样式配置函数 -------------------
def setup_dark_theme(root):
    """配置整个应用的暗色主题样式"""
    root.tk_setPalette(background='#2d2d2d', foreground='white')
    
    default_font = tkFont.Font(family="Microsoft YaHei", size=10)
    label_font = tkFont.Font(family="Microsoft YaHei", size=10, weight="bold")
    button_font = tkFont.Font(family="Microsoft YaHei", size=10)
    checkbutton_font = tkFont.Font(family="Microsoft YaHei", size=9)
    status_font = tkFont.Font(family="Microsoft YaHei", size=9)
    listbox_font = tkFont.Font(family="Microsoft YaHei", size=12) 


    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')

    bg_color = '#2d2d2d'
    fg_color = '#cccccc'
    entry_bg = '#3c3f41'
    entry_fg = '#bbbbbb'
    button_bg = '#4a4d4f'
    button_fg = 'white'
    select_bg = '#007ACC'

    style.configure('.', background=bg_color, foreground=fg_color, font=default_font, padding=3)
    style.configure('TFrame', background=bg_color)
    style.configure('TLabel', font=default_font, foreground=fg_color, background=bg_color)
    style.configure('secondary.TLabel', foreground='#999999')
    
    style.configure('TButton', font=button_font, foreground=button_fg, background=button_bg, padding=(6, 3))
    style.map('TButton', background=[('active', '#5a5d5f'), ('pressed', '#6a6d6f')], foreground=[('active', 'white')])
    
    style.configure('Treeview', background=entry_bg, foreground=fg_color, fieldbackground=entry_bg, rowheight=28)
    style.map('Treeview', background=[('selected', select_bg)], foreground=[('selected', 'white')])
    style.configure("Treeview.Heading", font=button_font, background=button_bg, foreground=button_fg)
    style.configure('TCheckbutton', font=checkbutton_font, foreground=fg_color, background=bg_color, padding=(3,1))
    style.map('TCheckbutton', background=[('active', bg_color)], indicatorcolor=[('selected', select_bg), ('!selected', '#555555')])
    style.configure('Toggle.TButton', padding=(0, 2), font=tkFont.Font(family="Arial", size=1, weight="bold"))
    style.configure('TRadiobutton', font=default_font, foreground=fg_color, background=bg_color)
    style.map('TRadiobutton', background=[('active', bg_color)], indicatorcolor=[('selected', select_bg), ('!selected', '#555555')])
    style.configure('TEntry', fieldbackground=entry_bg, foreground=entry_fg, insertcolor=fg_color, font=default_font)
    style.map('TEntry', fieldbackground=[('disabled', '#333333')], foreground=[('disabled', '#777777')])
    style.configure('TCombobox', fieldbackground=entry_bg, foreground=fg_color, selectbackground=entry_bg, selectforeground=fg_color, font=default_font)
    style.map('TCombobox', fieldbackground=[('readonly', entry_bg)], selectbackground=[('readonly', entry_bg)], foreground=[('readonly', entry_fg)])
    style.configure('TLabelframe', font=label_font, background=bg_color, borderwidth=1, relief="solid")
    style.configure('TLabelframe.Label', font=label_font, foreground=fg_color, background=bg_color)
    style.configure('Vertical.TScrollbar', background=button_bg, troughcolor=bg_color, bordercolor=bg_color, arrowcolor=fg_color)
    style.map('Vertical.TScrollbar', background=[('active', '#5a5d5f')], arrowcolor=[('active', 'white')])
    style.configure("TNotebook", background=bg_color, borderwidth=0)
    style.configure("TNotebook.Tab", background=button_bg, foreground=button_fg, padding=[8, 4], font=button_font)
    style.map("TNotebook.Tab", background=[("selected", select_bg), ("active", '#5a5d5f')], foreground=[("selected", "white"), ("active", "white")])
    style.configure('TNotebook.Tab', focuscolor=style.lookup('TNotebook.Tab', 'background'))
    
    return {
        'default_font': default_font,
        'status_font': status_font,
        'listbox_font': listbox_font,
        'plot_bg_color': '#333333',
        'plot_title_font_props': {'fontsize': 16, 'fontweight': 'bold', 'color': 'white', 'fontfamily': 'Microsoft YaHei'},
        'plot_label_font_props': {'fontsize': 13, 'color': 'lightgray', 'fontfamily': 'Microsoft YaHei'},
        'plot_legend_font_props': {'size': 10, 'family': 'Microsoft YaHei'}
    }
# --------------------------------------------------------------------------


class ExponentialFittingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("多指数拟合程序 (高级版)")
        self.root.geometry("1600x900")
        
        # 启用窗口大小调整
        self.root.minsize(1200, 700)  # 设置最小窗口尺寸
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.styles = setup_dark_theme(self.root)
        self.default_font = self.styles['default_font']
        self.status_font = self.styles['status_font']
        self.listbox_font = self.styles['listbox_font'] 
        self.plot_bg_color = self.styles['plot_bg_color']
        self.plot_title_font_props = self.styles['plot_title_font_props']
        self.plot_label_font_props = self.styles['plot_label_font_props']
        self.plot_legend_font_props = self.styles['plot_legend_font_props']

        self.data_loaded = False
        self.temperatures = np.array([])
        self.times = np.array([])
        self.data_matrix = np.array([[]])
        
        self.fit_results: Dict[float, List[Dict[str, Any]]] = {}
        
        self.current_temp_index = 0
        self.data_filename = ""

        self.plot_toolbar_backend = None
        self.coord_label = None

        self.draggable_t0_line = None
        self.dragging_t0_line = False
        self.manual_time_zero_var = tk.DoubleVar(value=0.0)
        self.select_time_zero_from_data_var = tk.BooleanVar(value=False)
        
        self.time_unit_label = "ps"
        self.signal_unit_label = "10\u207B\u00B3"

        self.available_strategy_names = fitting_dispatcher.get_available_strategy_names()
        self.strategy_name_var = tk.StringVar()
        self.current_strategy: Optional[FittingStrategy] = None
        if self.available_strategy_names:
            self.strategy_name_var.set(self.available_strategy_names[0])
            self.current_strategy = fitting_dispatcher.get_strategy(self.available_strategy_names[0])
        self.strategy_name_var.trace_add("write", self._on_strategy_change)

        self.active_fitting_dialog_preview_params = None
        self.active_fitting_dialog_preview_strategy_name: Optional[str] = None
        self.is_dialog_active = False
        self.current_fitting_dialog_window = None

        self.controls_to_manage_state = []
        
        # 将visible_fit_ids从集合改为字典，每个温度一个可见性集合
        self.visible_fit_ids = {}
        
        self.side_panel_width = 350

        self.create_widgets()
        self._update_strategy_dependent_ui()
        self.results_frame_visible = False
        self.toggle_results_panel()
        
    def create_widgets(self):
        menubar = tk.Menu(self.root, font=self.default_font, bg='#2d2d2d', fg='white', activebackground='#007ACC', activeforeground='white')
        self.root.config(menu=menubar)

        self.file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=self.file_menu)
        self.file_menu.add_command(label="打开数据文件", command=self.load_data_file)
        self.file_menu.add_command(label="打开项目...", command=self.load_project)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="保存项目...", command=self.save_project)
        self.file_menu.add_command(label="保存结果为文本", command=self.save_results)
        self.file_menu.add_command(label="导出曲线", command=self.export_curves)
        self.file_menu.add_command(label="导出多Sheet格式CSV", command=self.export_multi_sheet_csv)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="退出", command=self.root.quit)

        self.tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=self.tools_menu)
        self.debug_var = tk.BooleanVar(value=False)
        self.tools_menu.add_checkbutton(label="调试模式", variable=self.debug_var, command=self._on_debug_mode_toggle)
        self.tools_menu.add_command(label="观察拟合曲线变换", command=self.run_curve_tool)

        self.help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=self.help_menu)
        self.help_menu.add_command(label="使用说明", command=self.show_help)
        self.help_menu.add_command(label="关于", command=self.show_about)

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # 配置主框架的行列权重，实现等比例缩放
        self.main_frame.columnconfigure(0, weight=0, minsize=160)  # 温度列表区域
        self.main_frame.columnconfigure(1, weight=3)  # 图表区域，给予更多空间
        self.main_frame.columnconfigure(2, weight=0, minsize=30)  # 切换按钮区域
        self.main_frame.rowconfigure(0, weight=0)  # 控制面板行
        self.main_frame.rowconfigure(1, weight=1)  # 主内容区域
        self.main_frame.rowconfigure(2, weight=0)  # 状态栏行

        control_frame = ttk.LabelFrame(self.main_frame, text="控制面板", padding=(10,5))
        control_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(5,10), padx=5)
        
        # 控制面板的列配置
        control_frame.columnconfigure(0, weight=1)  # 文件区
        control_frame.columnconfigure(1, weight=1)  # 参数区
        control_frame.columnconfigure(2, weight=1)  # 操作区
        control_frame.columnconfigure(3, weight=1)  # 导航区

        file_group = ttk.LabelFrame(control_frame, text="数据文件", padding=5)
        file_group.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ns")
        self.select_data_button = ttk.Button(file_group, text="选择数据文件", command=self.load_data_file)
        self.select_data_button.pack(pady=2, anchor='w')
        self.file_label = ttk.Label(file_group, text="未选择文件", wraplength=180)
        self.file_label.pack(pady=2, anchor='w', fill=tk.X, expand=True)

        param_group = ttk.LabelFrame(control_frame, text="拟合参数", padding=5)
        param_group.grid(row=0, column=1, padx=5, pady=5, sticky="ns")
        
        ttk.Label(param_group, text="默认模型:").grid(row=0, column=0, sticky='w', pady=2)
        self.strategy_combobox = ttk.Combobox(param_group, textvariable=self.strategy_name_var,
                                             values=self.available_strategy_names, state="readonly", width=25)
        if not self.available_strategy_names:
            self.strategy_combobox.set("无可用模型")
            self.strategy_combobox.config(state="disabled")
        self.strategy_combobox.grid(row=0, column=1, columnspan=2, sticky='ew', padx=5, pady=2)
        
        ttk.Label(param_group, text="时间零点 (T0):").grid(row=1, column=0, sticky='w', pady=2)
        self.time_zero_var = tk.DoubleVar(value=0.0)
        self.time_zero_var.trace_add("write", lambda *args: self.plot_data(preserve_zoom=True))
        self.time_zero_entry = ttk.Entry(param_group, textvariable=self.time_zero_var, width=12)
        self.time_zero_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        self.refresh_button = ttk.Button(param_group, text="刷新", command=self._on_refresh_button_click)
        self.refresh_button.grid(row=1, column=2, sticky='ew', padx=(0, 5), pady=2)
        
        self.select_time_zero_from_data_checkbutton = ttk.Checkbutton(param_group, text="启用拖拽选择T0", variable=self.select_time_zero_from_data_var)
        self.select_time_zero_from_data_checkbutton.grid(row=2, column=0, columnspan=2, sticky='w', pady=(5,2))
        self.manual_time_zero_entry = ttk.Entry(param_group, textvariable=self.manual_time_zero_var, width=12, state='disabled')
        self.manual_time_zero_entry.grid(row=2, column=2, sticky='ew', padx=(0, 5), pady=(5,2))
        param_group.columnconfigure(1, weight=1)

        action_group = ttk.LabelFrame(control_frame, text="操作", padding=5)
        action_group.grid(row=0, column=2, padx=5, pady=5, sticky="ns")
        self.new_fit_button = ttk.Button(action_group, text="新建拟合", command=self.open_current_fit_window)
        self.new_fit_button.grid(row=0, column=0, padx=5, pady=2, sticky='ew')
        self.batch_fit_button = ttk.Button(action_group, text="批量拟合", command=self.open_batch_fit_dialog)
        self.batch_fit_button.grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        self.save_results_button = ttk.Button(action_group, text="保存结果", command=self.save_results)
        self.save_results_button.grid(row=1, column=0, padx=5, pady=2, sticky='ew')
        self.export_curves_button = ttk.Button(action_group, text="导出曲线", command=self.export_curves)
        self.export_curves_button.grid(row=1, column=1, padx=5, pady=2, sticky='ew')
        self.export_csv_button = ttk.Button(action_group, text="导出多Sheet格式CSV", command=self.export_multi_sheet_csv)
        self.export_csv_button.grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky='ew')
        self.curve_tool_button = ttk.Button(action_group, text="观察拟合曲线变换", command=self.run_curve_tool)
        self.curve_tool_button.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky='ew')
        
        nav_group = ttk.LabelFrame(control_frame, text="图形导航", padding=5)
        nav_group.grid(row=0, column=3, padx=(5,0), pady=5, sticky="nsew")
        self.zoom_button = ttk.Button(nav_group, text="缩放", command=self.activate_zoom_mode)
        self.zoom_button.grid(row=0, column=0, padx=3, pady=2, sticky="ew")
        self.pan_button = ttk.Button(nav_group, text="移动", command=self.activate_pan_mode)
        self.pan_button.grid(row=0, column=1, padx=3, pady=2, sticky="ew")
        self.reset_view_button = ttk.Button(nav_group, text="恢复", command=self.reset_plot_view)
        self.reset_view_button.grid(row=1, column=0, padx=3, pady=2, sticky="ew")
        self.back_view_button = ttk.Button(nav_group, text="上一步", command=self.go_back_plot_view)
        self.back_view_button.grid(row=1, column=1, padx=3, pady=2, sticky="ew")
        nav_group.columnconfigure((0,1), weight=1)
        
        temp_frame = ttk.LabelFrame(self.main_frame, text="温度选择", padding="10")
        temp_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        temp_frame.rowconfigure(0, weight=1)
        temp_frame.columnconfigure(0, weight=1)
        self.temp_listbox = tk.Listbox(temp_frame, width=25, exportselection=False, relief="flat", borderwidth=0, highlightthickness=0, bg='#3c3f41', fg='#cccccc', selectbackground='#007ACC', font=self.listbox_font)
        self.temp_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(temp_frame, orient=tk.VERTICAL, command=self.temp_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.temp_listbox.config(yscrollcommand=scrollbar.set)
        self.temp_listbox.bind('<<ListboxSelect>>', self.on_temperature_select)

        self.plot_frame = ttk.LabelFrame(self.main_frame, text="数据显示与拟合", padding="10")
        self.plot_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        # 使用Figure对象的tight_layout来改善绘图区域的布局
        self.fig = Figure(figsize=(9, 7), dpi=100, facecolor=self.plot_bg_color, tight_layout=True)
        self.ax = self.fig.add_subplot(111, facecolor=self.plot_bg_color)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.plot_toolbar_backend = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion_drag)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.coord_label = ttk.Label(self.canvas_widget, text="X: -, Y: -", background=self.plot_bg_color)
        self.coord_label.place(relx=1.0, rely=0.0, x=-5, y=5, anchor='ne')

        self.results_frame = ttk.LabelFrame(self.plot_frame, text="拟合结果列表", padding="10")
        self.results_frame.rowconfigure(0, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        
        self.fit_results_tree = ttk.Treeview(
            self.results_frame, columns=('strategy', 'r_squared'), show='tree headings'
        )
        self.fit_results_tree.heading('#0', text='显示')
        self.fit_results_tree.heading('strategy', text='模型')
        self.fit_results_tree.heading('r_squared', text='R²')
        self.fit_results_tree.column('#0', width=40, anchor='center', stretch=False)
        self.fit_results_tree.column('strategy', width=120, anchor='w')
        self.fit_results_tree.column('r_squared', width=80, anchor='center')
        self.fit_results_tree.grid(row=0, column=0, sticky='nsew')
        
        results_scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.fit_results_tree.yview)
        results_scrollbar.grid(row=0, column=1, sticky='ns')
        self.fit_results_tree.configure(yscrollcommand=results_scrollbar.set)
        
        self.fit_results_tree.tag_configure('checked', image=self._get_checked_image())
        self.fit_results_tree.tag_configure('unchecked', image=self._get_unchecked_image())
        self.fit_results_tree.bind('<Button-1>', self._on_treeview_click)
        self.fit_results_tree.bind('<<TreeviewSelect>>', self._on_treeview_selection_change)

        results_action_frame = ttk.Frame(self.results_frame)
        results_action_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(10,0))
        
        self.modify_fit_button = ttk.Button(results_action_frame, text="更改选中拟合", command=self.open_modify_fit_dialog, state=tk.DISABLED)
        self.modify_fit_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.delete_fit_button = ttk.Button(results_action_frame, text="删除选中拟合", command=self._delete_selected_fit, state=tk.DISABLED)
        self.delete_fit_button.pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)

        self.toggle_button = ttk.Button(self.main_frame, text=">", command=self.toggle_results_panel, style='Toggle.TButton', width=2)
        self.toggle_button.grid(row=1, column=2, sticky="ns", padx=(5,0))
        
        self.status_frame = ttk.Frame(self.main_frame, padding=(5,3))
        self.status_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(5,0))
        self.status_frame.columnconfigure(0, weight=1)  # 让状态标签能够自适应宽度
        
        self.status_label = ttk.Label(self.status_frame, text="准备就绪", font=self.status_font)
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.progress = ttk.Progressbar(self.status_frame, mode='indeterminate', length=250)
        self.select_time_zero_from_data_var.trace_add("write", self._on_select_time_zero_from_data_toggle_trace)

        self.controls_to_manage_state.extend([
            self.select_data_button, self.strategy_combobox,
            self.time_zero_entry, self.refresh_button,
            self.select_time_zero_from_data_checkbutton,
            self.new_fit_button, self.batch_fit_button,
            self.modify_fit_button, self.delete_fit_button,
            self.save_results_button, self.export_curves_button,
            self.temp_listbox, self.file_menu, self.tools_menu, self.help_menu,
            self.toggle_button, 
        ])
        
        self._update_strategy_dependent_ui()
        
    def toggle_results_panel(self):
        self.results_frame_visible = not self.results_frame_visible
        if self.results_frame_visible:
            # 使用相对尺寸，适应不同屏幕
            panel_width = min(self.side_panel_width, int(self.plot_frame.winfo_width() * 0.4))
            self.results_frame.place(in_=self.plot_frame, relx=1.0, rely=0, relheight=1, anchor='ne', width=panel_width)
            self.toggle_button.config(text=">")
        else:
            self.results_frame.place_forget()
            self.toggle_button.config(text="<")

    def _get_checked_image(self):
        if not hasattr(self, '_checked_image'):
            img_data = b'R0lGODlhDQANAPIAAAAAAP///8DAwKqqqrCwsPLy8gAAAAAAAAAAACH5BAEAAAYALAAAAAANAA0AAAM7CLrc/jDKSau9OOvNu/9gKI5kaZ5oqq5s675wLM90bd94ru987//AoHBILBqPyKRyyWw6n9CoRCgAOw=='
            self._checked_image = tk.PhotoImage(data=img_data)
        return self._checked_image

    def _get_unchecked_image(self):
        if not hasattr(self, '_unchecked_image'):
            img_data = b'R0lGODlhDQANAPIAAAAAAP///8DAwKqqqrCwsPLy8gAAAAAAAAAAACH5BAEAAAYALAAAAAANAA0AAAM6CLrc/jDKSau9OOvNu/9gKI5kaZ5oqq5s675wLM90bd94ru987//AoHBIJBqPyKRyqUAAAOw=='
            self._unchecked_image = tk.PhotoImage(data=img_data)
        return self._unchecked_image
        
    def _on_treeview_click(self, event):
        region = self.fit_results_tree.identify("region", event.x, event.y)
        if region == "heading" and self.fit_results_tree.identify_column(event.x) == "#0":
            return

        if region == "tree":
            row_id = self.fit_results_tree.identify_row(event.y)
            if row_id:
                # 获取当前温度
                temp = self.temperatures[self.current_temp_index]
                
                # 确保当前温度有对应的可见性集合
                if temp not in self.visible_fit_ids:
                    self.visible_fit_ids[temp] = set()
                    
                # 切换可见性
                if row_id in self.visible_fit_ids[temp]:
                    self.visible_fit_ids[temp].remove(row_id)
                    self.fit_results_tree.item(row_id, tags=('unchecked',))
                else:
                    self.visible_fit_ids[temp].add(row_id)
                    self.fit_results_tree.item(row_id, tags=('checked',))
                self.plot_data(preserve_zoom=True)

    def _on_treeview_selection_change(self, event):
        selected_items = self.fit_results_tree.selection()
        state = tk.NORMAL if selected_items else tk.DISABLED
        self.modify_fit_button.config(state=state)
        self.delete_fit_button.config(state=state)
            
    def load_data_file(self):
        filename = filedialog.askopenfilename(title="选择数据文件", filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*")])
        if filename:
            try:
                self.update_status("正在加载数据..."); self.root.update_idletasks()
                self.temperatures, self.times, self.data_matrix = read_data_file(filename)
                self.data_filename = filename
                self.data_loaded = True
                self.fit_results = {}
                self.visible_fit_ids = {} # 加载新文件时清空可见性设置
                self.select_time_zero_from_data_var.set(False)
                self.file_label.config(text=os.path.basename(filename))
                self.temp_listbox.delete(0, tk.END)
                for temp in self.temperatures: self.temp_listbox.insert(tk.END, f"{temp:.1f} K")
                if self.temperatures.size > 0:
                    self.temp_listbox.selection_set(0)
                    self.on_temperature_select(None)
                self.update_status(f"成功加载: {len(self.temperatures)} 温度, {len(self.times)} 时间点")
                self._update_strategy_dependent_ui()
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败:\n{str(e)}"); self.update_status("加载失败")
                self.data_loaded = False; traceback.print_exc()

    def on_temperature_select(self, event):
        if not self.temp_listbox.winfo_exists() or self.temp_listbox.size() == 0: return
        selection = self.temp_listbox.curselection()
        if selection:
            new_index = selection[0]
            if self.is_dialog_active:
                self.temp_listbox.selection_clear(0, tk.END)
                self.temp_listbox.selection_set(self.current_temp_index)
                return

            self.current_temp_index = new_index
            self.plot_data(preserve_zoom=False)
            self._update_fit_results_treeview()
            self._on_treeview_selection_change(None)

    def _update_fit_results_treeview(self):
        for item in self.fit_results_tree.get_children():
            self.fit_results_tree.delete(item)

        if not self.data_loaded: return
            
        temp = self.temperatures[self.current_temp_index]
        results_for_temp = self.fit_results.get(temp, [])
        
        # 确保当前温度有对应的可见性集合
        if temp not in self.visible_fit_ids:
            self.visible_fit_ids[temp] = set()
        
        # 将所有新添加的拟合默认设为可见
        for result in results_for_temp:
            fit_id = result.get('id')
            if fit_id:
                # 检查这个拟合ID是否在可见性集合中
                # 如果是新ID（之前没见过），则默认设置为可见
                if not any(fit_id in vis_set for vis_set in self.visible_fit_ids.values()):
                    self.visible_fit_ids[temp].add(fit_id)
            
        for result in results_for_temp:
            fit_id = result.get('id')
            if not fit_id: continue

            strategy_name = result.get('strategy_name_used', 'N/A')
            r_squared = result.get('r_squared', 0)
            
            tag = 'checked' if fit_id in self.visible_fit_ids[temp] else 'unchecked'
            
            self.fit_results_tree.insert(
                '', 'end', iid=fit_id,
                values=(strategy_name, f"{r_squared:.4f}"),
                tags=(tag,), text='' 
            )

    def plot_data(self, preserve_zoom=False):
        current_xlim, current_ylim = (None, None)
        if preserve_zoom and self.ax.lines:
            current_xlim, current_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        
        self.ax.clear()
        for spine in self.ax.spines.values(): spine.set_color('gray')
        self.ax.tick_params(axis='x', colors='lightgray')
        self.ax.tick_params(axis='y', colors='lightgray')

        if not self.data_loaded:
            self.ax.text(0.5, 0.5, "请先加载数据", ha='center', va='center', transform=self.ax.transAxes, **self.plot_label_font_props)
        else:
            temp = self.temperatures[self.current_temp_index]
            data_y = self.data_matrix[:, self.current_temp_index]
            gui_t0 = self.time_zero_var.get()

            # --- 智能截断逻辑 ---
            # 1. 确定数据范围和峰值信息
            max_abs_data_val = 0
            t_peak = self.times[0]
            if data_y.size > 0:
                max_abs_data_val = np.max(np.abs(data_y))
                peak_idx = np.argmax(np.abs(data_y))
                t_peak = self.times[peak_idx]

            self.ax.plot(self.times, data_y, 'o', markersize=5, label='原始数据', alpha=0.7, color='#00A0FF')
            
            # 获取当前温度的可见性设置
            visible_ids_for_temp = self.visible_fit_ids.get(temp, set())
            
            results_for_temp = self.fit_results.get(temp, [])
            for i, result in enumerate(results_for_temp):
                fit_id = result.get('id')
                if result.get('success') and fit_id in visible_ids_for_temp:
                    strategy = fitting_dispatcher.get_strategy(result['strategy_name_used'])
                    if strategy:
                        # 2. 生成高分辨率的拟合曲线
                        t_curve = np.linspace(self.times.min(), self.times.max(), 2000)
                        y_curve = strategy.model_function(t_curve, *result['params'])
                        
                        # 3. 计算截断点
                        start_index = 0
                        if max_abs_data_val > 0:
                            peak_curve_idx = np.argmin(np.abs(t_curve - t_peak))
                            y_curve_left = y_curve[:peak_curve_idx]
                            divergent_indices = np.where(np.abs(y_curve_left) > max_abs_data_val * 1.15)[0]
                            if divergent_indices.size > 0:
                                start_index = divergent_indices[-1] + 1
                        
                        label = f"拟合 {i+1} ({strategy.name[:5]} R²={result['r_squared']:.3f})"
                        # 4. 使用截断后的数据进行绘图
                        self.ax.plot(t_curve[start_index:], y_curve[start_index:], '-', label=label, linewidth=2.0)

            if self.is_dialog_active and self.active_fitting_dialog_preview_params and self.active_fitting_dialog_preview_strategy_name:
                strategy = fitting_dispatcher.get_strategy(self.active_fitting_dialog_preview_strategy_name)
                if strategy:
                    t_curve = np.linspace(self.times.min(), self.times.max(), 2000)
                    y_curve = strategy.model_function(t_curve, *self.active_fitting_dialog_preview_params)
                    
                    start_index_preview = 0
                    if max_abs_data_val > 0:
                        peak_curve_idx = np.argmin(np.abs(t_curve - t_peak))
                        y_curve_left = y_curve[:peak_curve_idx]
                        divergent_indices = np.where(np.abs(y_curve_left) > max_abs_data_val * 1.15)[0]
                        if divergent_indices.size > 0:
                            start_index_preview = divergent_indices[-1] + 1

                    self.ax.plot(t_curve[start_index_preview:], y_curve[start_index_preview:], '--', label=f'预览 ({strategy.name[:10]})', linewidth=2.0, color='#33FF57')
            
            self.ax.axvline(x=gui_t0, color='gray', linestyle='--', alpha=0.7, label=f'GUI T0 = {gui_t0:.2f}')
            
            if self.select_time_zero_from_data_var.get():
                manual_t0_val = self.manual_time_zero_var.get()
                line_label = f'拖拽 T0 = {manual_t0_val:.2f}'
                if self.draggable_t0_line and self.draggable_t0_line in self.ax.lines:
                    self.draggable_t0_line.set_xdata([manual_t0_val, manual_t0_val]); self.draggable_t0_line.set_label(line_label)
                else:
                    self.draggable_t0_line = self.ax.axvline(x=manual_t0_val, color='red', linestyle=':', linewidth=1.5, label=line_label, picker=5)
            
            self.ax.set_title(f'温度: {temp:.1f} K', **self.plot_title_font_props)
            self.ax.set_xlabel(f'时间 ({self.time_unit_label})', **self.plot_label_font_props)
            self.ax.set_ylabel(f'信号 ({self.signal_unit_label})', **self.plot_label_font_props)
        
        self.ax.grid(True, alpha=0.2, linestyle=':')
        # 确保图表在缩放时保持良好布局
        self.fig.tight_layout()
        
        if preserve_zoom and current_xlim: 
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
        self._update_legend()
        self.canvas.draw_idle()

    # --- DIALOG LAUNCHERS & HANDLERS ---
    def open_current_fit_window(self):
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据文件", parent=self.root)
            return
        if self.is_dialog_active:
            self.current_fitting_dialog_window.lift()
            return

        self.set_main_controls_state(tk.DISABLED)
        self.is_dialog_active = True
        self.active_fitting_dialog_preview_params = None
        
        current_temp_val = self.temperatures[self.current_temp_index]
        data_for_current_fit = self.data_matrix[:, self.current_temp_index]

        self.current_fitting_dialog_window = CurrentFitWindow(
            parent=self.root,
            current_temp_val=current_temp_val,
            times_original=self.times,
            data_original=data_for_current_fit,
            time_zero_main_context=self.time_zero_var.get(),
            debug_mode=self.debug_var.get(),
            preview_callback=self.handle_dialog_parameter_preview,
            finalized_callback=self.handle_dialog_closed,
            default_strategy_name=self.strategy_name_var.get()
        )
    
    def open_batch_fit_dialog(self):
        if not self.data_loaded:
            messagebox.showwarning("警告", "请先加载数据文件", parent=self.root)
            return
        if self.is_dialog_active:
            self.current_fitting_dialog_window.lift()
            return

        self.set_main_controls_state(tk.DISABLED)
        self.is_dialog_active = True
        self.active_fitting_dialog_preview_params = None

        self.current_fitting_dialog_window = BatchFitDialog(
            parent=self.root,
            all_temperatures=self.temperatures,
            times_original=self.times,
            data_matrix=self.data_matrix,
            time_zero_main_context=self.time_zero_var.get(),
            debug_mode=self.debug_var.get(),
            fit_results=self.fit_results,
            preview_callback=self.handle_dialog_parameter_preview,
            finalized_callback=self.handle_dialog_closed,
            switch_preview_context_callback=self.handle_switch_preview_context,
            default_strategy_name=self.strategy_name_var.get()
        )

    def handle_switch_preview_context(self, temp_index: int):
        if 0 <= temp_index < len(self.temperatures):
            self.current_temp_index = temp_index
            self.temp_listbox.selection_clear(0, tk.END)
            self.temp_listbox.selection_set(temp_index)
            self.active_fitting_dialog_preview_params = None
            self.active_fitting_dialog_preview_strategy_name = None
            self.plot_data(preserve_zoom=True)

    def handle_dialog_parameter_preview(self, params, strategy_name: Optional[str]):
        self.active_fitting_dialog_preview_params = params
        self.active_fitting_dialog_preview_strategy_name = strategy_name
        self.plot_data(preserve_zoom=True)

    def handle_dialog_closed(self, result: Dict[str, Any]):
        action = result.get('action')

        if action in ['cancel', 'cancel_batch']:
            pass
        elif action == 'fit_current':
            fit_res = result.get('fit_result')
            if fit_res and fit_res.get('success'):
                fit_res['id'] = str(uuid.uuid4())
                temp_val = self.temperatures[self.current_temp_index]
                self.fit_results.setdefault(temp_val, []).append(fit_res)
                
                # 确保当前温度有对应的可见性集合
                if temp_val not in self.visible_fit_ids:
                    self.visible_fit_ids[temp_val] = set()
                # 新拟合默认可见
                self.visible_fit_ids[temp_val].add(fit_res['id'])
                
                self.update_status(f"温度 {temp_val:.1f} K 新增拟合: R²={fit_res['r_squared']:.4f}")
                self._update_listbox_item(self.current_temp_index)
                self._update_fit_results_treeview()

        elif action == 'fit_batch_single':
            fit_res = result.get('fit_result')
            temp_val = result.get('temperature')
            if fit_res and fit_res.get('success'):
                fit_res['id'] = str(uuid.uuid4())
                self.fit_results.setdefault(temp_val, []).append(fit_res)
                
                # 确保该温度有对应的可见性集合
                if temp_val not in self.visible_fit_ids:
                    self.visible_fit_ids[temp_val] = set()
                # 新拟合默认可见
                self.visible_fit_ids[temp_val].add(fit_res['id'])
                
                temp_idx = np.where(self.temperatures == temp_val)[0][0]
                self._update_listbox_item(temp_idx)
                if temp_idx == self.current_temp_index:
                    self._update_fit_results_treeview()
        
        elif action == 'fit_batch_multiple':
            new_fits_by_temp = result.get('new_results', {})
            for temp_val, fit_res in new_fits_by_temp.items():
                if fit_res and fit_res.get('success'):
                    fit_res['id'] = str(uuid.uuid4())
                    self.fit_results[temp_val] = [fit_res]
                    
                    # 确保该温度有对应的可见性集合
                    if temp_val not in self.visible_fit_ids:
                        self.visible_fit_ids[temp_val] = set()
                    # 新拟合默认可见
                    self.visible_fit_ids[temp_val].add(fit_res['id'])
                    
                    temp_idx = np.where(self.temperatures == temp_val)[0][0]
                    self._update_listbox_item(temp_idx)
            self.update_status(f"批量拟合完成, 更新了 {len(new_fits_by_temp)} 个温度的结果。")
            if self.current_temp_index is not None:
                self._update_fit_results_treeview()

        self.set_main_controls_state(tk.NORMAL)
        self.is_dialog_active = False
        self.active_fitting_dialog_preview_params = None
        self.active_fitting_dialog_preview_strategy_name = None
        if self.current_fitting_dialog_window:
             self.current_fitting_dialog_window = None

    def open_modify_fit_dialog(self):
        selected_items = self.fit_results_tree.selection()
        if not selected_items:
            messagebox.showwarning("无选择", "请先在右侧列表中选择一个要修改的拟合结果。", parent=self.root)
            return
            
        selected_id = selected_items[0]
        temp = self.temperatures[self.current_temp_index]
        target_fit = next((fit for fit in self.fit_results.get(temp, []) if fit['id'] == selected_id), None)
        
        if not target_fit:
            messagebox.showerror("错误", "找不到选中的拟合数据，请刷新。", parent=self.root)
            return

        self.set_main_controls_state(tk.DISABLED)
        self.is_dialog_active = True
        
        self.current_fitting_dialog_window = ModifyFitDialog(
            parent=self.root,
            main_app_instance=self,
            existing_fit_result=target_fit,
            times_original=self.times,
            data_original=self.data_matrix[:, self.current_temp_index],
            preview_callback=self.handle_dialog_parameter_preview,
            finalized_callback=self.handle_modify_dialog_closed
        )
        
    def handle_modify_dialog_closed(self, result: Dict[str, Any]):
        self.set_main_controls_state(tk.NORMAL)
        self.is_dialog_active = False
        self.active_fitting_dialog_preview_params = None
        self.active_fitting_dialog_preview_strategy_name = None
        self.current_fitting_dialog_window = None

        if result.get('success') and result.get('updated_fit_result'):
            updated_res = result['updated_fit_result']
            fit_id_to_update = updated_res.get('id')
            temp_val = self.temperatures[self.current_temp_index]

            results_list = self.fit_results.get(temp_val, [])
            for i, res in enumerate(results_list):
                if res.get('id') == fit_id_to_update:
                    self.fit_results[temp_val][i] = updated_res
                    break
            
            self.update_status(f"拟合 {fit_id_to_update[:6]}... 已更新: R²={updated_res['r_squared']:.4f}")
            self._update_fit_results_treeview()
        
        self.plot_data(preserve_zoom=True)
            
    def _delete_selected_fit(self):
        selected_items = self.fit_results_tree.selection()
        if not selected_items:
            messagebox.showwarning("无选择", "请先在右侧列表中选择一个要删除的拟合结果。", parent=self.root)
            return
            
        selected_id = selected_items[0]
        if not messagebox.askyesno("确认删除", f"确定要删除选中的拟合结果 (ID: {selected_id[:6]}...) 吗？\n此操作无法撤销。", parent=self.root):
            return

        temp = self.temperatures[self.current_temp_index]
        results_list = self.fit_results.get(temp, [])
        
        self.fit_results[temp] = [fit for fit in results_list if fit.get('id') != selected_id]
        if selected_id in self.visible_fit_ids.get(temp, set()):
            self.visible_fit_ids[temp].remove(selected_id)
        
        self._update_listbox_item(self.current_temp_index)
        self._update_fit_results_treeview()
        self.plot_data(preserve_zoom=True)
        self.update_status(f"拟合 {selected_id[:6]}... 已删除。")

    def _update_listbox_item(self, index):
            temp = self.temperatures[index]
            successful_fits = [r for r in self.fit_results.get(temp, []) if r.get('success')]
            fit_count = len(successful_fits)
            
            item_text = f"{temp:.1f} K"
            if fit_count > 0:
                item_text += f" ({fit_count}个拟合)"

            self.temp_listbox.delete(index); self.temp_listbox.insert(index, item_text)
            self.temp_listbox.selection_set(index)

    def _prepare_results_for_export(self) -> Optional[Dict[float, Any]]:
        self.update_status("正在检查拟合结果...")
        temps_with_multiple_fits = {}
        final_fits_to_export = {}

        for temp, fit_list in self.fit_results.items():
            successful_fits = [r for r in fit_list if r.get('success')]
            
            if len(successful_fits) == 1:
                final_fits_to_export[temp] = successful_fits[0]
            elif len(successful_fits) > 1:
                temps_with_multiple_fits[temp] = successful_fits

        if temps_with_multiple_fits:
            self.update_status("发现多个拟合结果, 请选择...")
            dialog = ExportSelectionDialog(self.root, temps_with_multiple_fits)
            selections = dialog.show()

            if selections is None:
                self.update_status("导出已取消。")
                return None

            for temp, selected_fit_id in selections.items():
                selected_fit = next((fit for fit in temps_with_multiple_fits[temp] if fit.get('id') == selected_fit_id), None)
                if selected_fit:
                    final_fits_to_export[temp] = selected_fit
        
        self.update_status("结果准备就绪。")
        return final_fits_to_export

    def save_results(self):
        if not any(self.fit_results.values()):
            messagebox.showwarning("警告", "没有成功的拟合结果可保存。", parent=self.root)
            return
            
        fits_to_export = self._prepare_results_for_export()
        
        if fits_to_export is None:
            return

        if not fits_to_export:
            messagebox.showwarning("警告", "没有成功的拟合结果可保存。", parent=self.root)
            return
            
        filename = filedialog.asksaveasfilename(
            title="保存参数概览文件",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.update_status("正在导出参数...")
                export_parameters_summary(filename, self.temperatures, fits_to_export, self.data_filename)
                self.update_status(f"参数已保存至 {os.path.basename(filename)}")
                messagebox.showinfo("成功", f"参数概览已成功保存到:\n{filename}", parent=self.root)
            except Exception as e:
                messagebox.showerror("错误", f"保存参数失败: {e}", parent=self.root)
                traceback.print_exc()
            finally:
                self.update_status("准备就绪")

    def export_curves(self):
        if not any(self.fit_results.values()):
            messagebox.showwarning("警告", "没有成功的拟合结果可导出。", parent=self.root)
            return

        fits_to_export = self._prepare_results_for_export()
        
        if fits_to_export is None:
            return

        if not fits_to_export:
            messagebox.showwarning("警告", "没有成功的拟合结果可导出。", parent=self.root)
            return

        filename = filedialog.asksaveasfilename(
            title="保存数据和拟合曲线文件",
            defaultextension=".txt",
            filetypes=[("Tab-separated files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.update_status("正在导出曲线...")
                export_curves_data(filename, self.temperatures, self.times, self.data_matrix, fits_to_export, fitting_dispatcher)
                self.update_status(f"曲线数据已导出至 {os.path.basename(filename)}")
                messagebox.showinfo("成功", f"数据和拟合曲线已成功保存到:\n{filename}", parent=self.root)
            except Exception as e:
                messagebox.showerror("错误", f"导出曲线失败: {e}", parent=self.root)
                traceback.print_exc()
            finally:
                self.update_status("准备就绪")
    
    def export_multi_sheet_csv(self):
        """导出为多Sheet格式的CSV文件，可直接拖入Excel或Origin"""
        if not any(self.fit_results.values()):
            messagebox.showwarning("警告", "没有成功的拟合结果可导出。", parent=self.root)
            return

        fits_to_export = self._prepare_results_for_export()
        
        if fits_to_export is None:
            return

        if not fits_to_export:
            messagebox.showwarning("警告", "没有成功的拟合结果可导出。", parent=self.root)
            return

        filename = filedialog.asksaveasfilename(
            title="保存多Sheet格式CSV文件",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if filename:
            try:
                self.update_status("正在导出多Sheet格式CSV...")
                from io_functions import export_multi_sheet_csv  # 引入新的导出函数
                
                export_multi_sheet_csv(
                    filename, 
                    self.temperatures, 
                    self.times, 
                    self.data_matrix, 
                    fits_to_export, 
                    fitting_dispatcher,
                    self.data_filename
                )
                
                self.update_status(f"数据已导出至 {os.path.basename(filename)}")
                messagebox.showinfo(
                    "成功", 
                    f"多Sheet格式CSV已成功保存到:\n{filename}\n\n可直接拖入Excel或Origin查看。", 
                    parent=self.root
                )
            except Exception as e:
                messagebox.showerror("错误", f"导出多Sheet格式CSV失败: {e}", parent=self.root)
                traceback.print_exc()
            finally:
                self.update_status("准备就绪")

    def _gather_application_state(self) -> Dict[str, Any]:
        state = {
            'version': '1.0',
            'data_filename': self.data_filename,
            'temperatures': self.temperatures,
            'times': self.times,
            'data_matrix': self.data_matrix,
            'fit_results': self.fit_results,
            'visible_fit_ids': self.visible_fit_ids,
            'current_temp_index': self.current_temp_index,
            'time_zero_var': self.time_zero_var.get(),
            'manual_time_zero_var': self.manual_time_zero_var.get(),
            'select_time_zero_from_data_var': self.select_time_zero_from_data_var.get(),
            'strategy_name_var': self.strategy_name_var.get(),
            'debug_var': self.debug_var.get(),
        }
        return state

    def _apply_application_state(self, state: Dict[str, Any]):
        self.data_filename = state['data_filename']
        self.temperatures = state['temperatures']
        self.times = state['times']
        self.data_matrix = state['data_matrix']
        self.fit_results = state['fit_results']
        
        # 确保visible_fit_ids是字典格式
        if isinstance(state.get('visible_fit_ids'), dict):
            self.visible_fit_ids = state['visible_fit_ids']
        else:
            # 兼容旧版本，将旧格式转换为新格式
            old_visible_ids = state.get('visible_fit_ids', set())
            self.visible_fit_ids = {}
            # 遍历所有温度的拟合结果，将旧的可见性信息应用到相应温度
            for temp, fits in self.fit_results.items():
                self.visible_fit_ids[temp] = set(fit['id'] for fit in fits if fit['id'] in old_visible_ids)
        
        self.current_temp_index = state['current_temp_index']
        
        self.time_zero_var.set(state['time_zero_var'])
        self.manual_time_zero_var.set(state['manual_time_zero_var'])
        self.select_time_zero_from_data_var.set(state['select_time_zero_from_data_var'])
        self.strategy_name_var.set(state['strategy_name_var'])
        self.debug_var.set(state.get('debug_var', False))

        self.data_loaded = True
        
        self.file_label.config(text=os.path.basename(self.data_filename) if self.data_filename else "未选择文件")
        
        self.temp_listbox.delete(0, tk.END)
        for i in range(len(self.temperatures)):
            self._update_listbox_item(i)
        
        if 0 <= self.current_temp_index < len(self.temperatures):
            self.temp_listbox.selection_set(self.current_temp_index)
            self.on_temperature_select(None)
        
        self.update_status(f"成功加载项目: {os.path.basename(self.data_filename)}")
        self._update_strategy_dependent_ui()

    def save_project(self):
        if not self.data_loaded:
            messagebox.showwarning("无数据", "没有已加载的数据可以保存。", parent=self.root)
            return

        filename = filedialog.asksaveasfilename(
            title="保存拟合项目",
            defaultextension=".fitproj",
            filetypes=[("Fitting Project", "*.fitproj"), ("所有文件", "*.*")]
        )
        if not filename:
            return

        try:
            self.update_status("正在保存项目...")
            state_to_save = self._gather_application_state()
            with open(filename, 'wb') as f:
                pickle.dump(state_to_save, f)
            self.update_status(f"项目已成功保存至 {os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存项目时发生错误:\n{e}", parent=self.root)
            traceback.print_exc()
            self.update_status("项目保存失败")

    def load_project(self):
        if self.is_dialog_active:
             messagebox.showwarning("操作受限", "请先关闭当前打开的拟合对话框。", parent=self.root)
             return
             
        filename = filedialog.askopenfilename(
            title="打开拟合项目",
            filetypes=[("Fitting Project", "*.fitproj"), ("所有文件", "*.*")]
        )
        if not filename:
            return

        try:
            self.update_status("正在加载项目...")
            with open(filename, 'rb') as f:
                loaded_state = pickle.load(f)
            
            if 'version' not in loaded_state or 'fit_results' not in loaded_state:
                raise ValueError("文件格式无效或已损坏。")

            self._apply_application_state(loaded_state)

        except (pickle.UnpicklingError, ValueError, KeyError) as e:
            messagebox.showerror("加载失败", f"无法加载项目文件。文件可能已损坏或不是有效的项目文件。\n\n错误: {e}", parent=self.root)
            traceback.print_exc()
            self.update_status("项目加载失败")
        except Exception as e:
            messagebox.showerror("加载失败", f"加载项目时发生未知错误:\n{e}", parent=self.root)
            traceback.print_exc()
            self.update_status("项目加载失败")

    def _on_strategy_change(self, *args):
        selected_strategy_name = self.strategy_name_var.get()
        if selected_strategy_name:
            self.current_strategy = fitting_dispatcher.get_strategy(selected_strategy_name)
            if self.current_strategy:
                self.update_status(f"默认模型已更改: {self.current_strategy.name}")
            else:
                self.update_status(f"错误: 无法加载模型 '{selected_strategy_name}'")
        self._update_strategy_dependent_ui()

    def set_main_controls_state(self, state_to_set):
        for item in self.controls_to_manage_state:
            try:
                if state_to_set == tk.NORMAL and item in [self.modify_fit_button, self.delete_fit_button]:
                    self._on_treeview_selection_change(None)
                    continue
                
                if isinstance(item, tk.Menu):
                    if item.index(tk.END) is not None:
                        for i in range(item.index(tk.END) + 1):
                            try:
                                if item.type(i) in ["command", "checkbutton", "radiobutton", "cascade"]:
                                    item.entryconfig(i, state=state_to_set)
                            except tk.TclError: pass
                elif hasattr(item, 'configure'):
                    item.configure(state=state_to_set)
                    
            except Exception as e:
                 print(f"Info: Could not set state for {item}: {e}")

    def on_motion(self, event):
        if self.dragging_t0_line: return
        if event.inaxes == self.ax and event.xdata is not None:
            self.coord_label.config(text=f"X: {event.xdata:.2f}, Y: {event.ydata:.2f}")
        else:
            self.coord_label.config(text="X: -, Y: -")
            
    def _on_mouse_press(self, event):
        if event.inaxes != self.ax or not self.select_time_zero_from_data_var.get() or not self.draggable_t0_line:
            self.dragging_t0_line = False; return
        if self.draggable_t0_line.contains(event)[0]: self.dragging_t0_line = True

    def _on_mouse_motion_drag(self, event):
        if not self.dragging_t0_line or event.inaxes != self.ax or event.xdata is None: return
        new_x = round(event.xdata, 4)
        self.manual_time_zero_var.set(new_x)
        if self.draggable_t0_line:
            self.draggable_t0_line.set_xdata([new_x, new_x])
            self.draggable_t0_line.set_label(f'拖拽 T0 = {new_x:.2f}')
            self._update_legend(); self.canvas.draw_idle()
            
    def _on_mouse_release(self, event): self.dragging_t0_line = False
    def _on_refresh_button_click(self): self.plot_data(preserve_zoom=True)
    def _on_select_time_zero_from_data_toggle_trace(self, *args):
        is_manual_select_active = self.select_time_zero_from_data_var.get()
        self.manual_time_zero_entry.config(state=tk.NORMAL if is_manual_select_active else tk.DISABLED)
        if not is_manual_select_active and self.draggable_t0_line:
            if self.draggable_t0_line in self.ax.lines: self.draggable_t0_line.remove()
            self.draggable_t0_line = None
        self.plot_data(preserve_zoom=True)
        
    def activate_zoom_mode(self): self.plot_toolbar_backend.zoom()
    def activate_pan_mode(self): self.plot_toolbar_backend.pan()
    def reset_plot_view(self): self.plot_toolbar_backend.home()
    def go_back_plot_view(self): self.plot_toolbar_backend.back()
    
    def _update_legend(self): 
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, loc='best', prop=self.plot_legend_font_props, frameon=False,
                           labelcolor='lightgray')
        elif self.ax.get_legend():
            self.ax.get_legend().remove()

    def _on_debug_mode_toggle(self): print(f"Debug Mode: {self.debug_var.get()}")
    def update_status(self, msg): 
        if hasattr(self, 'status_label'): self.status_label.config(text=msg)
        self.root.update_idletasks()
        
    def show_help(self): messagebox.showinfo("帮助", "使用说明...")
    def show_about(self): messagebox.showinfo("关于", "拟合程序...")

    def run_curve_tool(self):
        """运行曲线变换观察工具"""
        try:
            self.update_status("启动曲线变换观察工具...")
            curve_main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curve", "curve_main.py")
            
            if not os.path.exists(curve_main_path):
                messagebox.showerror("错误", f"找不到曲线工具文件: {curve_main_path}", parent=self.root)
                return
                
            # 使用当前Python解释器运行curve_main.py
            python_executable = sys.executable
            subprocess.Popen([python_executable, curve_main_path])
            
            self.update_status("曲线变换观察工具已启动")
        except Exception as e:
            messagebox.showerror("错误", f"启动曲线工具失败: {str(e)}", parent=self.root)
            traceback.print_exc()
            self.update_status("启动失败")

    def _update_strategy_dependent_ui(self):
        fit_button_state = tk.NORMAL if self.current_strategy and self.data_loaded else tk.DISABLED
        if hasattr(self, 'new_fit_button'):
            self.new_fit_button.config(state=fit_button_state)
            self.batch_fit_button.config(state=fit_button_state)

def main():
    root = tk.Tk()
    try:
        fitting_dispatcher.discover_strategies(force_rediscover=True)
    except Exception as e:
        messagebox.showerror("策略加载错误", f"启动时加载拟合策略失败: {e}")
        traceback.print_exc()
        root.destroy()
        return

    app = ExponentialFittingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
