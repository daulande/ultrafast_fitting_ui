# gui_export_selection_dialog.py
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any

class ExportSelectionDialog(tk.Toplevel):
    """
    一个对话框，用于在导出时解决一个温度有多个拟合结果的冲突。
    """
    def __init__(self, parent, temps_with_multiple_fits: Dict[float, List[Dict[str, Any]]]):
        super().__init__(parent)
        self.transient(parent)
        self.title("选择要导出的拟合")
        self.geometry("600x500")
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        self.parent = parent
        self.temps_data = temps_with_multiple_fits
        self.selection_vars: Dict[float, tk.StringVar] = {}
        self.final_selection: Dict[float, str] = {} # {temperature: fit_id}
        
        self._create_widgets()
        self.grab_set()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=10)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        ttk.Label(scrollable_frame, text="以下温度存在多个拟合结果，请为每个温度选择一个用于导出：", wraplength=550).pack(fill=tk.X, pady=(0, 10))

        for temp, fits in self.temps_data.items():
            self.selection_vars[temp] = tk.StringVar(value=fits[0].get('id')) # Default to the first one
            
            lf = ttk.LabelFrame(scrollable_frame, text=f"温度: {temp:.1f} K", padding=10)
            lf.pack(fill=tk.X, expand=True, pady=5)
            
            for fit in fits:
                fit_id = fit.get('id', 'N/A')
                strategy = fit.get('strategy_name_used', 'Unknown')
                r2 = fit.get('r_squared', 0)
                
                radio_text = f"模型: {strategy}, R²: {r2:.4f} (ID: ...{fit_id[-6:]})"
                rb = ttk.Radiobutton(lf, text=radio_text, variable=self.selection_vars[temp], value=fit_id)
                rb.pack(anchor='w', padx=5)

        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(button_frame, text="确定", command=self._on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=self._on_cancel).pack(side=tk.RIGHT)

    def _on_ok(self):
        for temp, var in self.selection_vars.items():
            self.final_selection[temp] = var.get()
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        self.final_selection = None
        self.grab_release()
        self.destroy()

    def show(self):
        self.wait_window()
        return self.final_selection