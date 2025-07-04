# gui_current_fit.py
"""
Defines the dialog for fitting the currently selected temperature's data.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from typing import Optional, Callable, Dict, Any

from gui_fitting_panel import FittingPanel

class CurrentFitWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk,
                 current_temp_val: float, times_original: np.ndarray, data_original: np.ndarray,
                 time_zero_main_context: float, debug_mode: bool,
                 preview_callback: Optional[Callable] = None,
                 finalized_callback: Optional[Callable] = None,
                 default_strategy_name: Optional[str] = None):
        super().__init__(parent)
        self.transient(parent)
        self.parent = parent
        
        # --- Callbacks ---
        self.preview_callback = preview_callback
        self.finalized_callback = finalized_callback

        self.title(f"为温度 {current_temp_val:.1f} K 新建拟合")
        self.geometry("650x700")
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        self.config(bg='#2d2d2d')
        
        # --- Create main fitting panel ---
        self.fitting_panel = FittingPanel(
            parent=self,
            apply_callback=self._on_apply_and_close,
            current_temp_val=current_temp_val,
            times_original=times_original,
            data_original=data_original,
            time_zero_main_context=time_zero_main_context,
            debug_mode=debug_mode,
            preview_callback=self.preview_callback,
            default_strategy_name=default_strategy_name
        )
        self.fitting_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Bottom buttons ---
        bottom_frame = ttk.Frame(self, padding=(0, 10))
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Add a separator for visual clarity
        ttk.Separator(bottom_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # The "Apply" button is inside the panel. We just need a cancel/close button here.
        ttk.Button(bottom_frame, text="取消并关闭", command=self._on_cancel).pack(side=tk.RIGHT, padx=10)


    def _on_apply_and_close(self, fit_result: Dict):
        """Callback for the panel's 'Apply' button."""
        result_dict = {'action': 'fit_current', 'fit_result': fit_result}
        if self.finalized_callback:
            self.finalized_callback(result_dict)
        self.destroy()

    def _on_cancel(self):
        """Called on window close or 'Cancel' button press."""
        if messagebox.askyesno("确认", "未保存的拟合将会丢失。确定要关闭吗？", parent=self):
            if self.finalized_callback:
                self.finalized_callback({'action': 'cancel', 'success': False})
            self.destroy()

