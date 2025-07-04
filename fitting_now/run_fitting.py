
import sys
import os

# 确保导入路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 检查必要的依赖
    import numpy
    import scipy
    import matplotlib
    import pandas
    import tkinter
except ImportError as e:
    print(f"缺少必要的依赖包: {e}")
    print("\n请使用以下命令安装依赖:")
    print("pip install numpy scipy matplotlib pandas")
    sys.exit(1)

# 导入并运行主程序
from gui_main import main

if __name__ == "__main__":
    print("正在启动多指数拟合程序...")
    main()