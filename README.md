# 瞬态光谱动力学数据拟合分析软件

## 简介

本项目是一个使用 Python 开发的桌面应用程序，软件通过图形用户界面（GUI）操作，支持多种复杂的指数衰减模型，能够对实验数据进行精确的拟合，并提取关键的动力学参数。解决处理超快数据的时候无穷无尽的拟合问题
## 主要功能

- **图形用户界面 (GUI)**: 提供一个用户友好的界面，用于数据加载、模型选择、参数设置、拟合过程可视化和结果导出。
- **多种拟合模型**: 内置了多种常用的动力学模型，包括：
  - 单指数、双指数、三指数衰减模型 (`Mono-`, `Bi-`, `Tri-Exponential`)
  - 考虑仪器响应函数（IRF）的卷积拟合 (`Erf-Convoluted`)
  - 包含啁啾振荡的衰减模型 (`Chirped Oscillation`)
- **批量处理**: 支持对多个数据集进行自动化批量拟合，极大地提高了数据处理效率。
- **参数调整**: 用户可以在拟合前设定参数的初始值、边界和约束条件，也可以在拟合后对结果进行手动微调。
- **结果可视化与导出**:
  - 实时绘制原始数据、拟合曲线和残差图。
  - 支持将拟合结果（参数、标准差等）导出为常见的文本或表格格式。

## 项目结构

```
.
├── fitting/                      # 主程序目录
│   ├── gui_main.py               # GUI 主程序入口
│   ├── gui_fitting_panel.py      # 核心拟合控制面板界面
│   ├── gui_batch_fit.py          # 批量拟合界面
│   ├── gui_current_fit.py        # 当前拟合结果显示界面
│   ├── gui_modify_fit_dialog.py  # 修改拟合参数对话框
│   ├── io_functions.py           # 数据导入/导出功能
│   ├── fitting_core.py           # 拟合策略模块
|   ├── fitting_dispatcher.py     # 拟合策略调度器，根据用户选择调用不同模型
│   └── requirements.txt          # 项目依赖包
│   ├── fitting_strategies/           # 拟合函数目录
│   │   ├── base_fitting_strategy.py  # 拟合函数基类（请勿删除）
│   │   ├── biexp_strategy.py         # 双指数模型
│   │   ├── triexp_strategy.py        # 三指数模型
│   │   ├── erf_conv_biexp_strategy.py # 带卷积的双指数模型
└── └── └── ...                       # 其他拟合模型文件

```

## 安装与环境配置

为了保证程序正常运行，建议使用虚拟环境。

1.  **下载代码**: 下载本项目所有文件并解压到本地文件夹。

2.  **创建虚拟环境** (推荐):
    ```bash
    python -m venv venv
    ```

3.  **激活虚拟环境**:
    -   Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    -   macOS / Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **安装依赖**:
    项目的所有依赖项都记录在 `fitting/requirements.txt` 文件中。运行以下命令进行安装：
    ```bash
    pip install -r fitting/requirements.txt
    ```

## 使用方法

1.  确保已经激活虚拟环境并安装了所有依赖。

2.  运行 `gui_main.py` 启动主程序:
    ```bash
    python fitting/gui_main.py
    ```

3.  程序启动后，您可以按照界面提示进行操作：
    -   通过菜单栏或按钮加载您的实验数据文件。
    -   点击您要拟合的数据。
    -   在点击'单次拟合'，并在其中调整您的参数。
    -   设置初始参数（可选），然后点击“开始拟合”按钮。
    -   在结果区域查看拟合曲线、残差和拟合参数。
    -   当拟合结果如您所愿时，点击批量拟合。
    -   选择合适的策略
    -   使用导出功能保存您的分析结果。
  
    -   
4.  添加您自己的拟合策略
    -   当fitting strategy里的拟合函数无法满足您的使用需求时（例如您的超快光谱有两个相干声子振荡，或者对于超快THz实验需要一些新的函数的时候），请参照fitting/fitting_strategies/base_fitting_strategy.py来添加自己的拟合函数。拟合策略中的猜测初始值为可选项，但建议您针对自己的数据采用合适的猜测初始值策略，以提高处理数据的效率
5.  
## 贡献代码

欢迎对本项目做出贡献！如果您有任何改进建议或发现了 Bug，请通过以下方式参与：

1.  Fork 本项目仓库。
2.  创建一个新的分支 (`git checkout -b feature/YourAmazingFeature`)。
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)。
4.  将分支推送到远程仓库 (`git push origin feature/YourAmazingFeature`)。
5.  提交一个 Pull Request。

## 许可证

本项目采用 [GPL-3.0 许可证](https://www.gnu.org/licenses/gpl-3.0.html)。
