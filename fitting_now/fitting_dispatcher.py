# fitting_dispatcher.py
import os
import importlib
import inspect
import sys
from pathlib import Path
from typing import Optional, List # 导入 Optional 和 List

# 确保 fitting_strategies 目录可以被找到
try:
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from fitting_strategies.base_fitting_strategy import FittingStrategy
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import FittingStrategy. Ensure 'fitting_strategies/base_fitting_strategy.py' exists. Details: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempted project root: {project_root}")
    sys.exit(1)


_STRATEGIES = {} # 缓存已发现的策略实例

def discover_strategies(strategy_dir_name="fitting_strategies", force_rediscover=False):
    """
    从指定目录动态加载所有 FittingStrategy 实现。
    """
    global _STRATEGIES
    if _STRATEGIES and not force_rediscover:
        return

    _STRATEGIES = {}
    
    project_root = Path(__file__).resolve().parent
    strategy_dir_path = project_root / strategy_dir_name

    if not strategy_dir_path.is_dir():
        print(f"Warning: Strategy directory not found: {strategy_dir_path}")
        return

    for f_path in strategy_dir_path.glob("*.py"):
        if f_path.name.startswith("__"):
            continue
        
        module_name_for_import = f"{strategy_dir_name}.{f_path.stem}"
        
        try:
            module = importlib.import_module(module_name_for_import)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, FittingStrategy) and obj is not FittingStrategy:
                    try:
                        strategy_instance = obj()
                        if strategy_instance.name in _STRATEGIES:
                            print(f"Warning: Duplicate strategy name '{strategy_instance.name}' found. Overwriting.")
                        _STRATEGIES[strategy_instance.name] = strategy_instance
                        print(f"Discovered fitting strategy: '{strategy_instance.name}'")
                    except Exception as e_inst:
                        print(f"Error instantiating strategy '{name}' from '{f_path.name}': {e_inst}")
        except Exception as e_general:
            print(f"Error processing strategy file '{f_path.name}': {e_general}")


def get_available_strategy_names() -> List[str]: # 使用 List
    """返回所有可用拟合策略的名称列表。"""
    if not _STRATEGIES:
        discover_strategies()
    return sorted(list(_STRATEGIES.keys()))

def get_strategy(name: str) -> Optional[FittingStrategy]: # 改为 Optional[FittingStrategy]
    """根据名称获取拟合策略实例。"""
    if not _STRATEGIES:
        discover_strategies()
    return _STRATEGIES.get(name)

# 在模块首次加载时自动发现策略
discover_strategies()
