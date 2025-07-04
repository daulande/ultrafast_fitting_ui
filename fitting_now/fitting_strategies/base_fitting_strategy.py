# fitting_strategies/base_fitting_strategy.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Tuple, Dict # 导入 Dict

class FittingStrategy(ABC):
    """
    拟合策略的抽象基类。
    所有具体的拟合策略都应继承此类并实现其抽象方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        策略的名称，用于在UI中显示或内部识别。
        例如："Single Exponential", "Erf Convolved Biexponential"
        """
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """
        该模型（策略）所需的参数数量。
        """
        pass

    @abstractmethod
    def model_function(self, t_original: np.ndarray, *params):
        """
        实际的拟合模型函数。
        
        参数:
            t_original (np.ndarray): 原始时间轴数据 (通常未做任何平移)。
                                     策略实现需要明确其模型是基于此原始时间轴，
                                     还是期望一个相对于某个参考点（可能由 `t_context_zero_gui` 提供）调整过的时间。
                                     推荐所有模型都基于 `t_original` 设计，并将任何时间零点作为模型参数。
            *params: 模型的一系列参数。
        
        返回:
            np.ndarray: 根据模型和参数计算得到的y轴值。
            
        注意:
            - 如果模型本身包含时间零点参数 (例如Erf卷积模型中的t0_model)，
              该t0_model应该是params中的一个，并且代表绝对时间。
            - 此函数应始终基于传入的 `t_original` 进行计算，除非策略明确设计为处理调整后的时间。
              与 `estimate_initial_parameters` 协作，确保参数和时间轴的一致性。
        """
        pass

    @abstractmethod
    def estimate_initial_parameters(self,
                                    time_original_slice: np.ndarray,
                                    data_slice_processed: np.ndarray,
                                    data_slice_smooth_for_est: np.ndarray,
                                    t_context_zero_gui: Optional[float],
                                    fixed_params: Optional[Dict[int, float]] = None) -> List[List[float]]:
        """
        估算模型的初始参数。

        参数:
            time_original_slice (np.ndarray): 用于拟合的时间数据切片 (原始时间轴，通常未平移)。
            data_slice_processed (np.ndarray): 对应的y轴数据 (可能经过了如基线校正等的初步处理，但与time_original_slice对应)。
            data_slice_smooth_for_est (np.ndarray): 用于估算的平滑后y轴数据 (与time_original_slice对应)。
            t_context_zero_gui (Optional[float]): 从主GUI传入的T0值，仅作为上下文参考。
            fixed_params (Optional[Dict[int, float]]): 一个字典，包含固定参数的索引和值。
                                                       例如: {4: 0.5, 6: 10.2} 表示第4个和第6个参数被固定。
                                                       策略应使用这些值作为已知条件来估算其他参数。
        返回:
            List[List[float]]: 一个包含多个初始参数猜测列表的列表。
                               每个子列表代表一组完整的模型参数。
                               例如: [[A_guess1, tau_guess1, ...], [A_guess2, tau_guess2, ...]]
        
        重要: 
            此方法返回的参数应适用于 `model_function`。如果 `model_function` 是基于 `t_original` 定义的，
            那么这里估算的参数也应该是针对 `t_original` 的。如果 `model_function` 期望一个调整后的时间轴，
            那么这里的参数应该是针对那个调整后时间轴的（这种情况下，核心拟合器在调用 `curve_fit` 时需要传递调整后的时间轴）。
            推荐：所有 `model_function` 基于 `t_original`，所有估算参数也最终适配 `t_original`。
        """
        pass

    @abstractmethod
    def get_parameter_names(self) -> List[str]: # 使用 List
        """
        返回模型参数的名称列表，顺序应与 `model_function` 中的 `*params` 和
        `estimate_initial_parameters` 返回的子列表中的参数顺序一致。
        例如：["A", "tau", "baseline"]
        """
        pass

    @abstractmethod
    def get_bounds(self,
                   t_data_slice: np.ndarray,
                   y_data_slice: np.ndarray,
                   for_global_opt: bool = False) -> List[Tuple[float, float]]: # 使用 List 和 Tuple
        """
        获取参数的边界。

        参数:
            t_data_slice (np.ndarray): 用于拟合的时间数据切片 (通常是原始时间轴上的数据)。
            y_data_slice (np.ndarray): 对应的y轴数据切片。
            for_global_opt (bool): 如果为True，则可能返回更宽的边界，适用于全局优化算法。
                                   默认为False，适用于局部优化（如curve_fit）。
        返回:
            List[Tuple[float, float]]: 一个包含每个参数 (下界, 上界) 元组的列表。
                                       顺序应与 `get_parameter_names()` 返回的名称顺序一致。
        """
        pass

    @abstractmethod
    def check_parameter_validity(self,
                                 params: List[float], # 使用 List
                                 t_data_for_context: Optional[np.ndarray] = None) -> Tuple[bool, str]: # 使用 Optional 和 Tuple
        """
        检查给定的参数对于此模型是否有效。

        参数:
            params (List[float]): 要检查的一组模型参数。
            t_data_for_context (Optional[np.ndarray]): 可选的，用于上下文检查的时间数据（例如，检查t0是否在数据范围内）。
        
        返回:
            Tuple[bool, str]: 一个元组，第一个元素是布尔值 (True表示有效，False表示无效)，
                              第二个元素是字符串，如果无效，则为原因描述。
        """
        pass
