# io_functions.py
"""
文件输入输出模块
处理数据文件的读取和结果的保存 (已重构以支持策略)
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime
import fitting_dispatcher # 用于获取策略信息
import traceback # For detailed error printing
from typing import Dict, Any

def read_data_file(filename):
    """
    读取数据文件。
    (此函数基本保持不变，因为它处理的是原始数据格式)
    """
    try:
        # 尝试常用的分隔符
        for sep_char in ['\t', ',', ' ', ';']:
            try:
                with open(filename, 'r', encoding='utf-8-sig') as f: # utf-8-sig handles BOM
                    lines = f.readlines()
                
                if not lines: raise ValueError("文件为空")

                # 解析第一行作为温度行
                first_line_values = lines[0].strip().split(sep_char)
                
                time_col_present_in_header = False
                potential_temps_start_idx = 0
                if first_line_values:
                    try:
                        float(first_line_values[0]) 
                    except ValueError:
                        time_col_present_in_header = True
                        potential_temps_start_idx = 1
                
                if len(first_line_values) <= potential_temps_start_idx:
                    continue 

                raw_temps = first_line_values[potential_temps_start_idx:]
                if not raw_temps: continue

                temperatures = []
                all_temps_numeric = True
                for t_str in raw_temps:
                    try:
                        temperatures.append(float(t_str))
                    except ValueError:
                        all_temps_numeric = False
                        break
                if not all_temps_numeric: continue

                times_list = []
                data_matrix_list = []
                data_lines_to_parse = lines[1:]

                for line_num, line_content in enumerate(data_lines_to_parse):
                    values = line_content.strip().split(sep_char)
                    if not values or not values[0]: continue 

                    expected_cols_this_line = len(temperatures) + 1
                    if len(values) == expected_cols_this_line:
                        try:
                            times_list.append(float(values[0]))
                            data_row = [float(v) for v in values[1:]]
                            if len(data_row) != len(temperatures):
                                if times_list: times_list.pop()
                                continue
                            data_matrix_list.append(data_row)
                        except ValueError:
                            if times_list: times_list.pop()
                            continue
                
                if times_list and data_matrix_list:
                    temperatures_arr = np.array(temperatures)
                    times_arr = np.array(times_list)
                    data_matrix_arr = np.array(data_matrix_list)
                    if data_matrix_arr.shape[0] == len(times_arr) and data_matrix_arr.shape[1] == len(temperatures_arr):
                        return temperatures_arr, times_arr, data_matrix_arr
            except Exception:
                pass 
        
        # 如果所有分隔符都失败，尝试使用pandas的自动检测
        try:
            df = pd.read_csv(filename, header=0, sep=None, engine='python', encoding='utf-8-sig', skip_blank_lines=True, on_bad_lines='warn')
            if df.empty or df.shape[1] < 2:
                 raise ValueError("Pandas解析后数据为空或列数不足。")

            potential_temps_str = df.columns.tolist()
            time_col_name = potential_temps_str[0]
            temp_values_str = potential_temps_str[1:]
            pd_temperatures = [float(t) for t in temp_values_str]
            
            pd_times = df[time_col_name].values.astype(float)
            pd_data_matrix = df.iloc[:, 1:].values.astype(float)

            if pd_data_matrix.shape[1] != len(pd_temperatures):
                 raise ValueError(f"Pandas解析：数据列数 ({pd_data_matrix.shape[1]}) 与温度数 ({len(pd_temperatures)}) 不匹配。")
            if pd_data_matrix.shape[0] != len(pd_times):
                 raise ValueError(f"Pandas解析：数据行数 ({pd_data_matrix.shape[0]}) 与时间点数 ({len(pd_times)}) 不匹配。")

            return np.array(pd_temperatures), pd_times, pd_data_matrix
        except Exception as e_pandas_final:
            raise RuntimeError(f"读取文件失败：无法确定文件格式或分隔符。请确保文件是标准的文本或CSV格式。\n错误: {e_pandas_final}")

    except Exception as e_outer:
        traceback.print_exc()
        raise e_outer


def export_parameters_summary(filename: str, all_temperatures: np.ndarray, 
                              fit_results_map: Dict[float, Dict[str, Any]], 
                              data_filename_original: str):
    """
    导出所有拟合的参数概览。
    纵轴是温度，横轴是所有策略中出现过的参数。
    fit_results_map: 字典，键是温度，值是“单个”最佳或选定的拟合结果。
    """
    try:
        output_lines = []
        output_lines.append(f"# Fitting Parameters Summary")
        output_lines.append(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output_lines.append(f"# Original Data file: {os.path.basename(data_filename_original) if data_filename_original else 'N/A'}")
        output_lines.append("#")

        # 动态构建一个包含所有可能参数的表头
        all_param_names = set()
        for res in fit_results_map.values():
            if res and res.get('success'):
                strategy = fitting_dispatcher.get_strategy(res['strategy_name_used'])
                if strategy:
                    all_param_names.update(strategy.get_parameter_names())
        
        sorted_param_names = sorted(list(all_param_names))
        
        header_core = ["Temperature", "Strategy_Name", "R_squared"]
        param_headers = []
        for name in sorted_param_names:
            param_headers.extend([f"{name}_Value", f"{name}_Error"])
        
        output_lines.append("\t".join(header_core + param_headers))

        for temp in all_temperatures:
            result = fit_results_map.get(temp)
            
            row_data_parts = [f"{temp:.2f}"]
            
            if result and result.get('success'):
                strategy_name = result.get('strategy_name_used', 'N/A')
                row_data_parts.append(strategy_name)
                row_data_parts.append(f"{result.get('r_squared', np.nan):.6f}")

                params = result.get('params', [])
                errors = result.get('errors', [np.nan] * len(params))
                
                # 获取当前策略的参数名列表
                current_strategy_obj = fitting_dispatcher.get_strategy(strategy_name)
                current_param_names = current_strategy_obj.get_parameter_names() if current_strategy_obj else []
                
                # 创建一个从参数名到值的映射
                param_value_map = {name: (params[i], errors[i]) for i, name in enumerate(current_param_names)}

                # 按全局排序的参数名填充数据
                for name in sorted_param_names:
                    val, err = param_value_map.get(name, (np.nan, np.nan))
                    row_data_parts.append(f"{val:.6e}" if np.isfinite(val) else "N/A")
                    row_data_parts.append(f"{err:.6e}" if np.isfinite(err) else "N/A")

            else:
                # 如果该温度没有拟合数据，用空值填充
                num_to_pad = len(header_core) - 1 + len(sorted_param_names) * 2
                row_data_parts.extend(["N/A"] * num_to_pad)
                
            output_lines.append("\t".join(row_data_parts))

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        return True

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"保存参数概览失败: {str(e)}")


def export_curves_data(filename: str, temperatures: np.ndarray, times_original: np.ndarray, 
                       data_matrix: np.ndarray, fit_results_map: Dict[float, Dict[str, Any]], dispatcher):
    """
    导出原始数据和对应的拟合曲线。
    格式为：原始时间, 原始数据, 拟合时间, 拟合数据, ...
    fit_results_map: 字典，键是温度，值是“单个”最佳或选定的拟合结果。
    """
    try:
        all_series = []
        
        for i, temp in enumerate(temperatures):
            # 添加原始数据列
            all_series.append(pd.Series(times_original, name=f'T{temp:.1f}_Time_Raw'))
            all_series.append(pd.Series(data_matrix[:, i], name=f'T{temp:.1f}_Data_Raw'))

            result = fit_results_map.get(temp)
            if result and result.get('success'):
                strategy_name = result.get('strategy_name_used')
                params = result.get('params')
                
                if strategy_name and params is not None:
                    strategy_obj = dispatcher.get_strategy(strategy_name)
                    if strategy_obj:
                        try:
                            # 为拟合曲线使用相同的时间轴以确保一一对应
                            y_fit_curve = strategy_obj.model_function(times_original, *params)
                            
                            # 添加拟合数据列
                            all_series.append(pd.Series(times_original, name=f'T{temp:.1f}_Time_Fit'))
                            all_series.append(pd.Series(y_fit_curve, name=f'T{temp:.1f}_Data_Fit'))
                        except Exception as e_curve:
                            print(f"Warning: Error generating fit curve for T={temp}, Strategy='{strategy_name}'. Error: {e_curve}")
                            # 添加空列作为占位符
                            all_series.append(pd.Series(name=f'T{temp:.1f}_Time_Fit'))
                            all_series.append(pd.Series(name=f'T{temp:.1f}_Data_Fit'))
                    else:
                        print(f"Warning: Strategy '{strategy_name}' not found for T={temp} curve export.")
            else:
                # 如果没有拟合，添加空列以保持结构
                all_series.append(pd.Series(name=f'T{temp:.1f}_Time_Fit'))
                all_series.append(pd.Series(name=f'T{temp:.1f}_Data_Fit'))

        df = pd.concat(all_series, axis=1)
        df.to_csv(filename, index=False, float_format='%.6e', sep='\t')
        return True
        
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"导出曲线失败: {str(e)}")