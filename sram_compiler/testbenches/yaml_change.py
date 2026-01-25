from ruamel.yaml import YAML
from typing import List, Any


def update_sram6t_yaml_inplace(
    x: List[Any],
    yaml_path: str
):
    """
    使用 ruamel.yaml 修改 SRAM 6T YAML 文件
    """

    if len(x) != 7:
        raise ValueError("x 必须包含 7 个参数")

    yaml = YAML()
    yaml.preserve_quotes = True    # 保留引号
    yaml.width = 4096              # 防止自动换行
    yaml.indent(mapping=2, sequence=4, offset=2)

    # 1. 读取（保留格式）
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    params = data["SRAM_6T_CELL"]["parameters"]

    # 2. 数值原位修改（不改变结构）
    params["nmos_width"]["value"][0] = x[0]   # pd_width
    params["nmos_width"]["value"][1] = x[1]   # pg_width
    params["pmos_width"]["value"]    = x[2]   # pu_width
    params["length"]["value"]        = x[3]   # length

    params["nmos_model"]["value"][0] = x[4]   # pd_model
    params["nmos_model"]["value"][1] = x[5]   # pg_model
    params["pmos_model"]["value"]    = x[6]   # pu_model

    # 3. 覆盖写回（格式保持）
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

def update_global_yaml_inplace(
    x: List[Any],
    yaml_path: str
):
    """
    使用 ruamel.yaml 修改 global YAML 文件
    x: [num_rows, num_cols, choose_columnmux]
    """

    if len(x) != 3:
        raise ValueError("x 必须包含 3 个参数 [num_rows, num_cols, choose_columnmux]")

    yaml = YAML()
    yaml.preserve_quotes = True    # 保留引号
    yaml.width = 4096              # 防止自动换行
    yaml.indent(mapping=2, sequence=4, offset=2)

    # 1. 读取（保留格式）
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    # 2. 数值原位修改（不改变结构）
    data["num_rows"] = x[0]
    data["num_cols"] = x[1]
    data["choose_columnmux"] = x[2]

    # 3. 覆盖写回（格式保持）
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

import pandas as pd
import numpy as np

def summarize_from_csv(csv_path,operation):
    """
    从CSV文件中读取数据,计算Power和Delay
    
    参数:
    csv_path: CSV文件路径
    
    返回:
    y: numpy数组,y[0]为Delay,y[1]为Power
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 计算Power (PSTC + PDYN)
    power = df['PSTC'] + df['PDYN'] if operation == 'read' or operation == 'write' else df['PAVG']
    
    if operation == 'read':
        # 计算Delay (TDECODER + TPRCH + TSA + TSWING + TS_EN + TWLDRV)
        delay = (df['TDECODER'] + df['TPRCH'] + df['TSA'] + 
                df['TSWING'] + df['TS_EN'] + df['TWLDRV'])
    elif operation == 'write':
        # 计算Delay (TDECODER + TPRCH + TSA + TSWING + TS_EN + TWLDRV)
        delay = (df['TDECODER'] + df['TWDRV'] + df['TWLDRV'] + df['TWRITE_Q'] )
    elif operation == 'read&write':
        delay = (df['TVOUT_PERIOD'])
    

    
    # 创建结果向量，y[0]为Delay，y[1]为Power
    y = np.column_stack((delay.values, power.values))
    
    return y
