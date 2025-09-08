import yaml
from typing import List, Dict, Union, Any, Optional
#from parameters import ConfigLoader


class AttrDict:
    """支持点号访问的字典类"""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # 如果是字典，递归转换为AttrDict
                setattr(self, key, AttrDict(value))
            elif isinstance(value, list):
                # 处理列表中的字典
                setattr(self, key, [AttrDict(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)


class GlobalConfig:
    """表示全局配置，支持点号访问嵌套属性"""

    def __init__(self, config_data: Dict[str, Any]):
        # 基础配置
        self.vdd = config_data.get("vdd", 1.0)
        self.temperature = config_data.get("temperature", 27)
        self.num_rows = config_data.get("num_rows", 32)
        self.num_cols = config_data.get("num_cols", 1)
        self.monte_carlo_runs = config_data.get("monte_carlo_runs", 1)
        self.timeout = config_data.get("timeout", 120)
        self.pdk_path = config_data.get("pdk_path", "model_lib/models.spice")

        # 将嵌套字典转换为支持点号访问的对象
        self.evaluator = AttrDict(config_data.get("evaluator", {}))
        self.simulator = AttrDict(config_data.get("simulator", {}))
        self.objectives = AttrDict(config_data.get("objectives", {}))

        # 特殊处理性能指标
        self.metrics = self._process_performance_metrics(config_data.get("performance_metrics", {}))

        # 规范化性能指标的上下界值
        self._normalize_metric_bounds()

    def _process_performance_metrics(self, metrics_dict: dict) -> AttrDict:
        """处理性能指标，转换为支持点号访问的结构"""
        metrics = {}
        for metric_name, metric_data in metrics_dict.items():
            # 创建支持点号访问的指标对象
            metric_obj = AttrDict(metric_data)

            # 特殊处理 names 属性
            if hasattr(metric_obj, 'names') and not isinstance(metric_obj.names, list):
                metric_obj.names = [metric_obj.names]

            metrics[metric_name] = metric_obj

        # 返回支持点号访问的指标字典
        return AttrDict(metrics)

    def _normalize_metric_bounds(self):
        """规范化性能指标的上下界值，支持标量和列表"""
        if not hasattr(self, 'metrics'):
            return

        # 遍历所有指标
        for metric_name in dir(self.metrics):
            if metric_name.startswith('__'):
                continue

            metric = getattr(self.metrics, metric_name)

            # 处理上界
            if hasattr(metric, 'upper'):
                upper = metric.upper
                if isinstance(upper, (int, float)):
                    metric.upper = [upper]
                elif not isinstance(upper, list):
                    metric.upper = [upper]

            # 处理下界
            if hasattr(metric, 'lower'):
                lower = metric.lower
                if isinstance(lower, (int, float)):
                    metric.lower = [lower]
                elif not isinstance(lower, list):
                    metric.lower = [lower]

    def __repr__(self):
        """返回配置的字符串表示"""
        return (
            f"GlobalConfig:\n"
            f"  VDD: {self.vdd} V\n"
            f"  温度: {self.temperature} °C\n"
            f"  阵列: {self.num_rows}×{self.num_cols}\n"
            f"  蒙特卡洛次数: {self.monte_carlo_runs}\n"
            f"  超时: {self.timeout} 秒\n"
            f"  PDK路径: {self.pdk_path}\n"
            f"  性能指标: {list(vars(self.metrics).keys() if hasattr(self, 'metrics') else [])}\n"
            f"  目标函数: {getattr(self.objectives, 'formula', '未定义')}"
        )

    def get_metric(self, metric_name: str) -> Any:
        """获取指定的性能指标配置"""
        if hasattr(self, 'metrics') and hasattr(self.metrics, metric_name):
            return getattr(self.metrics, metric_name)
        return None

    def get_metric_names(self) -> List[str]:
        """获取所有性能指标名称"""
        if hasattr(self, 'metrics'):
            return [name for name in dir(self.metrics) if not name.startswith('__')]
        return []

    def get_objective_formula(self) -> str:
        """获取目标函数公式"""
        return getattr(self.objectives, 'formula', '')

    def get_constraints(self) -> List[str]:
        """获取约束条件"""
        constraints = getattr(self.objectives, 'constraints', [])
        # 如果约束是字符串，则转换为列表
        if isinstance(constraints, str):
            return [constraints]
        return constraints
# 全局配置加载函数
def load_global_config(file_path: str) -> GlobalConfig:
    """加载全局配置文件"""
    try:
        # 尝试不同编码打开文件
        for encoding in ['utf-8', 'utf-8-sig', 'gbk']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    config_data = yaml.safe_load(f)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"无法解码文件 {file_path}，尝试的编码: utf-8, utf-8-sig, gbk")

        # 创建全局配置对象
        return GlobalConfig(config_data)

    except Exception as e:
        print(f"加载全局配置文件 {file_path} 错误: {e}")
        raise


# 通用参数类
class Parameter:
    """表示一个设计参数"""

    def __init__(
            self,
            name: str,
            param_type: str,
            instance_names: Union[str, List[str]],
            description: str,
            value: Any,
            upper: Optional[Union[float, List[float]]] = None,
            lower: Optional[Union[float, List[float]]] = None,
            choices: Optional[List[str]] = None
    ):
        self.name = name
        self.param_type = param_type
        self.instance_names = instance_names
        self.description = description
        self.value = value
        self.upper = upper
        self.lower = lower
        self.choices = choices
        self._validate_types()

    def _validate_types(self):
        """验证参数类型一致性"""
        # 检查值类型与参数类型是否匹配
        if "scalar" in self.param_type:
            if isinstance(self.value, list):
                raise ValueError(f"标量参数 {self.name} 不能有列表值")
        elif "list" in self.param_type:
            if not isinstance(self.value, list):
                raise ValueError(f"列表参数 {self.name} 需要列表值")

        # 检查边界类型是否匹配
        if self.upper is not None:
            if "continuous" in self.param_type:
                if "scalar" in self.param_type:
                    if not isinstance(self.upper, (float, int)) or not isinstance(self.lower, (float, int)):
                        raise TypeError(f"连续标量参数 {self.name} 的边界必须是数值")
                else:
                    if not isinstance(self.upper, list) or not isinstance(self.lower, list):
                        raise TypeError(f"连续列表参数 {self.name} 的边界必须是列表")
            else:
                raise ValueError(f"离散参数 {self.name} 不应有数值边界")

        # 检查离散型参数的选项
        if self.choices is not None:
            if "categorical" not in self.param_type and "discrete" not in self.param_type:
                raise ValueError(f"非离散参数 {self.name} 不应有choices选项")

    def __repr__(self):
        return (f"<Parameter {self.name} ({self.param_type}): "
                f"instances={self.instance_names}, value={self.value}>")


# 通用电路配置类
class CircuitConfig:
    """表示任意电路的完整配置"""

    def __init__(self, config_name: str, parameters: Dict[str, Parameter]):
        self.config_name = config_name
        self.parameters = parameters

    def get_parameter(self, name: str) -> Parameter:
        """按名称获取参数"""
        return self.parameters[name]

    def __getattr__(self, name: str) -> Parameter:
        """允许通过属性访问参数"""
        if name in self.parameters:
            return self.parameters[name]
        raise AttributeError(f"在{self.config_name}配置中无此参数: {name}")

    def __repr__(self):
        params = "\n  ".join([f"{k}: {v}" for k, v in self.parameters.items()])
        return f"{self.config_name}配置:\n  {params}"


class ConfigLoader:
    """加载和管理多个电路配置"""

    def __init__(self):
        self.configs = {}

    def load_config(self, file_path: str, config_name: str) -> CircuitConfig:
        """加载单个配置文件"""
        try:
            # 尝试不同编码打开文件
            for encoding in ['utf-8', 'utf-8-sig', 'gbk']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        config_data = yaml.safe_load(f)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"无法解码文件 {file_path}，尝试的编码: utf-8, utf-8-sig, gbk")

            # 提取配置部分
            circuit_config = config_data.get(config_name, {})
            if not circuit_config:
                raise ValueError(f"YAML文件中未找到 {config_name} 配置")

            # 解析参数
            parameters = {}
            for param_name, param_data in circuit_config.get("parameters", {}).items():
                param = Parameter(
                    name=param_name,
                    param_type=param_data["type"],
                    instance_names=param_data["names"],
                    description=param_data["description"],
                    value=param_data["value"],
                    upper=param_data.get("upper"),
                    lower=param_data.get("lower"),
                    choices=param_data.get("choices")
                )
                parameters[param_name] = param

            # 创建并存储配置对象
            config = CircuitConfig(config_name, parameters)
            self.configs[config_name] = config
            return config

        except Exception as e:
            print(f"加载配置文件 {file_path} 错误: {e}")
            raise

    def get_config(self, config_name: str) -> CircuitConfig:
        """获取已加载的配置"""
        if config_name in self.configs:
            return self.configs[config_name]
        raise KeyError(f"未找到配置: {config_name}")

    def load_all_configs(self, config_list: Dict[str, str]):
        """批量加载多个配置"""
        for config_name, file_path in config_list.items():
            try:
                self.load_config(file_path, config_name)
                print(f"✓ 成功加载 {config_name} 配置")
            except Exception as e:
                print(f"⚠ 加载 {config_name} 配置失败: {e}")




#总的 SRAM_CONFIG 类
class SRAM_CONFIG:
    """集成所有SRAM相关配置的顶级类"""

    def __init__(self):
        self.global_config = None
        self.sram_6t_cell = None
        self.wordline_driver = None
        self.precharge = None
        self.column_mux = None
        self.senseamp = None
        self.write_driver = None
        self.decoder = None

    def load_all_configs(self, global_file: str, circuit_configs: Dict[str, str]):
        """加载所有配置"""

        # 加载全局配置
        self.global_config = load_global_config(global_file)
        print(f"✓ 全局配置加载完成: {global_file}")
        # 创建子电路配置加载器
        loader = ConfigLoader()
        # 批量加载所有子电路配置
        print("\n开始加载电路配置:")
        loader.load_all_configs(circuit_configs)

        # 将加载的配置分配给各个属性
        self.sram_6t_cell = loader.get_config("SRAM_6T_CELL")
        self.wordline_driver = loader.get_config("WORDLINEDRIVER")
        self.precharge = loader.get_config("PRECHARGE")
        self.column_mux = loader.get_config("COLUMNMUX")
        self.senseamp = loader.get_config("SENSEAMP")
        self.write_driver = loader.get_config("WRITEDRIVER")
        self.decoder = loader.get_config("DECODER")
        

        print("\n所有配置加载完成!")

    # def get_all_configs(self) -> Dict[str, Any]:
    #     """返回所有配置的字典"""
    #     return {
    #         "global": self.global_config,
    #         "sram_6t_cell": self.sram_6t_cell,
    #         "wordline_driver": self.wordline_driver,
    #         "precharge": self.precharge,
    #         "column_mux": self.column_mux,
    #         "sense_amp": self.sense_amp,
    #         "write_driver": self.write_driver
    #     }

    def __repr__(self):
        """返回配置的简洁摘要"""
        configs = [
            f"全局配置: {self.global_config is not None}",
            f"SRAM单元: {self.sram_6t_cell is not None}",
            f"字线驱动器: {self.wordline_driver is not None}",
            f"预充电: {self.precharge is not None}",
            f"列选择器: {self.column_mux is not None}",
            f"灵敏放大器: {self.senseamp is not None}",
            f"写入驱动器: {self.write_driver is not None}",
            f"译码器: {self.decoder is not None}"
        ]
        return "SRAM配置状态:\n  " + "\n  ".join(configs)





# 在文件末尾添加使用示例
if __name__ == "__main__":
    # 创建SRAM配置管理器
    sram_config = SRAM_CONFIG()

    # 定义配置文件路径
    global_file = "yaml/global.yaml"
    circuit_files = {
        "SRAM_6T_CELL": "yaml/sram_6t_cell.yaml",
        "WORDLINEDRIVER": "yaml/wordline_driver.yaml",
        "PRECHARGE": "yaml/precharge.yaml",
        "COLUMNMUX": "yaml/mux.yaml",
        "SENSEAMP": "yaml/sa.yaml",
        "WRITEDRIVER": "yaml/write_driver.yaml",
        "DECODER":"yaml/decoder.yaml"
    }

    # 加载所有配置
    sram_config.load_all_configs(global_file, circuit_files)

    # 打印配置状态
    print("\n配置状态摘要:")
    print(sram_config)

    # 访问特定配置
    print("\n访问SRAM单元参数:")
    sram_cell = sram_config.sram_6t_cell
    if sram_cell:
        print(f"PMOS宽度: {sram_cell.pmos_width.value} m")
        print(f"NMOS宽度: PD={sram_cell.nmos_width.value[0]}m, PG={sram_cell.nmos_width.value[1]}m")
        print(f"沟道长度: {sram_cell.length.value} m")

    # 访问灵敏放大器配置
    print("\n访问灵敏放大器参数:")
    sense_amp = sram_config.senseamp
    if sense_amp:
        first_param = list(sense_amp.parameters.values())[0]
        print(f"第一个参数: {first_param.name} = {first_param.value}")

    print(sram_config.global_config.evaluator.class_name)
    print(sram_config.global_config.metrics.snm.type)
    print(sram_config.senseamp.length.param_type)
    print(sram_config.precharge.pmos_model.value)
    print(sram_config.decoder.pmos_width.value)
