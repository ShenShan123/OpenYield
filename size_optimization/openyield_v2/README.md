# OpenYieldV2 优化算法

该目录是独立的离线优化子包，不修改或调用现有等效电路、逐器件波动和 Xyce 仿真代码。
优化器读取 `datasets/` 中的 6T、10T 训练数据，所有算法共用同一套目标、约束、预算和输出格式。

## 算法

- Evolutionary：NSGA2、SPEA2、UNSGA3、CTAEA
- Bayesian：GPBO、PAREGO、MACE
- Proposed：coarse search + 可微 TabPFN refinement

配置文件为 `configs/experiment.yaml`。主要配置项包括：

- `optimization_problem`：目标和约束
- `optimization_budget`：各算法的代理模型查询预算
- `shared_optimizer`：训练集划分和搜索空间边界
- `evolutionary`、`bayesian`、`proposed`：算法参数
- `comparison`：默认运行的算法、设备和输出目录

## 安装额外依赖

先安装项目环境，再安装本目录的优化依赖：

```bash
python -m pip install -r size_optimization/openyield_v2/requirements.txt
```

## 运行

先检查解析后的命令，不启动 TabPFN：

```bash
python -m size_optimization.openyield_v2.run_experiment --dry-run
```

按 YAML 运行默认对比：

```bash
python -m size_optimization.openyield_v2.run_experiment
```

选择算法运行：

```bash
python -m size_optimization.openyield_v2.run compare -- \
  --algorithms NSGA2,CTAEA,PAREGO,MACE,PROPOSED \
  --max-evals 1000 \
  --device auto
```

运行单个算法族：

```bash
python -m size_optimization.openyield_v2.run optimize \
  --family evolutionary -- --algorithms NSGA2,CTAEA

python -m size_optimization.openyield_v2.run optimize \
  --family bayesian -- --algorithms PAREGO,MACE

python -m size_optimization.openyield_v2.run optimize \
  --family proposed
```

如果需要指定物理 GPU，在 `optimize` 后增加 `--gpu-id N`；对比模式可在 `--` 后传入同名参数。

## 输出

一次对比只保留统一结果：

```text
runs/optimization/comparison/<timestamp>/
├── evaluations/<algorithm>.csv
├── pareto_fronts/<algorithm>.csv
├── algorithm_summary.csv
└── run_config.json
```

`evaluations` 保存实际代理查询记录，`pareto_fronts` 从这些记录中筛选满足约束的非支配点。
输出目录已在本子包的 `.gitignore` 中排除。

## 数据边界

附带训练集来自 TT / 25 °C、等效电路开启、逐器件波动关闭的已有采样数据。
因此当前结果反映这两份静态数据，不等同于最新逐器件波动仿真结果。后续若要更新训练集，应通过当前电路接口重新采样，而不是使用旧版 `data_collection` 代码。
