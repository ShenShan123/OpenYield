"""Data preparation and TabPFN helpers used by the V2 optimizers."""

from __future__ import annotations

import math
import os
import random
import inspect
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabpfn import TabPFNRegressor
from tabpfn.inference import InferenceEngineBatchedNoPreprocessing


def seed_set(seed: int) -> None:
    """
    Fix the random seed for reproducibility
    固定随机种子以确保结果可重现
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add SRAM architecture features used by the surrogate."""
    df = df.copy()
    if "rows" in df.columns and "cols" in df.columns:
        df["aspect_ratio"] = df["rows"].astype(float) / df["cols"].astype(float)
        df["log_rows"] = np.log2(df["rows"].astype(float))
        df["log_cols"] = np.log2(df["cols"].astype(float))
    return df

# raw数据变成系统级指标
def process_raw_to_system_metrics(
    raw_df_metrics: pd.DataFrame,
    rows_array,
    cols_array,
    total_KB: int = 32,
    output_cols: int = 16
) -> pd.DataFrame:
    """
    将 TabPFN 预测的 raw 指标转换为系统级 processed 指标。

    输入 raw_df_metrics 必须包含：
    hold_snm, read_snm, write_snm,
    raw_read_delay, raw_write_delay,
    read_pstc, read_pdyn, write_pstc, write_pdyn,
    single_array_area

    输出：
    hold_snm, read_snm, write_snm,
    read_delay, write_delay,
    read_power, write_power, area

    面积逻辑：
    area = predicted_single_array_area * num_arrays
    """
    df = raw_df_metrics.copy()

    rows_array = np.asarray(rows_array).astype(int)
    cols_array = np.asarray(cols_array).astype(int)

    total_bits = int(total_KB * 1024 * 8)
    capacity_per_array = rows_array * cols_array

    if np.any(capacity_per_array <= 0):
        raise ValueError("rows_array * cols_array must be positive.")

    num_arrays = np.ceil(total_bits / capacity_per_array).astype(int)
    num_arrays = np.maximum(num_arrays, 1)

    log_num_arrays = np.log2(num_arrays)

    cs_delay_adder = 4.167213500e-11 * log_num_arrays
    mux_delay = 1.8e-10
    mux_power = 0.1e-6

    raw_read_delay = df["raw_read_delay"].to_numpy(dtype=float)
    raw_write_delay = df["raw_write_delay"].to_numpy(dtype=float)
    read_pstc = df["read_pstc"].to_numpy(dtype=float)
    read_pdyn = df["read_pdyn"].to_numpy(dtype=float)
    write_pstc = df["write_pstc"].to_numpy(dtype=float)
    write_pdyn = df["write_pdyn"].to_numpy(dtype=float)
    single_array_area = df["single_array_area"].to_numpy(dtype=float)

    read_delay = raw_read_delay + cs_delay_adder + log_num_arrays * mux_delay
    write_delay = raw_write_delay + cs_delay_adder

    factor = np.ceil(output_cols / cols_array).astype(int)
    factor = np.maximum(factor, 1)
    multi_cycle_mask = cols_array < output_cols

    read_delay = np.where(multi_cycle_mask, read_delay * factor, read_delay)
    write_delay = np.where(multi_cycle_mask, write_delay * factor, write_delay)

    read_power = read_pstc * num_arrays + read_pdyn
    write_power = write_pstc * num_arrays + write_pdyn

    # 多周期时，动态功耗乘以 factor，不重复多加一次。
    read_power = np.where(multi_cycle_mask, read_pstc * num_arrays + read_pdyn * factor, read_power)
    write_power = np.where(multi_cycle_mask, write_pstc * num_arrays + write_pdyn * factor, write_power)

    # MUX 修正项保留，但不再用 abs 掩盖负值；若出现非正值，clip 到极小正数。
    read_power = read_power - (num_arrays - 1) * log_num_arrays * mux_power
    read_power = np.clip(read_power, 1e-15, None)
    write_power = np.clip(write_power, 1e-15, None)

    area = single_array_area * num_arrays
    area = np.clip(area, 1e-18, None)

    return pd.DataFrame(
        {
            "hold_snm": df["hold_snm"].to_numpy(dtype=float),
            "read_snm": df["read_snm"].to_numpy(dtype=float),
            "write_snm": df["write_snm"].to_numpy(dtype=float),
            "read_delay": read_delay,
            "write_delay": write_delay,
            "read_power": read_power,
            "write_power": write_power,
            "area": area,
        }
    )


def compute_padding_dim(
    n_features: int,
    multiple: int = 3,
    min_dim: Optional[int] = None,
) -> int:
    """Return the smallest aligned feature dimension."""
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    base = max(
        int(n_features),
        int(min_dim) if min_dim is not None else int(n_features),
    )
    return int(math.ceil(base / multiple) * multiple)


class DataLoader:
    """Load, encode and transform the shared surrogate dataset."""

    def __init__(self, config: Dict):
        self.cont_features = config.get("cont_features", [])
        self.arch_features = config.get("arch_features", [])
        self.cat_features = config.get("cat_features", [])
        self.targets = config.get("targets", [])
        self.filter_positive_targets = config.get("filter_positive_targets", True)
        self.filepath = config.get("filepath")

        self.num_features = self.cont_features + self.arch_features
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.num_features),
                (
                    "cat",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    self.cat_features,
                ),
            ],
            remainder="drop",
        )
        self.feature_engineering_funcs = []

    def add_feature_engineering(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> None:
        self.feature_engineering_funcs.append(func)

    def load_and_preprocess(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"[Error] File not found: {self.filepath}")

        df = pd.read_csv(self.filepath)
        for func in self.feature_engineering_funcs:
            df = func(df)

        required_cols = self.num_features + self.cat_features + self.targets
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"[Error] 缺失配置中要求的列: {missing_cols}")

        if self.filter_positive_targets:
            df = df[(df[self.targets] > 0).all(axis=1)]

        # 提取 
        df = df.dropna(subset=required_cols).copy()
        y_df = df[self.targets].copy()

        # 5. 提取并转换特征 X
        # preprocessor.fit_transform 会自动完成分类变量的 One-Hot 编码，并将结果转为 numpy 数组
        X_array = self.preprocessor.fit_transform(df)
        
        # 获取转换后的特征名称（方便调试和后续的特征重要性分析）
        cat_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.cat_features) if self.cat_features else []
        self.feature_names_out = self.num_features + list(cat_feature_names)
        
        # 将 X 转回 DataFrame 以保持可读性，或者直接输出 float32 的 numpy 数组供模型使用
        X_df = pd.DataFrame(X_array, columns=self.feature_names_out, index=df.index).astype(np.float32)

        return X_df.values, y_df, df

    def transform_features(self, df_in: pd.DataFrame) -> np.ndarray:
        """
        将外部候选设计 DataFrame 转成与训练阶段完全一致的 TabPFN 输入矩阵。
        注意：必须在 load_and_preprocess() 之后调用，因为此时 preprocessor 已经 fit。
        """
        df = df_in.copy()
        df.columns = df.columns.str.strip()

        for func in self.feature_engineering_funcs:
            df = func(df)

        required_cols = self.num_features + self.cat_features
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"[transform_features] 候选点缺失特征列: {missing_cols}")

        X_array = self.preprocessor.transform(df[required_cols])
        return X_array.astype(np.float32)

class TabPFNSurrogate:
    """
    TabPFN 代理模型封装器
    """
    def __init__(self, 
                 device: str = 'cpu', 
                 n_estimators: int = 1, 
                 shared_regressor = None,     
                 scaler = None,               
                 seed: int = 42):             
        
        self.device = device
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.seed = seed
        self.original_dim = None 
        self.padding_dim = None
        self._new_engine_api = (
            "feature_schema"
            in inspect.signature(InferenceEngineBatchedNoPreprocessing).parameters
        )
        
        if shared_regressor:
            self._regressor = shared_regressor
        else:
            self._regressor = TabPFNRegressor(device=device, n_estimators=n_estimators, ignore_pretraining_limits=True)
        self.is_fitted = False

    def _pad_features(self, X_input, is_tensor=False):
        """内部工具：将特征维度填充到指定的 self.padding_dim"""
        pad_dim = self.padding_dim - self.original_dim
        if pad_dim <= 0:
            return X_input

        n = X_input.shape[0]
        noise_scale = 1e-2

        if is_tensor:
            gen = torch.Generator(device=X_input.device)
            gen.manual_seed(self.seed)

            pad = torch.randn(
                n,
                pad_dim,
                device=X_input.device,
                dtype=X_input.dtype,
                generator=gen,
            ) * noise_scale

            # padding 噪声只是常数占位，不需要对它求梯度
            pad = pad.detach()
            return torch.cat([X_input, pad], dim=1)

        else:
            rng = np.random.RandomState(self.seed)
            pad = rng.randn(n, pad_dim).astype(np.float32) * noise_scale
            return np.hstack([X_input.astype(np.float32), pad])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练模型：实际上 TabPFN 是元学习模型，这里主要是构建上下文 (Context) 数据集。
        """
        y = y.ravel()
        self.original_dim = X.shape[1]
        self.padding_dim = compute_padding_dim(self.original_dim)
        
        # 1. 数据标准化与维度填充
        X_scaled_real = self.scaler.fit_transform(X)
        X_padded = self._pad_features(X_scaled_real, is_tensor=False)
        
        # 2. 调用库函数构建索引
        self._regressor.fit(X_padded, y)
        self.is_fitted = True
        
        # 3. 提取归一化参数 (用于反标准化预测结果)
        if hasattr(self._regressor, 'y_train_mean_'):
            self.y_mean = self._regressor.y_train_mean_
            self.y_std = self._regressor.y_train_std_
        else:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y) + 1e-8

        # 4. 将训练数据缓存为 Tensor，作为推理时的 Context
        self.X_ctx = torch.tensor(X_padded, dtype=torch.float32, device=self.device).unsqueeze(0)
        y_normalized = (y - self.y_mean) / self.y_std
        self.y_ctx = torch.tensor(y_normalized, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 5. 提取内部模型配置
        self.models = self._regressor.models_

        for model in self.models:
            model.eval()  
            for param in model.parameters():
                param.requires_grad = False

        if self._new_engine_api:
            from tabpfn.preprocessing.datamodel import FeatureSchema

            configs = self._regressor.ensemble_configs_
            if len(configs) != 1:
                raise RuntimeError(
                    "Differentiable TabPFN 7.x inference requires n_estimators=1."
                )
            schema = FeatureSchema.from_only_categorical_indices(
                categorical_indices=[],
                num_columns=self.padding_dim,
            )
            self.ensemble_configs = [[configs[0]]]
            self.engine = InferenceEngineBatchedNoPreprocessing(
                X_trains=[self.X_ctx],
                y_trains=[self.y_ctx],
                feature_schema=[[schema]],
                models=self.models,
                ensemble_configs=self.ensemble_configs,
                devices=self._regressor.devices_,
                force_inference_dtype=torch.float32,
                inference_mode=False,
                dtype_byte_size=4,
                save_peak_mem=False,
            )
        else:
            configs = (
                self._regressor.executor_.ensemble_configs
                if hasattr(self._regressor.executor_, "ensemble_configs")
                else self._regressor.ensemble_configs
            )
            self.ensemble_configs = [configs]
            dummy_cat_ix = [[[] for _ in range(len(configs))]]
            self.engine = InferenceEngineBatchedNoPreprocessing(
                X_trains=[self.X_ctx],
                y_trains=[self.y_ctx],
                cat_ix=dummy_cat_ix,
                models=self.models,
                ensemble_configs=self.ensemble_configs,
                force_inference_dtype=torch.float32,
                inference_mode=False,
                dtype_byte_size=4,
                save_peak_mem=False,
            )
    
    def _forward_pass(self, X_input: torch.Tensor, requires_grad: bool = False, return_std: bool = False):
        """
        可微分的前向传播函数。
        
        Args:
            X_input: 输入 Tensor (标准化后的)。
            requires_grad: 是否需要保留梯度计算图 (优化时为 True)。
            return_std :是否需要返回标准差
            
        Returns:
            预测的物理量值 (Tensor)。
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        # 自动填充维度
        if X_input.shape[1] == self.original_dim:
            X_tensor_padded = self._pad_features(X_input, is_tensor=True)
        else:
            X_tensor_padded = X_input

        # 执行前向传播
        engine_args = {
            "X": [X_tensor_padded.unsqueeze(0)],
            "autocast": False,
        }
        if self._new_engine_api:
            engine_args.update(
                differentiable_input=requires_grad,
                task_type="regression",
            )
        iterator = self.engine.iter_outputs(**engine_args)
        all_logits = [out[0] for out in iterator]
        avg_logits = torch.mean(torch.stack(all_logits, dim=0), dim=0)
        
        # 将 Transformer 输出的 Logits 映射回真实物理值 (期望值计算)
        borders = self._regressor.znorm_space_bardist_.borders.to(self.device)
        probs = torch.softmax(avg_logits, dim=-1)
        
        # 计算分布的中心点
        centers = (borders[:-1] + borders[1:]) / 2 if borders.shape[0] == probs.shape[-1] + 1 else borders
            
        pred_norm = torch.sum(probs * centers, dim=-1)

        # 反标准化： Norm -> Real Value
        pred_real = pred_norm * self.y_std + self.y_mean

        if return_std:
            # 计算方差： sum( p * (x - mu)^2 )
            var_norm = torch.sum(probs * (centers - pred_norm.unsqueeze(-1))**2, dim=-1)
            # 反标准化：预测值和标准差
            pred_real = pred_norm * self.y_std + self.y_mean
            std_real = torch.sqrt(var_norm) * self.y_std 
            return pred_real.squeeze(0), std_real.squeeze(0)
        else:
            pred_real = pred_norm * self.y_std + self.y_mean
            return pred_real.squeeze(0)

    def predict(self, X_input: np.ndarray, batch_size: int = 1024, return_std: bool = False):
        """标准的 sklearn 风格预测接口 (仅用于评估，不支持梯度)。"""
        preds = []
        stds = []
        
        with torch.no_grad():
            X_scaled = self.scaler.transform(X_input)
            for i in range(0, len(X_scaled), batch_size):
                X_batch = X_scaled[i:i + batch_size]
                X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=self.device)
                
                if return_std:
                    pred_batch, std_batch = self._forward_pass(X_tensor, requires_grad=False, return_std=True)
                    preds.append(pred_batch.cpu().numpy().flatten())
                    stds.append(std_batch.cpu().numpy().flatten())
                else:
                    pred_batch = self._forward_pass(X_tensor, requires_grad=False, return_std=False)
                    preds.append(pred_batch.cpu().numpy().flatten())
                torch.cuda.empty_cache()
                
        if return_std:
            return np.concatenate(preds), np.concatenate(stds)
        return np.concatenate(preds)
    
    def predict_tensor(self, X_input_tensor: torch.Tensor):
        """
        专门为梯度优化设计的纯 PyTorch 预测接口。
        输入输出必须全是 Tensor，且保留梯度计算图 (requires_grad=True)。
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        # 1. 使用 PyTorch 实现 StandardScaler 的标准化（保留梯度）
        # 将 sklearn scaler 的参数转为 Tensor
        mean = torch.tensor(self.scaler.mean_, dtype=torch.float32, device=self.device)
        scale = torch.tensor(self.scaler.scale_, dtype=torch.float32, device=self.device)
        
        # 纯 Tensor 操作，梯度流不会断
        X_scaled_tensor = (X_input_tensor - mean) / scale
        
        # 2. 调用前向传播
        pred_real = self._forward_pass(X_scaled_tensor, requires_grad=True, return_std=False)
        return pred_real


class MultiTargetSurrogateManager:
    def __init__(self, target_names, device='cpu', padding_dim=18, scalers_dict=None):
        """
        参数:
            target_names: 目标列名列表 ['delay', 'power']
            padding_dim: TabPFN 要求的特征维度
            scalers_dict: 字典，允许为不同目标指定不同的 scaler。例如 {'power': MinMaxScaler()}
        """
        self.target_names = target_names
        self.device = device
        self.padding_dim = padding_dim
        self.scalers_dict = scalers_dict or {} 
        self.models = {}   
        self.shared_regressor = TabPFNRegressor(device=device, n_estimators=1, ignore_pretraining_limits=True)
        # One query means predicting all raw targets for one candidate design.
        # This counter is shared by EA, BO and the proposed method.
        self.query_count = 0

    def fit_all(self, X_train, y_train_df):
        print(f"Fitting models for {len(self.target_names)} targets...")
        for target in self.target_names:
            print(f"  -> Training surrogate for target: [{target}]")
            y_vals = y_train_df[target].values.ravel()
            
            specific_scaler = self.scalers_dict.get(target, StandardScaler())
            
            model = TabPFNSurrogate(
                device=self.device, 
                shared_regressor=self.shared_regressor,
                scaler=specific_scaler 
            )
            model.fit(X_train, y_vals)
            self.models[target] = model
            
    def predict(self, X_input, return_std=False):
        """批量预测接口，支持返回标准差"""
        self.query_count += int(len(X_input))
        predictions = {}
        stds = {}
        for target, model in self.models.items():
            if return_std:
                # 当 return_std=True 时，底层预测返回 (pred, std)
                predictions[target], stds[target] = model.predict(X_input, return_std=True)
            else:
                # 默认只返回 pred
                predictions[target] = model.predict(X_input, return_std=False)
        
        if return_std:
            return predictions, stds
        return predictions

    def predict_tensor(self, X_input_tensor: torch.Tensor):
        """批量张量预测接口"""
        self.query_count += int(X_input_tensor.shape[0])
        predictions = {}
        for target, model in self.models.items():
            # 获取单目标的预测值 (Batch, 1) -> Flatten to (Batch,)
            predictions[target] = model.predict_tensor(X_input_tensor).squeeze()
        return predictions

    def reset_query_count(self):
        self.query_count = 0

    def get_query_count(self):
        return int(self.query_count)

