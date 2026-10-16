import sys
import types

import numpy as np
import pytest

from size_optimization.demo_online_ctr import (
    MAX_TRAIN_SAMPLES,
    MIN_TRAIN_SAMPLES,
    OnlineTabPFNCTR,
    _farthest_indices,
    _kmeans_representative_indices,
    _load_ctr_config,
    _predicted_fom,
    _resolve_device,
    is_pareto_mask,
    resolve_training_target,
)


class _TwoDimensionalSpace:
    bounds = np.asarray([[0.0, 0.0], [1.0, 1.0]])

    @staticmethod
    def convert_params(point):
        return {"x0": float(point[0]), "x1": float(point[1])}


def test_ctr_gpu_index_selects_requested_visible_device(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 3)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _resolve_device("auto", 1) == "cuda:1"
    assert _resolve_device("cuda", 2) == "cuda:2"
    assert _resolve_device("cuda:0", 2) == "cuda:0"

    with pytest.raises(RuntimeError, match="only 3 CUDA"):
        _resolve_device("cuda:3", 0)


def test_context_budget_is_adaptive_and_reserves_thirty_audits():
    assert MIN_TRAIN_SAMPLES == 30
    assert MAX_TRAIN_SAMPLES == 300
    assert resolve_training_target(10) == 10
    assert resolve_training_target(59) == 59
    assert resolve_training_target(60) == 30
    assert resolve_training_target(200) == 170
    assert resolve_training_target(400) == MAX_TRAIN_SAMPLES
    assert resolve_training_target(1500) == MAX_TRAIN_SAMPLES
    assert resolve_training_target(400, 180) == 180

    with pytest.raises(ValueError, match="positive"):
        resolve_training_target(0)
    with pytest.raises(ValueError, match="training_evals"):
        resolve_training_target(400, 301)
    with pytest.raises(ValueError, match="training_evals"):
        resolve_training_target(60, 31)


def test_default_config_is_loaded_from_online_ctr_package():
    config = _load_ctr_config()
    assert config["target_context_size"] == 300
    assert config["coarse_evals"] == 1500
    assert config["device"] == "auto"


def test_acquisition_mode_matches_shared_budget_and_result_contract(monkeypatch):
    optimizer = OnlineTabPFNCTR(
        _TwoDimensionalSpace(),
        seed=7,
        device="cpu",
        gpu_index=0,
        training_evals=None,
        target_context_size=300,
        min_context_for_ctr=30,
        min_final_audits=30,
        initial_lhs_size=1,
        active_batch_size=1,
        active_candidates=128,
        pareto_bonus=2.0,
        novelty_weight=0.25,
        coarse_evals=4,
        coarse_pop_size=4,
        representative_points=1,
        refine_steps=1,
        refine_lr=0.05,
        refine_smoothing=0.05,
        physical_penalty=10.0,
        gradient_clip=5.0,
        fom_top_k=1,
        hv_reference_margin=0.1,
    )
    monkeypatch.setattr(optimizer, "fit_tabpfn", lambda: None)

    calls = []

    def evaluate(params):
        calls.append(params)
        result = {
            "min_snm": 0.2,
            "max_power": 1.0e-4,
            "area": 2.0e-12,
            "read_delay": 1.0e-9,
            "write_delay": 1.2e-9,
            "max_delay": 1.2e-9,
        }
        return [-0.2, 1.0e-4, 2.0e-12, 1.2e-9], [0.0, 0.0], result, True

    result = optimizer.run(evaluate, 3, initial_x=[0.25, 0.75])

    assert len(calls) == result["circuit_evaluations"] == result["n_evaluations"] == 3
    assert calls[0] == {"x0": pytest.approx(0.25), "x1": pytest.approx(0.75)}
    assert result["algorithm"] == "Online-CTR"
    assert result["mode"] == "acquisition-only"
    assert result["is_multiobjective"] is True
    assert result["params"] is not None
    assert result["result"] is not None
    assert result["merit"] is not None


def test_four_objective_pareto_mask_and_fom_are_consistent():
    # All objectives are minimized: [-SNM, power, area, delay].
    costs = np.asarray(
        [
            [-0.10, 2.0e-4, 2.0e-12, 2.0e-9],
            [-0.20, 1.0e-4, 1.0e-12, 1.0e-9],  # dominates row 0
            [-0.25, 3.0e-4, 1.0e-12, 1.0e-9],  # SNM/power trade-off
            [-0.20, 1.0e-4, 1.0e-12, 2.0e-9],  # dominated by row 1
        ]
    )
    assert is_pareto_mask(costs).tolist() == [False, True, True, False]
    fom = _predicted_fom(costs)
    assert np.isfinite(fom).all()
    assert fom[1] > fom[0]


def test_representative_selection_preserves_priorities_and_spreads_points():
    costs = np.asarray(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.25, 0.75, 0.7, 0.8],
            [0.5, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.3, 0.2],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    selected = _farthest_indices(costs, 3, priority=[2])
    assert selected[0] == 2
    assert len(selected) == len(set(selected)) == 3


def test_kmeans_representatives_are_existing_distinct_designs():
    costs = np.column_stack(
        [
            np.linspace(-0.3, -0.1, 30),
            np.linspace(1.0e-4, 3.0e-4, 30),
            np.linspace(1.0e-12, 3.0e-12, 30),
            np.linspace(1.0e-9, 3.0e-9, 30),
        ]
    )
    selected = _kmeans_representative_indices(costs, 20, seed=7)
    assert len(selected) == len(set(selected)) == 20
    assert all(0 <= index < len(costs) for index in selected)


def test_audit_selection_always_prioritizes_predicted_fom_top10():
    optimizer = OnlineTabPFNCTR.__new__(OnlineTabPFNCTR)
    x = np.column_stack([np.linspace(0.01, 0.99, 20), np.linspace(0.99, 0.01, 20)])
    # A trade-off curve with physical values; all decoded designs are distinct.
    y = np.column_stack(
        [
            -np.linspace(0.10, 0.30, 20),
            np.linspace(1.0e-4, 3.0e-4, 20),
            np.linspace(2.0e-12, 1.0e-12, 20),
            np.linspace(2.0e-9, 1.0e-9, 20),
        ]
    )
    optimizer.coarse_history_x = list(x)
    optimizer.coarse_history_y = list(y)
    optimizer.refine_history_x = []
    optimizer.refine_history_y = []
    optimizer.real_design_keys = set()
    optimizer.real_observations = []
    optimizer.fom_top_k = 10
    optimizer.hv_reference_margin = 0.10
    optimizer._decode = lambda point: {"x0": float(point[0]), "x1": float(point[1])}

    candidates, summary = optimizer.select_audit_candidates(10)
    labels = [label for _, label in candidates]
    assert len(candidates) == 10
    assert labels.count("final-fom-top10-audit") == 10
    assert summary["selected_fom_top10"] == 10
