import numpy as np
import pytest

from size_optimization.sram_opt.contract import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
)
from size_optimization.sram_opt.evaluator import system_metrics
from size_optimization.sram_opt.predictor import validate_bundle
from size_optimization.sram_opt.runner import run_search
from size_optimization.sram_opt.sampling import constrained_lhs, maximin_indices
from size_optimization.sram_opt.space import SramOptDesignSpace


class _Predictor:
    feasibility_target = 0.90
    provenance = {"source": "test-double"}

    @staticmethod
    def predict(designs):
        return np.full(len(designs), 0.95)


def _evaluate(params):
    scale = float(params["pd_width"] / params["pg_width"])
    result = {
        "min_snm": 0.15 + 0.01 * scale,
        "max_power": 1.0e-4,
        "area": 2.0e-10,
        "max_delay": 1.5e-9,
        "read_delay": 1.5e-9,
        "write_delay": 1.0e-9,
    }
    return [-result["min_snm"], result["max_power"], result["area"], result["max_delay"]], [0.0], result, True


def test_v226_space_preserves_the_original_feature_contract():
    space = SramOptDesignSpace()
    point = space.get_default_normalized_vector()
    params = space.convert_params(point)
    row = space.feature_row(point)

    assert len(FEATURE_COLUMNS) == 26
    assert space.search_dimension == 25
    assert space.bounds.shape == (2, 25)
    assert tuple(row) == FEATURE_COLUMNS
    assert params["pd_model_name"] == params["pg_model_name"] == row["nmos_model_name"]
    assert row["rows"] * row["cols"] * row["num_arrays"] == 262_144
    assert row["cols"] <= 256


def test_constrained_lhs_is_deterministic_and_space_filling():
    space = SramOptDesignSpace()
    first, first_report = constrained_lhs(
        space, 4, seed=73, oversample_factor=3, models=None
    )
    second, second_report = constrained_lhs(
        space, 4, seed=73, oversample_factor=3, models=None
    )
    assert np.array_equal(first, second)
    assert first_report == second_report
    assert first.shape == (4, 25)
    assert len(set(maximin_indices(first, 3))) == 3


@pytest.mark.parametrize("backend", ["SA", "LHS"])
def test_injected_sram_opt_search_needs_no_model_or_spice_data(backend):
    result = run_search(
        SramOptDesignSpace(),
        _evaluate,
        _Predictor(),
        max_iter=3,
        seed=42,
        model_aware_screen=False,
        backend=backend,
    )
    assert result["algorithm"] == "SramOpt"
    assert result["n_evaluations"] == 3
    assert result["feasible_evaluations"] == 3
    assert result["params"] is not None
    assert result["result"] is not None
    assert result["predicted_yield"] == pytest.approx(0.95)


def test_search_rejects_a_good_nominal_circuit_below_the_yield_target():
    result = run_search(
        SramOptDesignSpace(),
        _evaluate,
        _Predictor(),
        max_iter=2,
        seed=42,
        yield_target=0.99,
        model_aware_screen=False,
    )
    assert result["n_evaluations"] == 2
    assert result["feasible_evaluations"] == 0
    assert result["params"] is None
    assert result["result"] is None
    assert all(item["circuit_success"] for item in result["history"])


def test_bundle_contract_rejects_feature_reordering():
    bundle = {
        "feature_columns": list(reversed(FEATURE_COLUMNS)),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "categorical_feature_indices": [
            FEATURE_COLUMNS.index(name) for name in CATEGORICAL_COLUMNS
        ],
        "ordinal_encoder": object(),
        "numeric_medians": {},
        "x_train": np.zeros((2, 26)),
        "y_train_model": np.zeros(2),
        "checkpoint": "model.ckpt",
        "n_estimators": 1,
        "target_transform": "raw",
        "feasibility_target": 0.9,
    }
    with pytest.raises(ValueError, match="ordered 26-feature"):
        validate_bundle(bundle)


def test_system_metrics_rejects_failed_raw_delay_before_scaling():
    raw = {
        "hold_snm": 0.2,
        "read_snm": 0.15,
        "write_snm": 0.25,
        "read_delay": -1.0,
        "write_delay": 1.0e-9,
        "read_pstc": 1.0e-6,
        "read_pdyn": 2.0e-6,
        "write_pstc": 1.0e-6,
        "write_pdyn": 2.0e-6,
    }
    with pytest.raises(ValueError, match="invalid raw read_delay"):
        system_metrics({"num_arrays": 4, "cols": 64}, raw, 1.0e-10)
