import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any
from .pv_model import SolarPowerPlantModel


@dataclass
class InverterOptimizationResult:
    """Контейнер для хранения результатов оптимизации одного инвертора."""
    metrics: Dict[str, Any]
    actual_full: pd.Series
    sim_full: pd.Series
    actual_train: pd.Series
    sim_train: pd.Series
    actual_test: pd.Series
    sim_test: pd.Series


def fit_scale_factor(actual: np.ndarray, model: np.ndarray) -> float:
    """МНК для поиска оптимального коэффициента масштабирования."""
    actual_arr = np.asarray(actual, dtype=float)
    model_arr = np.asarray(model, dtype=float)

    denom = np.dot(model_arr, model_arr)
    if denom <= 0:
        return 1.0

    scale = np.dot(actual_arr, model_arr) / denom
    return float(np.clip(scale, 0.5, 2.0))


def optimize_inverter_params(
    inv_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    dt_model: SolarPowerPlantModel,
    string_grid: range,
    modules_grid: list
) -> InverterOptimizationResult:
    """
    Перебор параметров (Grid Search) для поиска лучшей конфигурации инвертора.
    """

    actual_train = train_df["AC_POWER"].astype(float)
    actual_test = test_df["AC_POWER"].astype(float)
    actual_full = merged_df["AC_POWER"].astype(float)

    best_strings = None
    best_modules = None
    best_scale_factor = None
    best_train_nmae = np.inf
    best_train_sim = None

    inverter_paco_kw = dt_model.inverter_paco_kw

    for mps in modules_grid:
        for s in string_grid:
            sim_train = dt_model.simulate(train_df, s, mps)
            scale_factor = fit_scale_factor(
                actual_train.values, sim_train.values)
            sim_train_adj = sim_train * scale_factor

            mae = np.abs(actual_train.values - sim_train_adj.values).mean()
            nmae = (mae / inverter_paco_kw) * 100

            if nmae < best_train_nmae:
                best_train_nmae = nmae
                best_strings = s
                best_scale_factor = scale_factor
                best_train_sim = sim_train_adj
                best_modules = mps

    if best_strings is None or best_modules is None:
        raise RuntimeError(
            f"Не удалось подобрать параметры для инвертора {inv_key}."
            )

    # Расчет полного временного ряда с лучшими параметрами
    sim_full = (
        dt_model.simulate(
            merged_df, best_strings, best_modules) * best_scale_factor
    )
    sim_test = sim_full.loc[test_df.index]

    test_mae = np.abs(actual_test.values - sim_test.values).mean()
    test_nmae = (test_mae / inverter_paco_kw) * 100

    metrics_dict = {
        "SOURCE_KEY": inv_key,
        "best_strings": best_strings,
        "best_modules_per_string": best_modules,
        "scale_factor": best_scale_factor,
        "train_nmae_%": best_train_nmae,
        "test_nmae_%": test_nmae,
    }

    return InverterOptimizationResult(
        metrics=metrics_dict,
        actual_full=actual_full,
        sim_full=sim_full,
        actual_train=actual_train,
        sim_train=best_train_sim,
        actual_test=actual_test,
        sim_test=sim_test
    )
