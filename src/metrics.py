import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from typing import Dict, Any


def calculate_station_metrics(
    station_actual: pd.Series,
    station_model: pd.Series,
    station_actual_train: pd.Series,
    station_model_train: pd.Series,
    station_actual_test: pd.Series,
    station_model_test: pd.Series,
    n_inverters: int,
    inverter_paco_kw: float
) -> Dict[str, Any]:

    station_nominal_ac_kw = n_inverters * inverter_paco_kw

    station_mae = np.abs(station_actual - station_model).mean()
    station_rmse = np.sqrt(np.mean((station_actual - station_model) ** 2))
    station_nmae = (station_mae / station_nominal_ac_kw) * 100
    station_r2 = r2_score(station_actual, station_model)

    train_mae = np.abs(station_actual_train - station_model_train).mean()
    train_nmae = (train_mae / station_nominal_ac_kw) * 100

    test_mae = np.abs(station_actual_test - station_model_test).mean()
    test_nmae = (test_mae / station_nominal_ac_kw) * 100

    return {
        "n_inverters": n_inverters,
        "nominal_power_kw": station_nominal_ac_kw,
        "station_nmae": station_nmae,
        "station_rmse": station_rmse,
        "train_nmae": train_nmae,
        "test_nmae": test_nmae,
        "station_r2": station_r2
    }
