import numpy as np
import pandas as pd


def calculate_scale_factor(actual: np.ndarray, model: np.ndarray) -> float:
    """
    Расчет коэффициента масштабирования k_scale методом наименьших квадратов
    """
    denom = np.dot(model, model)
    if denom <= 0:
        return 1.0
    scale = np.dot(actual, model) / denom
    # Ограничиваем физически адекватными рамками
    return float(np.clip(scale, 0.5, 2.0))


def run_2d_identification(
        dt_engine,
        train_df: pd.DataFrame,
        strings_range: list,
        modules_range: list,
        nominal_kw: float):
    """
    Выполнение 2D Grid Search для параметрической идентификации.
    dt_engine: экземпляр класса SolarDigitalTwin из pv_physics
    """
    actual_train = train_df["AC_POWER"].astype(float).values

    best_nmae = np.inf
    best_params = {"strings": None, "modules": None, "k_scale": None}
    best_sim_train = None

    print("Начинаем структурно-параметрическую идентификацию (Grey-box)...")

    for mps in modules_range:
        for strings in strings_range:
            # 1. Расчет базовой физики
            sim_train_base = dt_engine.simulate(train_df, strings, mps).values

            # 2. Идентификация k_scale
            k_scale = calculate_scale_factor(actual_train, sim_train_base)

            # 3. Расчет адаптированной мощности
            sim_train_adj = sim_train_base * k_scale

            # 4. Расчет невязки (nMAE)
            mae = np.abs(actual_train - sim_train_adj).mean()
            nmae = (mae / nominal_kw) * 100

            if nmae < best_nmae:
                best_nmae = nmae
                best_params = {"strings": strings, "modules": mps, "k_scale": k_scale}
                best_sim_train = sim_train_adj

    print(
        f"Оптимум найден: стрингов={best_params['strings']}, "
        f"модулей={best_params['modules']}, "
        f"k_scale={best_params['k_scale']:.3f}")
    return best_params, best_sim_train, best_nmae
