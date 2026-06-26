import logging
import os
import pandas as pd
from src.config import config
from src.data_loader import (
    load_raw_datasets, prepare_inverter_data, split_train_test_days
    )
from src.pv_model import SolarPowerPlantModel
from src.calibrator import optimize_inverter_params
from src.metrics import calculate_station_metrics
from src.visualization import plot_station_results

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Инициализация Цифрового Двойника СЭС...")

    # 1. Загрузка данных
    df_gen, df_weather = load_raw_datasets(
        config.gen_data_path, config.weather_data_path)
    logger.info(
        f"Данные загружены. Генерация: {df_gen.shape}, "
        f"Погода: {df_weather.shape}"
        )

    train_days, test_days = (
        split_train_test_days(df_weather, config.grid_search.train_size)
    )
    logger.info(
        f"Дней для калибровки: {len(train_days)}, "
        f"Дней для проверки: {len(test_days)}"
        )

    inverter_keys = sorted(df_gen["SOURCE_KEY"].unique())
    logger.info(f"Найдено инверторов: {len(inverter_keys)}")

    # 2. Инициализация физической модели
    dt_model = SolarPowerPlantModel()

    # 3. Агрегаторы станции
    calibration_rows = []
    station_actual, station_model = None, None
    station_actual_train, station_model_train = None, None
    station_actual_test, station_model_test = None, None

    # 4. Цикл калибровки инверторов
    for inv_key in inverter_keys:
        merged = prepare_inverter_data(
            df_gen,
            df_weather,
            inv_key,
            config.station.tz,
            config.daylight_threshold
            )

        if len(merged) < 100:
            logger.warning(
                f"Пропуск инвертора {inv_key}: "
                f"недостаточно данных ({len(merged)} строк)."
                )
            continue

        day_index = pd.Series(
            merged.index.date,
            index=merged.index
            )
        train_df = merged[day_index.isin(train_days)].copy()
        test_df = merged[day_index.isin(test_days)].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning(
                f"Пропуск инвертора {inv_key}: пустой train или test."
                )
            continue

        # Grid Search
        result = optimize_inverter_params(
                inv_key=inv_key,
                train_df=train_df,
                test_df=test_df,
                merged_df=merged,
                dt_model=dt_model,
                string_grid=config.grid_search.string_grid,
                modules_grid=config.grid_search.modules_grid
            )

        calibration_rows.append(result.metrics)
        logger.info(
            f"{inv_key}: strings={result.metrics['best_strings']}, "
            f"modules={result.metrics['best_modules_per_string']}, "
            f"scale={result.metrics['scale_factor']:.3f}, "
            f"train_nMAE={result.metrics['train_nmae_%']:.2f}%, "
            f"test_nMAE={result.metrics['test_nmae_%']:.2f}%"
        )

        # Агрегация рядов на уровне станции
        station_actual = (
            result.actual_full
            if station_actual is None
            else station_actual.add(result.actual_full, fill_value=0)
        )
        station_model = (
            result.sim_full
            if station_model is None
            else station_model.add(result.sim_full, fill_value=0)
        )

        station_actual_train = (
            result.actual_train
            if station_actual_train is None
            else station_actual_train.add(result.actual_train, fill_value=0)
        )
        station_model_train = (
            result.sim_train
            if station_model_train is None
            else station_model_train.add(result.sim_train, fill_value=0)
        )

        station_actual_test = (
            result.actual_test
            if station_actual_test is None
            else station_actual_test.add(result.actual_test, fill_value=0)
        )
        station_model_test = (
            result.sim_test
            if station_model_test is None
            else station_model_test.add(result.sim_test, fill_value=0)
        )

    # 5. Итоговые метрики
    calibration_df = (
        pd.DataFrame(calibration_rows)
        .sort_values("train_nmae_%")
        .reset_index(drop=True)
    )
    logger.info("Калибровка завершена.\n" + str(calibration_df.head(10)))

    metrics = calculate_station_metrics(
        station_actual, station_model,
        station_actual_train, station_model_train,
        station_actual_test, station_model_test,
        n_inverters=len(calibration_df),
        inverter_paco_kw=dt_model.inverter_paco_kw
    )

    print("=" * 60)
    print("ЦИФРОВАЯ МОДЕЛЬ СОЛНЕЧНОЙ ЭЛЕКТРОСТАНЦИИ")
    print(
        f"Число смоделированных инверторных подсистем: "
        f"{metrics['n_inverters']}"
        )
    print(
        f"Суммарная номинальная AC-мощность: "
        f"{metrics['nominal_power_kw']:.1f} кВт"
        )
    print(f"nMAE по всей станции: {metrics['station_nmae']:.2f}%")
    print(f"RMSE по всей станции: {metrics['station_rmse']:.2f} кВт")
    print(f"nMAE на калибровке: {metrics['train_nmae']:.2f}%")
    print(f"nMAE на проверке: {metrics['test_nmae']:.2f}%")
    print(f"R² по всей станции: {metrics['station_r2']:.4f}")
    print("=" * 60)

    # 6. Отрисовка графиков
    os.makedirs("images", exist_ok=True)
    plot_station_results(station_actual, station_model, "images/station")


if __name__ == "__main__":
    main()
