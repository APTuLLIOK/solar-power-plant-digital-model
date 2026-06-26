import pandas as pd
from typing import Tuple, Set


def load_raw_datasets(
        gen_path: str,
        weather_path: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка сырых датасетов генерации и погоды."""
    df_gen = pd.read_csv(gen_path, parse_dates=["DATE_TIME"], dayfirst=True)
    df_weather = pd.read_csv(weather_path, parse_dates=["DATE_TIME"])

    df_gen = df_gen.sort_values("DATE_TIME").reset_index(drop=True)
    df_weather = df_weather.sort_values("DATE_TIME").reset_index(drop=True)

    return df_gen, df_weather


def prepare_inverter_data(
    df_gen: pd.DataFrame,
    df_weather: pd.DataFrame,
    inverter_key: str,
    tz: str,
    daylight_threshold: float
) -> pd.DataFrame:
    """
    Объединение данных погоды и генерации
    для конкретного инвертора с фильтрацией дня.
    """
    inv = df_gen[df_gen["SOURCE_KEY"] == inverter_key].copy()
    weather_clean = df_weather.drop(
        columns=["PLANT_ID", "SOURCE_KEY"], errors="ignore").copy()

    merged = pd.merge(inv, weather_clean, on="DATE_TIME", how="inner")
    merged = merged.sort_values("DATE_TIME").set_index("DATE_TIME")

    # Оставляем только дневные точки
    merged = merged[merged["IRRADIATION"] > daylight_threshold].copy()
    merged["IRRADIATION_WM2"] = merged["IRRADIATION"] * 1000.0

    if merged.index.tz is None:
        merged.index = merged.index.tz_localize(tz)
    else:
        merged.index = merged.index.tz_convert(tz)

    return merged


def split_train_test_days(
        df_weather: pd.DataFrame, train_size: float) -> Tuple[Set, Set]:
    """Разбиение дней на обучающую и тестовую выборки (без перемешивания)."""
    all_days = pd.Index(sorted(pd.unique(df_weather["DATE_TIME"].dt.date)))
    split_idx = int(len(all_days) * train_size)

    train_days = set(all_days[:split_idx])
    test_days = set(all_days[split_idx:])

    if not train_days.isdisjoint(test_days):
        raise ValueError("Train и test дни пересекаются!")

    return train_days, test_days
