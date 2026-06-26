from dataclasses import dataclass, field
from typing import List


@dataclass
class StationConfig:
    lat: float = 14.81
    lon: float = 78.01
    tz: str = "Asia/Kolkata"
    altitude: float = 300.0
    name: str = "India_Plant_1"


@dataclass
class HardwareConfig:
    module_name: str = "Canadian_Solar_Inc__CS6X_320P"
    inverter_paco_min: float = 950000
    inverter_paco_max: float = 1050000
    temperature_model: str = "sapm"
    temperature_mount: str = "open_rack_glass_glass"


@dataclass
class GridSearchConfig:
    # Использование range и списков для перебора
    string_grid: range = field(default_factory=lambda: range(180, 281, 5))
    modules_grid: List[int] = field(
        default_factory=lambda: [18, 19, 20, 21, 22])
    train_size: float = 0.7


@dataclass
class AppConfig:
    gen_data_path: str = "data/Plant_1_Generation_Data.csv"
    weather_data_path: str = "data/Plant_1_Weather_Sensor_Data.csv"
    daylight_threshold: float = 0.01

    station: StationConfig = field(default_factory=StationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    grid_search: GridSearchConfig = field(default_factory=GridSearchConfig)


config = AppConfig()
