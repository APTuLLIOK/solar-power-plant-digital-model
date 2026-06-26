import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array
from pvlib.modelchain import ModelChain
from .config import config


class SolarPowerPlantModel:
    def __init__(self):
        self.site_location = Location(
            config.station.lat,
            config.station.lon,
            config.station.tz,
            config.station.altitude,
            config.station.name
        )
        self._load_hardware()

    def _load_hardware(self):
        """Загрузка профилей оборудования из баз CEC."""
        cec_modules = pvlib.pvsystem.retrieve_sam("CECMod")
        cec_inverters = pvlib.pvsystem.retrieve_sam("cecinverter")

        self.module = cec_modules[config.hardware.module_name]

        # Поиск инвертора
        paco_min = config.hardware.inverter_paco_min
        paco_max = config.hardware.inverter_paco_max
        suitable = cec_inverters.loc[
            :,
            (cec_inverters.loc["Paco"] > paco_min)
            & (cec_inverters.loc["Paco"] < paco_max)
            ]

        if suitable.shape[1] == 0:
            raise ValueError(
                f"Не найден инвертор в диапазоне {paco_min}-{paco_max} Вт."
                )

        self.inverter = suitable[suitable.columns[0]]
        self.inverter_paco_kw = self.inverter["Paco"] / 1000.0

        self.temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
            config.hardware.temperature_model
        ][config.hardware.temperature_mount]

    def build_system(self, strings: int, modules_per_string: int) -> PVSystem:
        """Сборка PV системы."""
        array = Array(
            mount=pvlib.pvsystem.FixedMount(
                surface_tilt=15, surface_azimuth=180
                ),
            module_parameters=self.module,
            temperature_model_parameters=self.temp_params,
            modules_per_string=modules_per_string,
            strings=strings,
        )
        return PVSystem(arrays=[array], inverter_parameters=self.inverter)

    def simulate(
            self,
            merged_df: pd.DataFrame,
            strings: int,
            modules_per_string: int
            ) -> pd.Series:
        """Расчет генерации для заданного датафрейма."""
        solpos = self.site_location.get_solarposition(merged_df.index)
        erbs_out = pvlib.irradiance.erbs(
            merged_df["IRRADIATION_WM2"],
            solpos["zenith"],
            merged_df.index
        )

        model_weather = pd.DataFrame({
            "ghi": merged_df["IRRADIATION_WM2"],
            "dni": erbs_out["dni"],
            "dhi": erbs_out["dhi"],
            "temp_air": merged_df["AMBIENT_TEMPERATURE"],
            "wind_speed": 0.0,
            "module_temperature": merged_df["MODULE_TEMPERATURE"],
        }, index=merged_df.index)

        system = self.build_system(strings, modules_per_string)
        mc = ModelChain(
            system,
            self.site_location,
            aoi_model="physical",
            spectral_model="no_loss"
        )
        mc.run_model(model_weather)

        # Возврат в кВт
        return mc.results.ac / 1000.0
