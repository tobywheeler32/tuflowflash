import configparser as ConfigParser
import datetime
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

Tuflow_settings = {
    "tuflow_executable": str,
    "tcf_file": Path,
    "manage_states": bool,
    "initial_states_folder": Path,
    "export_states_folder": Path,
    "states_expiry_time_days": int,
    "gauge_rainfall_file": Path,
    "netcdf_forecast_rainfall_file": Path,
    "netcdf_nowcast_rainfall_file": Path,
    "archive_folder": Path,
}

lizard_settings = {
    "apikey": str,
    "precipitation_uuid_file": Path,
    "depth_raster_uuid": str,
    "rainfall_raster_uuid": str,
    "waterlevel_result_uuid_file": Path,
    "raster_upload_list": list,
}

switches_settings = {
    "get_historical_precipitation": bool,
    "get_bom_forecast": bool,
    "get_bom_nowcast": bool,
    "run_simulation": bool,
    "post_to_lizard": bool,
    "archive_simulation": bool,
}

bom_settings = {
    "bom_username": str,
    "bom_password": str,
    "bom_url": str,
    "bom_forecast_file": str,
    "bom_nowcast_file": str,
}


class MissingFileException(Exception):
    pass


class MissingSettingException(Exception):
    pass


class FlashSettings:
    def __init__(self, settingsFile: Path):
        self.settingsFile = settingsFile
        self.config = ConfigParser.RawConfigParser()
        try:
            self.config.read(settingsFile)
        except FileNotFoundError as e:
            msg = f"Settings file '{settingsFile}' not found"
            raise MissingFileException(msg) from e

        self.read_settings_file(Tuflow_settings, "tuflow")
        self.read_settings_file(lizard_settings, "lizard")
        self.read_settings_file(switches_settings, "switches")
        self.read_settings_file(bom_settings, "bom")

        self.read_tcf_parameters(self.tcf_file)

        self.start_time = self.convert_relative_time(
            self.config.get("general", "relative_start_time")
        )
        self.end_time = self.convert_relative_time(
            self.config.get("general", "relative_end_time")
        )

        logger.info("settings have been read")

    def extract_variable_from_tcf(self, line):
        variable = line.split("==")[1]
        variable = variable.split("!")[0].strip()
        return variable

    def read_tcf_parameters(self, tcf_file):
        with open(tcf_file) as f:
            lines = f.readlines()
        for line in lines:
            if line.lower().startswith("start time"):
                setattr(
                    self,
                    "tuflow_start_time",
                    float(self.extract_variable_from_tcf(line)),
                )
            if line.lower().startswith("output folder"):
                setattr(
                    self, "output_folder", Path(self.extract_variable_from_tcf(line))
                )
            if line.lower().startswith("shp projection"):
                setattr(self, "prj_file", Path(self.extract_variable_from_tcf(line)))

    def roundTime(self, dt=None, roundTo=60):
        """Round a datetime object to any time lapse in seconds
        dt : datetime.datetime object, default now.
        roundTo : Closest number of seconds to round to, default 1 minute.
        """
        if dt is None:
            dt = datetime.datetime.utcnow()
        seconds = (dt.replace(tzinfo=None) - dt.min).seconds
        rounding = (seconds + roundTo / 2) // roundTo * roundTo
        return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)

    def convert_relative_time(self, relative_time):
        now = self.roundTime()
        if relative_time.startswith("-"):
            time = now - datetime.timedelta(hours=float(relative_time.strip("-")))
        elif relative_time.startswith("+"):
            time = now + datetime.timedelta(hours=float(relative_time.strip("-")))
        else:
            time = now + datetime.timedelta(hours=float(relative_time))
        return time

    def read_settings_file(self, variables, variable_header):
        # maak de output van deze functie aan
        for variable, datatype in variables.items():
            value = self.config.get(variable_header, variable)
            if len(value) > 0:
                try:
                    if datatype == int:
                        setattr(self, variable, int(value))
                    if datatype == Path:
                        try:
                            setattr(self, variable, Path(value))
                        except:
                            logger.warning("no value found for %s",variable)
                    if datatype == str:
                        setattr(self, variable, str(value))
                    if datatype == bool:
                        setattr(
                            self, variable, value.lower() == "true",
                        )
                    if datatype == list:
                        input_list = string_to_list(variable)
                        setattr(self,variable,input_list)
                except:
                    raise MissingSettingException(
                        f"Setting: '{variable}' is missing in " f"{self.settingsFile}."
                    )

def string_to_list(string):
    return string.split(",")