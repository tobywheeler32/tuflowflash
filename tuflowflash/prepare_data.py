import datetime
import logging
import os
import shutil
import urllib.request as request
from contextlib import closing
from pathlib import Path
from typing import List
import ftplib
import time

import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

TIMESERIES_URL = "https://rhdhv.lizard.net/api/v4/timeseries/{}/events/"


class MissingFileException(Exception):
    pass


class prepareData:
    def __init__(self, settings):
        self.settings = settings

    def get_historical_precipitation(self):
        logger.info("Started gathering historical precipitation data")
        rainfall_gauges_uuids = self.read_rainfall_timeseries_uuids()

        rain_df = self.get_lizard_timeseries(rainfall_gauges_uuids,)
        logger.info("gathered lizard rainfall timeseries")

        ## preprocess rain data
        rain_df = self.process_rainfall_timeseries_for_tuflow(rain_df)
        rain_df.to_csv(self.settings.gauge_rainfall_file)
        logger.info("succesfully written rainfall file")

    def get_future_precipitation(self):
        sourcePath = Path(r"temp/tmp_rain.nc")
        self.download_bom_data()

        self.write_netcdf_with_time_indexes(
            sourcePath, self.settings.start_time, self.settings.end_time
        )
        logger.info("succesfully prepared netcdf rainfall")

    def read_rainfall_timeseries_uuids(self):
        rainfall_timeseries = pd.read_csv(self.settings.precipitation_uuid_file)
        return rainfall_timeseries

    def get_lizard_timeseries(self, rainfall_timeseries):
        # ugly code
        json_headers = {
            "username": "__key__",
            "password": self.settings.apikey,
            "Content-Type": "application/json",
        }

        timeseries_df_list = []
        gauge_names_list = []

        params = {
            "start": self.settings.start_time.isoformat(),
            "end": self.settings.end_time.isoformat(),
            "page_size": 100000,
        }

        for index, row in rainfall_timeseries.iterrows():
            r = requests.get(
                TIMESERIES_URL.format(row["gauge_uuid"]),
                params=params,
                headers=json_headers,
            )
            if r.ok:
                ts_df = pd.DataFrame(r.json()["results"])
                ts_df = ts_df[["time", "value"]]
                ts_df = ts_df.rename(columns={"value": row["gauge_name"]})
                ts_df.set_index("time", inplace=True)
                ts_df.index = pd.to_datetime(ts_df.index)
                timeseries_df_list.append(ts_df)
                gauge_names_list.append(row["gauge_name"])

        result = pd.concat(timeseries_df_list, axis=1)
        return result

    def process_rainfall_timeseries_for_tuflow(self, rain_df):
        rain_df["time"] = 0.0
        for x in range(len(rain_df)):
            if x > 0:
                rain_df["time"][x] = (
                    rain_df.index[x] - rain_df.index[0]
                ).seconds / 3600
        rain_df.set_index("time", inplace=True)
        for col in rain_df.columns:
            rain_df[col].values[0] = 0
        return rain_df

    def download_bom_data(self):

        ftp_server = ftplib.FTP(
            self.settings.bom_url,
            self.settings.bom_username,
            self.settings.bom_password,
        )
        ftp_server.encoding = "utf-8"
        ftp_server.cwd("radar/")

        radar_files = []
        files = ftp_server.nlst()

        for file in files:
            if file[6:8] == "EN":
                radar_files.append(file)

        bomfile = radar_files[-1]
        if not os.path.exists("temp"):
            os.mkdir("temp")

        with open("temp/tmp_rain.nc", "wb") as file:
            ftp_server.retrbinary(f"RETR {bomfile}", file.write)
        ftp_server.close()
        return

    def timestamps_from_netcdf(
        self, source_file: Path
    ) -> List[cftime.DatetimeGregorian]:
        source = nc.Dataset(source_file)
        timestamps = nc.num2date(source["valid_time"][:], source["valid_time"].units)
        source.close()
        return timestamps

    def get_p50_netcdf_rainfall(self, source):
        # select 50pth percentile rainfall
        cum_rainfall_list = []
        for x in range(len(source.variables["precipitation"])):
            cum_rainfall_list.append(
                np.sum(np.sum(source.variables["precipitation"][x]))
            )
            p = np.percentile(cum_rainfall_list, 10)
            closest_to_p50 = min(cum_rainfall_list, key=lambda x: abs(x - p))
            p50_index = cum_rainfall_list.index(closest_to_p50)
        return p50_index

    def write_new_netcdf(self, source_file: Path, time_indexes: List):

        source = nc.Dataset(source_file)
        target = nc.Dataset(self.settings.netcdf_rainfall_file, mode="w")
        p50_index = self.get_p50_netcdf_rainfall(source)
        # Create the dimensions of the file.
        for name, dim in source.dimensions.items():
            dim_length = len(dim)
            if name == "valid_time":
                dim_length = len(time_indexes)
            target.createDimension(name, dim_length if not dim.isunlimited() else None)

        # Copy the global attributes.
        target.setncatts({a: source.getncattr(a) for a in source.ncattrs()})
        # Create the variables in the file.
        for name, var in source.variables.items():

            if name == "precipitation":
                target.createVariable(name, var.dtype, ("valid_time", "y", "x"))
            elif name == "valid_time":
                target.createVariable(name, float, var.dimensions)
            else:
                target.createVariable(name, var.dtype, var.dimensions)
            # Copy the variable attributes.
            target.variables[name].setncatts(
                {a: var.getncattr(a) for a in var.ncattrs()}
            )
            data = source.variables[name][:]
            # Copy the variables values.
            if name == "valid_time":
                data = data[time_indexes]
                data = data - data[0]
                data = data / 3600
                target.renameVariable("valid_time", "time")
                target.variables["time"][:] = data
            elif name == "precipitation":
                data = data[p50_index, time_indexes]
                target.renameVariable("precipitation", "rainfall_depth")
                target.variables["rainfall_depth"][:, :, :] = data
            elif name == "x" or name == "y":
                target.variables[name][:] = data

        # Save the file.
        target.close()
        source.close()

    def write_netcdf_with_time_indexes(self, source_file: Path, start, end):
        """Return netcdf file with only time indexes"""
        if not source_file.exists():
            raise MissingFileException("Source netcdf file %s not found", source_file)

        # logger.info("Converting %s to a file with only time indexes", source_file)
        relevant_timestamps = self.timestamps_from_netcdf(source_file)
        # Figure out which timestamps are valid for the given simulation period.
        time_indexes: List = (
            np.argwhere(  # type: ignore
                (relevant_timestamps >= start)  # type: ignore
                & (relevant_timestamps <= end)  # type: ignore
            )
            .flatten()
            .tolist()
        )

        self.write_new_netcdf(source_file, time_indexes)
        logger.debug(
            "Wrote new time-index-only netcdf to %s", self.settings.netcdf_rainfall_file
        )
