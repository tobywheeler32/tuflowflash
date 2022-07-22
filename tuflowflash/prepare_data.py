from pathlib import Path
from pyproj import Proj
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from requests.adapters import Retry
from shapely.geometry import mapping
from typing import List

import cftime
import ftplib
import geopandas
import glob
import gzip
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import pytz
import requests
import rioxarray
import shutil
import time


logger = logging.getLogger(__name__)

TIMESERIES_URL = "https://rhdhv.lizard.net/api/v4/timeseries/{}/events/"
FTP_RETRY_COUNT = 10
FTP_RETRY_SLEEP = 5

S = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
S.mount("http://", HTTPAdapter(max_retries=retries))


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
        local = pytz.timezone("Australia/Sydney")
        local_reference_time = local.localize(self.settings.reference_time, is_dst=None)
        utc_reference_time = local_reference_time.astimezone(pytz.utc)

        rain_df = self.process_rainfall_timeseries_for_tuflow(
            rain_df, utc_reference_time
        )
        rain_df = rain_df.fillna(-99)
        rain_df.to_csv(self.settings.gauge_rainfall_file)
        logger.info("succesfully written rainfall file")

    def get_precipitation_nowcast(self):
        sourcePath = Path(r"temp/radar_rain.nc")
        self.download_bom_radar_data(self.settings.bom_nowcast_file)

        local = pytz.timezone("Australia/Sydney")
        local_start = local.localize(self.settings.start_time, is_dst=None)
        utc_start = local_start.astimezone(pytz.utc)
        local_end = local.localize(self.settings.end_time, is_dst=None)
        utc_end = local_end.astimezone(pytz.utc)

        self.write_nowcast_netcdf_with_time_indexes(
            sourcePath,
            self.settings.netcdf_nowcast_rainfall_file,
            utc_start,
            utc_end,
            self.settings.reference_time,
        )
        logger.info("succesfully prepared netcdf radar rainfall")

    def get_precipitation_forecast(self):
        sourcePath = Path(r"temp/forecast_rain.nc")
        self.download_bom_forecast_data(self.settings.bom_forecast_file)

        self.write_forecast_netcdf_with_time_indexes(
            sourcePath,
            self.settings.netcdf_forecast_rainfall_file,
            self.settings.forecast_clipshape,
            self.settings.start_time,
            self.settings.end_time,
            self.settings.reference_time,
        )
        logger.info("succesfully prepared netcdf radar rainfall")

    def convert_csv_file_to_bc_file(self):
        csv_df = pd.read_csv(self.settings.boundary_csv_input_file, delimiter=",")
        csv_df["Time (h)"] = pd.to_datetime(csv_df["datetime"], dayfirst=True)
        csv_df.set_index("Time (h)", inplace=True)
        csv_df.index = (csv_df.index - self.settings.reference_time) / np.timedelta64(
            1, "h"
        )
        csv_df.to_csv(self.settings.boundary_csv_tuflow_file)
        logger.info("succesfully converted csv to boundary file")

    def write_forecast_netcdf_with_time_indexes(
        self, sourcePath, output_file, clipshape, start_time, end_time, reference_time
    ):
        geodf = geopandas.read_file(clipshape)
        xds = rioxarray.open_rasterio(sourcePath)
        xds = xds.rio.write_crs(4326)
        source = xds.rio.clip(geodf.geometry.apply(mapping), geodf.crs)
        xds_lonlat = source.rio.reproject("EPSG:7856", resolution=5500)
        xds_lonlat = xds_lonlat.rename("rainfall_depth")
        xds_lonlat[:, :, :] = np.where(
            xds_lonlat == xds_lonlat.attrs["_FillValue"], 0, xds_lonlat
        )
        xds_lonlat = xds_lonlat.sel(time=slice(start_time, end_time))
        xds_lonlat = xds_lonlat.assign_coords(
            time=(xds_lonlat["time"] - reference_time) / 3600000000000
        )
        xds_lonlat.to_netcdf(output_file)

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

        local = pytz.timezone("Australia/Sydney")
        local_start = local.localize(self.settings.start_time, is_dst=None)
        utc_start = local_start.astimezone(pytz.utc)
        local_end = local.localize(self.settings.end_time, is_dst=None)
        utc_end = local_end.astimezone(pytz.utc)

        params = {
            "time__gte": utc_start.isoformat(),
            "time__lte": utc_end.isoformat(),
            "page_size": 100000,
        }
        for index, row in rainfall_timeseries.iterrows():
            r = S.get(
                TIMESERIES_URL.format(row["gauge_uuid"]),
                params=params,
                headers=json_headers,
            )
            if r.ok:
                ts_df = pd.DataFrame(r.json()["results"])
                if "time" in ts_df.columns:
                    ts_df = ts_df[["time", "value"]]
                    ts_df = ts_df.rename(columns={"value": row["gauge_name"]})
                    ts_df.set_index("time", inplace=True)
                    ts_df.index = pd.to_datetime(ts_df.index)
                    timeseries_df_list.append(ts_df)
                    gauge_names_list.append(row["gauge_name"])
                else:
                    data = {"time": [utc_start], row["gauge_name"]: [-99]}
                    ts_df = pd.DataFrame(data)
                    ts_df.set_index("time", inplace=True)
                    ts_df.index = pd.to_datetime(ts_df.index)
                    timeseries_df_list.append(ts_df)
                    gauge_names_list.append(row["gauge_name"])

        result = pd.concat(timeseries_df_list, axis=1)
        return result

    def process_rainfall_timeseries_for_tuflow(self, rain_df, utc_reference_time):
        rain_df["time"] = 0.0
        rain_df["datetime"] = rain_df.index
        for x in range(len(rain_df)):
            timedifference = rain_df.index[x] - utc_reference_time
            rain_df["time"][x] = (
                timedifference.days * 86400 + timedifference.seconds
            ) / 3600
        rain_df.set_index("time", inplace=True)
        for col in rain_df.columns:
            rain_df[col].values[0] = 0
        return rain_df

    def download_bom_radar_data(self, nowcast_file):
        for x in range(FTP_RETRY_COUNT):
            try:
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
                    if file.startswith(nowcast_file):
                        radar_files.append(file)

                bomfile = radar_files[-1]
                if not os.path.exists("temp"):
                    os.mkdir("temp")

                with open("temp/radar_rain.nc", "wb") as file:
                    ftp_server.retrbinary(f"RETR {bomfile}", file.write)
                ftp_server.close()
            except ftplib.error_temp:
                logger.warning("Temporary ftp issue, retrying in: %s", FTP_RETRY_SLEEP)
                time.sleep(FTP_RETRY_SLEEP)
            else:
                break
        return

    def download_bom_forecast_data(self, bomfile):
        for x in range(FTP_RETRY_COUNT):
            try:
                ftp_server = ftplib.FTP(
                    self.settings.bom_url,
                    self.settings.bom_username,
                    self.settings.bom_password,
                )
                ftp_server.encoding = "utf-8"
                ftp_server.cwd("adfd/")
                tmp_rainfile = Path("temp/" + bomfile)

                with open(tmp_rainfile, "wb") as f:
                    ftp_server.retrbinary(f"RETR {bomfile}", f.write)

                with gzip.open(tmp_rainfile, "rb") as f_in:
                    with open("temp/forecast_rain.nc", "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                logging.info("succesfully downloaded %s", bomfile)
            except ftplib.error_temp:
                logger.warning("Temporary ftp issue, retrying in: %s", FTP_RETRY_SLEEP)
                time.sleep(FTP_RETRY_SLEEP)
            else:
                break

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

    def write_new_netcdf(
        self, source_file: Path, dest_file: Path, time_indexes: List, reference_time
    ):
        source = nc.Dataset(source_file)
        x_center, y_center = self.reproject_bom(
            source.variables["proj"].longitude_of_central_meridian,
            source.variables["proj"].latitude_of_projection_origin,
        )
        target = nc.Dataset(dest_file, mode="w")
        p50_index = self.get_p50_netcdf_rainfall(source)
        # Create the dimensions of the file.
        for name, dim in source.dimensions.items():
            dim_length = len(dim)
            if name == "valid_time":
                dim_length = len(time_indexes)
                target.createDimension(
                    "time", dim_length if not dim.isunlimited() else None
                )
            if name == "x" or name == "y":
                target.createDimension(
                    name, dim_length if not dim.isunlimited() else None
                )
            if name == "precipitation":
                target.createDimension(
                    "rainfall_depth", dim_length if not dim.isunlimited() else None
                )

        # Copy the global attributes.
        target.setncatts({a: source.getncattr(a) for a in source.ncattrs()})
        # Create the variables in the file.

        for name, var in source.variables.items():

            if name == "precipitation":
                target.createVariable("rainfall_depth", float, ("time", "y", "x"))
            elif name == "valid_time":
                target.createVariable("time", float, "time")
                target.variables["time"].setncatts(
                    {
                        "standard_name": "time",
                        "long_name": "time",
                        "units": "hours",
                        "axis": "T",
                    }
                )
            elif name == "x":
                target.createVariable(name, var.dtype, var.dimensions)
                target.variables[name].setncatts(
                    {
                        "standard_name": "projection_x_coordinate",
                        "long_name": "x-coordinate in cartesian system",
                        "units": "m",
                        "axis": "X",
                    }
                )
            elif name == "y":
                target.createVariable(name, var.dtype, var.dimensions)
                target.variables[name].setncatts(
                    {
                        "standard_name": "projection_y_coordinate",
                        "long_name": "y-coordinate in cartesian system",
                        "units": "m",
                        "axis": "Y",
                    }
                )

            data = source.variables[name][:]
            # Copy the variables values.
            if name == "valid_time":
                data = data[time_indexes]
                data = (data - reference_time.timestamp()) / 3600
                target.variables["time"][:] = data
            elif name == "precipitation":
                data = data[p50_index, time_indexes]
                data = data * 0.05
                data = np.where(data < 0, -999, data)
                target.variables["rainfall_depth"][:, :, :] = data
            elif name == "x":
                target.variables[name][:] = data * 1000 + x_center
            elif name == "y":
                target.variables[name][:] = data * 1000 + y_center
        # Save the file.
        target.close()
        source.close()

    def write_nowcast_netcdf_with_time_indexes(
        self, source_file: Path, dest_file: Path, start, end, reference_time
    ):
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
        self.write_new_netcdf(source_file, dest_file, time_indexes, reference_time)
        logger.debug("Wrote new time-index-only netcdf to %s", dest_file)

    def reproject_bom(self, x, y):
        transformer = Transformer.from_proj(Proj("epsg:4326"), Proj("epsg:7856"))
        x2, y2 = transformer.transform(y, x)
        return x2, y2

    def forecast_nowcast_netcdf_to_ascii(self, netcdf_file):
        nc_data_obj = nc.Dataset(netcdf_file)
        Lon = nc_data_obj.variables["x"][:]
        Lat = nc_data_obj.variables["y"][:]
        precip_arr = np.asarray(
            nc_data_obj.variables["rainfall_depth"]
        )  # read data into an array
        # the upper-left and lower-right coordinates of the image
        LonMin, LatMax, LatMin = [Lon.min(), Lat.max(), Lat.min()]

        # resolution calculation
        N_Lat = len(Lat)
        Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)

        for i in range(len(precip_arr[:])):
            header = "ncols     %s\n" % precip_arr[i].shape[1]
            header += "nrows    %s\n" % precip_arr[i].shape[0]
            header += "xllcorner {}\n".format(LonMin)
            header += "yllcorner {}\n".format(LatMin)
            header += "cellsize {}\n".format(Lat_Res)
            header += "NODATA_value -9999\n"

            np.savetxt(
                os.path.join(
                    self.settings.rain_grids_folder,
                    str(nc_data_obj.variables["time"][i]) + ".asc",
                ),
                precip_arr[i],
                header=header,
                fmt="%1.2f",
                comments="",
            )

    def hindcast_netcdf_to_ascii(self):
        for f in glob.glob(str(self.settings.historic_rain_folder) + "/*.nc"):
            print(f)

    def write_ascii_csv(self):
        time_stamps = nc_data_obj.variables["time"][:]
        file_names = []
        for timestamp in time_stamps:
            file_names.append("RFG\\" + str(timestamp) + ".asc")
        df = pd.DataFrame()
        df["Time (hrs)"] = time_stamps
        df["Rainfall Grid"] = file_names
        df.set_index("Time (hrs)", inplace=True)
        df.to_csv(self.settings.rain_grids_csv)
        nc_data_obj.close()
