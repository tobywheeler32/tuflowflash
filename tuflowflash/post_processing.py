import os

from osgeo import gdalconst, osr
from pyproj import Proj, Transformer
import pandas as pd

try:
    import gdal
except:
    from osgeo import gdal

import datetime
import glob
import logging
from pathlib import Path
import json
import requests
import shutil

import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)
RASTER_SOURCES_URL = (
    "https://rhdhv.lizard.net/api/v4/rastersources/"  # use rastersources endpoint
)

TIMESERIES_URL = "https://rhdhv.lizard.net/api/v4/timeseries/"


class ProcessFlash:
    def __init__(self, settings):
        self.settings = settings

    def process_tuflow(self):
        self.convert_flt_to_tiff()
        logger.info("Tuflow results converted to tiff")
        self.post_raster_to_lizard()
        if self.settings.waterlevel_result_uuid_file:
            self.post_timeseries()

    def process_bom(self):
        self.NC_to_tiffs(Path("temp"))
        self.post_bom_to_lizard()
        logger.info("Tuflow results posted to Lizard")

    def archive_simulation(self):
        folder_time_string = (
            str(self.settings.start_time).replace(":", "_").replace(" ", "_")
        )
        result_folder = Path("../results/results_" + folder_time_string)
        os.mkdir(result_folder)
        shutil.copytree("log", os.path.join(result_folder, "log"))
        shutil.copytree(
            self.settings.output_folder, os.path.join(result_folder, "results")
        )
        if self.settings.netcdf_rainfall_file:
            shutil.copyfile(
                self.settings.netcdf_rainfall_file,
                os.path.join(result_folder, "netcdf_rain.nc"),
            )
        if self.settings.gauge_rainfall_file:
            shutil.copyfile(
                self.settings.gauge_rainfall_file,
                os.path.join(result_folder, "gauge_rain.csv"),
            )
        shutil.make_archive(result_folder, "zip", result_folder)
        shutil.rmtree(result_folder)
        logging.info("succesfully archived files to: %s", result_folder)

    def create_projection(self):
        """obtain wkt definition of the tuflow spatial projection. Used to write 
        geotiff format files with gdal.
        """
        prj_text = open(self.settings.prj_file, "r").read()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(prj_text)
        srs_wkt = srs.ExportToWkt()
        return srs_wkt

    def convert_flt_to_tiff(self):
        gdal.UseExceptions()
        filenames = glob.glob(
            os.path.join(self.settings.output_folder, "grids", "*_d_Max*.flt")
        )
        for file in filenames:
            # load dem raster file
            data = gdal.Open(file, gdalconst.GA_ReadOnly)
            nodata = data.GetRasterBand(1).GetNoDataValue()
            data_array = data.GetRasterBand(1).ReadAsArray()
            geo_transform = data.GetGeoTransform()
            proj = self.create_projection()
            x_res = data.RasterXSize
            y_res = data.RasterYSize

            output = file.replace(".flt", ".tif")
            target_ds = gdal.GetDriverByName("GTiff").Create(
                output, x_res, y_res, 1, gdal.GDT_Float32
            )
            target_ds.SetGeoTransform(geo_transform)
            target_ds.GetRasterBand(1).WriteArray(data_array)
            target_ds.SetProjection(proj)
            target_ds.GetRasterBand(1).SetNoDataValue(nodata)
            target_ds.FlushCache()
            target_ds = None
        return

    def post_raster_to_lizard(self):
        filenames = glob.glob(
            os.path.join(self.settings.output_folder, "grids", "*_d_Max*.tif")
        )
        username = "__key__"
        password = self.settings.apikey
        headers = {
            "username": username,
            "password": password,
        }
        raster_url = RASTER_SOURCES_URL + self.settings.depth_raster_uuid + "/"
        url = raster_url + "data/"

        for file in filenames:
            logger.debug("posting file %s to lizard", file)
            file = {"file": open(file, "rb")}
            requests.post(url=url, files=file, headers=headers)
        return

    def create_post_element(self, series):
        data = []
        for index, value in series.iteritems():
            data.append({"datetime": index.isoformat() + "Z", "value": str(value)})
        return json.dumps(data)

    def post_timeseries(self):
        username = "__key__"
        password = self.settings.apikey
        headers = {
            "username": username,
            "password": password,
        }

        result_ts_uuids = pd.read_csv(self.settings.waterlevel_result_uuid_file)
        # temp
        file_name = "Fm_Exst_061_02.00p+01440m+tp07+21D+T_HAT_40m+FP.csv"

        results_dataframe = pd.read_csv(file_name)
        results_dataframe.columns = results_dataframe.columns.str.rstrip(
            "[" + Path(file_name).stem + "]"
        )
        results_dataframe.columns = results_dataframe.columns.str.rstrip(" ")
        for index, row in results_dataframe.iterrows():
            results_dataframe.at[
                index, "time"
            ] = self.settings.start_time + datetime.timedelta(hours=row["Time (h)"])
        results_dataframe.set_index("time", inplace=True)

        for index, row in result_ts_uuids.iterrows():
            timeserie = self.create_post_element(results_dataframe[row["po_name"]])
            url = TIMESERIES_URL + row["ts_uuid"]
            requests.post(url=url, data=timeserie, headers=headers)

    def NC_to_tiffs(self, Output_folder):
        nc_data_obj = nc.Dataset(self.settings.netcdf_rainfall_file)
        Lon = nc_data_obj.variables["y"][:]
        Lat = nc_data_obj.variables["x"][:]
        precip_arr = np.asarray(
            nc_data_obj.variables["rainfall_depth"]
        )  # read data into an array
        # the upper-left and lower-right coordinates of the image
        LonMin, LatMax, LonMax, LatMin = [Lon.min(), Lat.max(), Lon.max(), Lat.min()]

        # resolution calculation
        N_Lat = len(Lat)
        N_Lon = len(Lon)
        Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
        Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)

        for i in range(len(precip_arr[:])):
            # to create. tif file
            driver = gdal.GetDriverByName("GTiff")
            out_tif_name = os.path.join(
                Output_folder, str(nc_data_obj.variables["time"][i]) + ".tif"
            )
            out_tif = driver.Create(out_tif_name, N_Lon, N_Lat, 1, gdal.GDT_Float32)

            #  set the display range of the image
            # -Lat_Res must be - the
            geotransform = (LonMin, Lon_Res, 0, LatMax, 0, -Lat_Res)
            out_tif.SetGeoTransform(geotransform)

            # get geographic coordinate system information to select the desired geographic coordinate system
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(7856)  #  the coordinate system of the output
            out_tif.SetProjection(
                srs.ExportToWkt()
            )  #  give the new layer projection information

            # data write
            out_tif.GetRasterBand(1).WriteArray(
                precip_arr[i]
            )  #  writes data to memory, not to disk at this time
            out_tif.FlushCache()  #  write data to hard disk
            out_tif = None  #  note that the tif file must be closed

    def post_bom_to_lizard(self):
        filenames = glob.glob(os.path.join("temp", "*.tif"))
        username = "__key__"
        password = self.settings.apikey
        headers = {
            "username": username,
            "password": password,
        }
        raster_url = RASTER_SOURCES_URL + self.settings.rainfall_raster_uuid + "/"
        url = raster_url + "data/"

        for file in filenames:
            rain_timestamp = Path(file).stem
            timestamp = self.settings.start_time + datetime.timedelta(
                hours=float(rain_timestamp)
            )
            logger.debug("posting file %s to lizard", file)
            lizard_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:00Z")
            file = {"file": open(file, "rb")}
            data = {"timestamp": lizard_timestamp}
            requests.post(url=url, data=data, files=file, headers=headers)
        return
