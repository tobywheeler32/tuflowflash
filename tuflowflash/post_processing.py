import os

import requests
from osgeo import gdalconst, osr
from pyproj import Proj, Transformer

try: 
    import gdal
except:
    from osgeo import gdal

import datetime
import glob
import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np

logger = logging.getLogger(__name__)
RASTER_SOURCES_URL = (
    "https://rhdhv.lizard.net/api/v4/rastersources/"  # use rastersources endpoint
)

class ProcessFlash():
    def __init__(self,settings):
        self.settings=settings

    def process_tuflow(self):
        self.convert_flt_to_tiff()
        logger.info("Tuflow results converted to tiff")
        self.delete_former_lizard_results()
        # _create_new_rastersource(settings)
        self.post_results_to_lizard()

    def process_bom(self):
        self.NC_to_tiffs(Path("temp"))
        self.post_bom_to_lizard()
        logger.info("Tuflow results posted to Lizard")

    def _create_new_rastersource(self):
        headers = {
            "username": "__key__",
            "password": self.settings.apikey,
        }
        configuration = {
            "name": "Ivar test raster bom",
            "description": "This is the decription of the test raster",
            "access_modifier": "Public",
            "organisation": "568a4d88c1b345668759dd9b305f619d",  # rhdhv organisation
            "temporal": True,  # temporal=true then
            "interval": "00:10:00",  # ISO 8601-format, ("1 01:00:00")
        }

        r = requests.post(url=RASTER_SOURCES_URL, json=configuration, headers=headers)


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
        filenames = glob.glob(os.path.join(self.settings.output_folder, "grids", "*_d_*.flt"))
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


    def post_results_to_lizard(self):
        filenames = glob.glob(os.path.join(self.settings.output_folder, "grids", "*_d_*.tif"))
        username = "__key__"
        password = self.settings.apikey
        headers = {
            "username": username,
            "password": password,
        }
        raster_url = RASTER_SOURCES_URL + self.settings.depth_raster_uuid + "/"
        url = raster_url + "data/"

        for file in filenames:
            tuflow_timestamp = Path(file).stem.split("_d_")[1]
            if tuflow_timestamp.lower() != "max":
                tuflow_timestamp_hours = int(tuflow_timestamp.split("_")[0])
                tuflow_timestamp_minutes = int(tuflow_timestamp.split("_")[1])
                timestamp = self.settings.start_time + datetime.timedelta(
                    hours=tuflow_timestamp_hours, minutes=tuflow_timestamp_minutes
                )
                logger.debug("posting file %s to lizard", file)
                lizard_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:00Z")
                file = {"file": open(file, "rb")}
                data = {"timestamp": lizard_timestamp}
                requests.post(url=url, data=data, files=file, headers=headers)
        return


    def delete_former_lizard_results(self):
        username = "__key__"
        password = self.settings.apikey
        headers = {
            "username": username,
            "password": password,
        }
        raster_url = RASTER_SOURCES_URL + self.settings.depth_raster_uuid + "/"
        url = raster_url + "data/"
        # hardcoded
        data = {"start": "2020-01-01T00:01:00Z", "stop": "2020-01-01T00:15:00Z"}
        r = requests.delete(url=url, json=data, headers=headers)


    def reproject_bom(self,x, y):
        transformer = Transformer.from_proj(Proj("epsg:4326"), Proj("epsg:3577"))
        x2, y2 = transformer.transform(y, x)
        return x2, y2


    def NC_to_tiffs(self, Output_folder):
        nc_data_obj = nc.Dataset(self.settings.netcdf_rainfall_file)
        x_center, y_center = self.reproject_bom(
            nc_data_obj.variables["proj"].longitude_of_central_meridian,
            nc_data_obj.variables["proj"].latitude_of_projection_origin,
        )
        Lon = nc_data_obj.variables["y"][:] * 1000 + x_center
        Lat = nc_data_obj.variables["x"][:] * 1000 + y_center
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
            srs.ImportFromEPSG(
                3577
            )  #  the coordinate system of the output
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
