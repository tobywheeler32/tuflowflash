import argparse
import configparser as ConfigParser
import datetime
import logging
import os
import subprocess
from pathlib import Path
from typing import List

import cftime
import netCDF4 as nc
import pandas as pd
import requests
from osgeo import gdalconst, osr
from pyproj import Proj, Transformer

try:
    import gdal
except:
    from osgeo import gdal

import glob
import shutil
import urllib.request as request
from contextlib import closing

import numpy as np

logger = logging.getLogger(__name__)
tijd = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TIMESERIES_URL = "https://rhdhv.lizard.net/api/v4/timeseries/{}/events/"
RASTER_SOURCES_URL = (
    "https://rhdhv.lizard.net/api/v4/rastersources/"  # use rastersources endpoint
)


class MissingFileException(Exception):
    pass


def _format_logger(kwargs):
    """Format logger to write logging accross python modules 
    as well as formatting output format and switching between
    verbose and regular logging.
    """

    if kwargs["verbose"]:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logger.setLevel(log_level)

    if not os.path.exists("log"):
        os.mkdir("log")

    fh = logging.FileHandler(
        os.path.join("log", "tuflow_simulation{dt}.log".format(dt=tijd))
    )
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%d/%m/%Y %I:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _create_new_rastersource(settings):
    headers = {
        "username": "__key__",
        "password": settings["apikey"],
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
    print(r.json())


def create_projection(settings):
    """obtain wkt definition of the tuflow spatial projection. Used to write 
    geotiff format files with gdal.
    """
    prj_text = open(settings["prj_file"], "r").read()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(prj_text)
    srs_wkt = srs.ExportToWkt()
    return srs_wkt


def convert_flt_to_tiff(settings):

    gdal.UseExceptions()
    filenames = glob.glob(os.path.join(settings["output_folder"], "grids", "*_d_*.flt"))
    for file in filenames:
        # load dem raster file
        data = gdal.Open(file, gdalconst.GA_ReadOnly)
        nodata = data.GetRasterBand(1).GetNoDataValue()
        data_array = data.GetRasterBand(1).ReadAsArray()
        geo_transform = data.GetGeoTransform()
        proj = create_projection(settings)
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


def post_results_to_lizard(settings):
    filenames = glob.glob(os.path.join(settings["output_folder"], "grids", "*_d_*.tif"))
    username = "__key__"
    password = settings["apikey"]
    headers = {
        "username": username,
        "password": password,
    }
    raster_url = RASTER_SOURCES_URL + settings["depth_raster_uuid"] + "/"
    url = raster_url + "data/"

    for file in filenames:
        tuflow_timestamp = Path(file).stem.split("_d_")[1]
        if tuflow_timestamp.lower() != "max":
            tuflow_timestamp_hours = int(tuflow_timestamp.split("_")[0])
            tuflow_timestamp_minutes = int(tuflow_timestamp.split("_")[1])
            timestamp = settings["start_time"] + datetime.timedelta(
                hours=tuflow_timestamp_hours, minutes=tuflow_timestamp_minutes
            )
            logger.debug("posting file %s to lizard", file)
            lizard_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:00Z")
            file = {"file": open(file, "rb")}
            data = {"timestamp": lizard_timestamp}
            requests.post(url=url, data=data, files=file, headers=headers)
    return


def delete_former_lizard_results(settings):
    username = "__key__"
    password = settings["apikey"]
    headers = {
        "username": username,
        "password": password,
    }
    raster_url = RASTER_SOURCES_URL + settings["depth_raster_uuid"] + "/"
    url = raster_url + "data/"
    # hardcoded
    data = {"start": "2020-01-01T00:01:00Z", "stop": "2020-01-01T00:15:00Z"}
    r = requests.delete(url=url, json=data, headers=headers)
    print(r.json())


def read_rainfall_timeseries_uuids(settings):
    rainfall_timeseries = pd.read_csv(Path(settings["precipitation_uuid_file"]))
    return rainfall_timeseries


def get_lizard_timeseries(settings, rainfall_timeseries, starttime, endtime):
    # ugly code
    json_headers = {
        "username": "__key__",
        "password": settings["apikey"],
        "Content-Type": "application/json",
    }

    timeseries_df_list = []
    gauge_names_list = []

    params = {
        "start": starttime.isoformat(),
        "end": endtime.isoformat(),
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


def process_rainfall_timeseries_for_tuflow(rain_df):
    rain_df["time"] = 0.0
    for x in range(len(rain_df)):
        if x > 0:
            rain_df["time"][x] = (rain_df.index[x] - rain_df.index[0]).seconds / 3600
    rain_df.set_index("time", inplace=True)
    for col in rain_df.columns:
        rain_df[col].values[0] = 0
    return rain_df


def download_bom_data(settings):
    bomfile = settings["bom_file"] + ".20190417215000.nc"
    if not os.path.exists("temp"):
        os.mkdir("temp")
    tmp_rainfile = Path("temp/tmp_rain.nc")
    with closing(request.urlopen(settings["bom_url"] + bomfile)) as r:
        with open(tmp_rainfile, "wb") as f:
            shutil.copyfileobj(r, f)


def reproject_bom(x, y):
    transformer = Transformer.from_proj(Proj("epsg:4326"), Proj("epsg:3577"))
    x2, y2 = transformer.transform(y, x)
    return x2, y2


def NC_to_tiffs(data, Output_folder):
    nc_data_obj = nc.Dataset(data)
    x_center, y_center = reproject_bom(
        nc_data_obj.variables["proj"].longitude_of_central_meridian,
        nc_data_obj.variables["proj"].latitude_of_projection_origin,
    )
    Lon = nc_data_obj.variables["y"][:] * 1000 + x_center
    Lat = nc_data_obj.variables["x"][:] * 1000 + y_center
    precip_arr = np.asarray(
        nc_data_obj.variables["rainfall_depth"]
    )  # read ndvi data into an array
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
        )  #  the coordinate system of the output is defined as "WGS 84"，AUTHORITY["EPSG","4326"]
        out_tif.SetProjection(
            srs.ExportToWkt()
        )  #  give the new layer projection information

        # data write
        out_tif.GetRasterBand(1).WriteArray(
            precip_arr[i]
        )  #  writes data to memory, not to disk at this time
        out_tif.FlushCache()  #  write data to hard disk
        out_tif = None  #  note that the tif file must be closed


def post_bom_to_lizard(settings):
    filenames = glob.glob(os.path.join("temp", "*.tif"))
    username = "__key__"
    password = settings["apikey"]
    headers = {
        "username": username,
        "password": password,
    }
    raster_url = RASTER_SOURCES_URL + settings["rainfall_raster_uuid"] + "/"
    url = raster_url + "data/"

    for file in filenames:
        rain_timestamp = Path(file).stem
        timestamp = settings["start_time"] + datetime.timedelta(
            hours=float(rain_timestamp)
        )
        logger.debug("posting file %s to lizard", file)
        lizard_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:00Z")
        file = {"file": open(file, "rb")}
        data = {"timestamp": lizard_timestamp}
        requests.post(url=url, data=data, files=file, headers=headers)
    return


def timestamps_from_netcdf(source_file: Path) -> List[cftime.DatetimeGregorian]:
    source = nc.Dataset(source_file)
    timestamps = nc.num2date(source["valid_time"][:], source["valid_time"].units)
    source.close()
    return timestamps


def get_p50_netcdf_rainfall(source):
    # select 50pth percentile rainfall
    cum_rainfall_list = []
    for x in range(len(source.variables["precipitation"])):
        cum_rainfall_list.append(np.sum(np.sum(source.variables["precipitation"][x])))
        p = np.percentile(cum_rainfall_list, 10)
        closest_to_p50 = min(cum_rainfall_list, key=lambda x: abs(x - p))
        p50_index = cum_rainfall_list.index(closest_to_p50)
    return p50_index


def write_new_netcdf(source_file: Path, target_file: Path, time_indexes: List):

    source = nc.Dataset(source_file)
    target = nc.Dataset(target_file, mode="w")
    p50_index = get_p50_netcdf_rainfall(source)
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
        target.variables[name].setncatts({a: var.getncattr(a) for a in var.ncattrs()})
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


def write_netcdf_with_time_indexes(source_file: Path, settings, start, end):
    """Return netcdf file with only time indexes"""
    if not source_file.exists():
        raise MissingFileException("Source netcdf file %s not found", source_file)

    # logger.info("Converting %s to a file with only time indexes", source_file)
    relevant_timestamps = timestamps_from_netcdf(source_file)
    # Figure out which timestamps are valid for the given simulation period.
    time_indexes: List = (
        np.argwhere(  # type: ignore
            (relevant_timestamps >= start)  # type: ignore
            & (relevant_timestamps <= end)  # type: ignore
        )
        .flatten()
        .tolist()
    )

    write_new_netcdf(source_file, settings["netcdf_rainfall_file"], time_indexes)
    logger.debug(
        "Wrote new time-index-only netcdf to %s", settings["netcdf_rainfall_file"]
    )
    return


def run_tuflow(cmd):
    try:
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
        )
        output, error = process.communicate()
        returncode = process.poll()
        logger.info(output)
        if returncode != 0:
            raise ValueError("Executable terminated, see log file")
    except (ValueError, IndexError):
        exit("Executable terminated, see log file")


def copy_states(settings):
    trf_input_file = settings["tcf_file"].replace(".tcf", ".trf")
    trf_result_file = Path(os.path.join(settings["output_folder"], trf_input_file))

    if trf_result_file.exists():
        shutil.copyfile(trf_result_file, trf_input_file)
    else:
        raise MissingFileException("Source trf file %s not found", trf_result_file)

    erf_input_file = settings["tcf_file"].replace(".tcf", ".erf")
    erf_result_file = Path(os.path.join(settings["output_folder"], erf_input_file))
    if erf_result_file.exists():
        shutil.copyfile(erf_result_file, erf_input_file)
    else:
        raise MissingFileException("Source erf file %s not found", erf_result_file)


def extract_variable_from_tcf(line):
    variable = line.split("==")[1]
    variable = variable.split("!")[0].strip()
    return variable


def read_tcf_parameters(kwargs):
    with open(kwargs["tcf_file"]) as f:
        lines = f.readlines()
    for line in lines:
        if line.lower().startswith("start time"):
            kwargs["tuflow_start_time"] = float(extract_variable_from_tcf(line))
        if line.lower().startswith("output folder"):
            kwargs["output_folder"] = Path(extract_variable_from_tcf(line))
        if line.lower().startswith("shp projection"):
            kwargs["prj_file"] = Path(extract_variable_from_tcf(line))

    return kwargs


def roundTime(dt=None, roundTo=60):
    """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   """
    if dt is None:
        dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def convert_relative_time(relative_time):
    now = roundTime()
    if relative_time.startswith("-"):
        time = now - datetime.timedelta(hours=float(relative_time.strip("-")))
    elif relative_time.startswith("+"):
        time = now + datetime.timedelta(hours=float(relative_time.strip("-")))
    else:
        time = now + datetime.timedelta(hours=float(relative_time))
    return time


def set_settings(**kwargs):
    # maak de output van deze functie aan
    _kwargs = kwargs

    # lees instellingen uit de inifile
    config = ConfigParser.RawConfigParser()
    config.read(kwargs["instellingen"])

    # general settings
    _kwargs["start_time"] = convert_relative_time(
        config.get("general", "relative_start_time")
    )
    _kwargs["end_time"] = convert_relative_time(
        config.get("general", "relative_end_time")
    )

    # tuflow settings
    _kwargs["tuflow_executable"] = config.get("tuflow", "tuflow_executable")
    _kwargs["tcf_file"] = config.get("tuflow", "tcf_file")
    _kwargs["prepare_state_for_next_run"] = config.get(
        "tuflow", "prepare_state_for_next_run"
    )
    _kwargs["sim_duration"] = config.get("tuflow", "sim_duration")
    _kwargs["gauge_rainfall_file"] = config.get("tuflow", "gauge_rainfall_file")
    _kwargs["netcdf_rainfall_file"] = config.get("tuflow", "netcdf_rainfall_file")

    # bom settings
    _kwargs["bom_url"] = config.get("bom", "bom_url")
    _kwargs["bom_file"] = config.get("bom", "bom_file")

    # lizard settings
    _kwargs["apikey"] = config.get("lizard", "apikey")
    _kwargs["precipitation_uuid_file"] = config.get("lizard", "precipitation_uuid_file")
    _kwargs["depth_raster_uuid"] = config.get("lizard", "depth_raster_uuid")
    _kwargs["rainfall_raster_uuid"] = config.get("lizard", "rainfall_raster_uuid")

    # switches
    _kwargs["run_simulation"] = (
        config.get("switches", "run_simulation").lower() == "true"
    )
    _kwargs["get_historical_precipitation"] = (
        config.get("switches", "get_historical_precipitation").lower() == "true"
    )
    _kwargs["get_future_precipitation"] = (
        config.get("switches", "get_future_precipitation").lower() == "true"
    )
    _kwargs["post_to_lizard"] = (
        config.get("switches", "post_to_lizard").lower() == "true"
    )

    _kwargs = read_tcf_parameters(_kwargs)
    return _kwargs


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    # POSITIONAL ARGUMENTS

    parser.add_argument(
        "instellingen",
        metavar="instellingen",
        help="Het .ini bestand met de instellingen voor de voor/naverwerking.",
    )

    # OPTIONAL ARGUMENTS
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Verbose output",
    )

    parser.add_argument(
        "-rainfall", metavar="netcdf", default=None, help="netcdf rainfall file"
    )
    return parser


def main():
    """ Call command with args from parser. """
    ## read settings
    kwargs = vars(get_parser().parse_args())
    logger = _format_logger(kwargs)
    settings = set_settings(**kwargs)
    logger.info("settings have been read")

    ## historical precipitation
    if settings["get_historical_precipitation"]:
        logger.info("Started gathering historical precipitation data")
        rainfall_gauges_uuids = read_rainfall_timeseries_uuids(settings)

        rain_df = get_lizard_timeseries(
            settings,
            rainfall_gauges_uuids,
            settings["start_time"],
            settings["end_time"],
        )
        logger.info("gathered lizard rainfall timeseries")

        ## preprocess rain data
        rain_df = process_rainfall_timeseries_for_tuflow(rain_df)
        rain_df.to_csv(settings["gauge_rainfall_file"])
        logger.info("succesfully written rainfall file")
    else:
        logger.info("not gathering historical rainfall, skipping..")

    ## future precipitation
    if settings["get_future_precipitation"]:
        download_bom_data(settings)

        sourcePath = Path(r"temp\tmp_rain.nc")
        # temp
        start = datetime.datetime(2019, 4, 17, 21, 50)
        end = datetime.datetime(2019, 4, 17, 23, 25)
        # regular
        write_netcdf_with_time_indexes(sourcePath, settings, start, end)
        logger.info("succesfully prepared netcdf rainfall")
    else:
        logger.info("not gathering future rainfall data, skipping..")

    ## run simulation
    if settings["run_simulation"]:
        logger.info("starting TUFLOW simulation")
        run_tuflow([settings["tuflow_executable"], "-b", settings["tcf_file"]])
        if settings["prepare_state_for_next_run"]:
            copy_states(settings)
        logger.info("Tuflow simulation finished")
    else:
        logger.info("Not running Tuflow simulation, skipping..")

    ## uploading to Lizard
    if settings["post_to_lizard"]:
        convert_flt_to_tiff(settings)
        logger.info("Tuflow results converted to tiff")
        delete_former_lizard_results(settings)
        # _create_new_rastersource(settings)
        post_results_to_lizard(settings)
        NC_to_tiffs(settings["netcdf_rainfall_file"], Path("temp"))
        post_bom_to_lizard(settings)
        logger.info("Tuflow results posted to Lizard")
    else:
        logger.info("Not uploading files to Lizard, skipping..")
    return 1


if __name__ == "__main__":
    exit(main())