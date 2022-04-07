import subprocess
import argparse
import logging
import os
import configparser as ConfigParser
from pathlib import Path
from typing import List
import cftime
import netCDF4 as nc
import pandas as pd
import datetime
import shutil
import requests
from osgeo import osr, gdalconst
try:
    import gdal
except:
    from osgeo import gdal
import glob
import numpy as np

logger = logging.getLogger(__name__)
tijd = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TIMESERIES_URL = "https://rhdhv.lizard.net/api/v4/timeseries/{}/events/"
RASTER_SOURCES_URL = (
    "https://rhdhv.lizard.net/api/v4/rastersources/"  # use rastersources endpoint
)

class MissingFileException(Exception):
    pass

def format_logger():
    logger.setLevel(logging.DEBUG)
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


def create_new_rastersource(settings):
    headers = {
        "username": "__key__",
        "password": settings["apikey"],
    }
    configuration = {
        "name": "Ivar test raster Tuflow5",
        "description": "This is the decription of the test raster",
        "access_modifier": "Public",
        "organisation": "568a4d88c1b345668759dd9b305f619d",  # rhdhv organisation
        "temporal": True,  # temporal=true then
        "interval": "00:05:00",  # ISO 8601-format, ("1 01:00:00")
    }

    r=requests.post(url=RASTER_SOURCES_URL, json=configuration, headers=headers)
    print(r.json())

def create_projection(settings):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32760)
    srs_wkt = srs.ExportToWkt()
    return srs_wkt


def convert_flt_to_tiff(settings):

    gdal.UseExceptions()
    filenames = glob.glob(os.path.join(settings["grid_results_dir"], "*_d_*.flt"))
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
    filenames = glob.glob(os.path.join(settings["grid_results_dir"], "*_d_*.tif"))
    username = "__key__"
    password = settings["apikey"]
    headers = {
        "username": username,
        "password": password,
    }
    raster_url = RASTER_SOURCES_URL + settings["depth_raster_uuid"] + "/"
    url = raster_url + "data/"

    for file in filenames:
        print(file)
        file = {"file": open(file, "rb")}
        # hardcoded
        timestamp = "2020-01-01T00:10:00Z"
        data = {"timestamp": timestamp}

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
    r=requests.delete(url=url, json=data, headers=headers)
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


def timestamps_from_netcdf(source_file: Path) -> List[cftime.DatetimeGregorian]:
    source = nc.Dataset(source_file)
    timestamps = nc.num2date(source["valid_time"][:], source["valid_time"].units)
    source.close()
    return timestamps

def get_p50_netcdf_rainfall(source):
    #select 50pth percentile rainfall
    cum_rainfall_list=[]
    for x in range(len(source.variables['precipitation'])):
        cum_rainfall_list.append(np.sum(np.sum(source.variables['precipitation'][x])))
        p = np.percentile(cum_rainfall_list, 10)
        closest_to_p50=min(cum_rainfall_list, key=lambda x:abs(x-p))
        p50_index=cum_rainfall_list.index(closest_to_p50)
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

        if name=='precipitation':
            target.createVariable(name, var.dtype,('valid_time', 'y', 'x'))
        elif name=='valid_time':
            target.createVariable(name, float,var.dimensions)
        else:
            target.createVariable(name, var.dtype, var.dimensions)
        # Copy the variable attributes.
        target.variables[name].setncatts({a: var.getncattr(a) for a in var.ncattrs()})
        data = source.variables[name][:]
        # Copy the variables values (as 'f4' eventually).
        if name =="valid_time":
            data = data[time_indexes]
            data = data/3600
            target.variables[name][:] = data
        elif name == "precipitation":
            data = data[p50_index,time_indexes]
            target.variables[name][:,:,:] = data
        elif name == "x" or name == "y":
            target.variables[name][:] = data
    target.renameVariable("precipitation", "rainfall_depth")
    target.renameVariable("valid_time", "time")
    # Save the file.
    target.close()
    source.close()


def write_netcdf_with_time_indexes(source_file: Path, start, end):
    """Return netcdf file with only time indexes"""
    if not source_file.exists():
        raise MissingFileException("Source netcdf file %s not found", source_file)

    #logger.info("Converting %s to a file with only time indexes", source_file)
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

    # Create new file with only time indexes
    #temp_dir = Path(tempfile.mkdtemp(prefix="fews-3di"))
    #target_file = temp_dir / source_file.name

    target_file=r"C:\Users\922383\Documents\TUFLOW\test.nc"
    write_new_netcdf(source_file, target_file, time_indexes)
    #logger.debug("Wrote new time-index-only netcdf to %s", target_file)
    return target_file

def system_custom(cmd):
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


def set_settings(**kwargs):
    # maak de output van deze functie aan
    _kwargs = kwargs

    # lees instellingen uit de inifile
    config = ConfigParser.RawConfigParser()
    config.read(kwargs["instellingen"])

    # tuflow settings
    _kwargs["tuflow_executable"] = config.get("tuflow", "tuflow_executable")
    _kwargs["tcf_file"] = config.get("tuflow", "tcf_file")
    _kwargs["use_states"] = config.get("tuflow", "use_states")
    _kwargs["trf_file"] = config.get("tuflow", "trf_file")
    _kwargs["erf_file"] = config.get("tuflow", "erf_file")
    _kwargs["sim_duration"] = config.get("tuflow", "sim_duration")
    _kwargs["gauge_rainfall_file"] = config.get("tuflow", "gauge_rainfall_file")
    _kwargs["grid_results_dir"] = config.get("tuflow", "grid_results_dir")

    # lizard settings
    _kwargs["apikey"] = config.get("lizard", "apikey")
    _kwargs["precipitation_uuid_file"] = config.get("lizard", "precipitation_uuid_file")
    _kwargs["depth_raster_uuid"] = config.get("lizard", "depth_raster_uuid")

    # switches
    _kwargs["run_simulation"] = (
        config.get("switches", "run_simulation").lower() == "true"
    )
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
        "-rainfall", metavar="netcdf", default=None, help="netcdf rainfall file"
    )
    return parser


def main():
    """ Call command with args from parser. """
    kwargs = vars(get_parser().parse_args())
    logger = format_logger()
    settings = set_settings(**kwargs)
    logger.info("settings have been read")
    rainfall_gauges_uuids = read_rainfall_timeseries_uuids(settings)
    # create_new_rastersource(settings)
    starttime = datetime.datetime.now() - datetime.timedelta(days=5)
    endtime = (
        datetime.datetime.now()
        - datetime.timedelta(days=5)
        + datetime.timedelta(hours=int(settings["sim_duration"]))
    )

    rain_df = get_lizard_timeseries(settings, rainfall_gauges_uuids, starttime, endtime)
    logger.info("gathered lizard rainfall timeseries")
    rain_df = process_rainfall_timeseries_for_tuflow(rain_df)
    rain_df.to_csv(settings["gauge_rainfall_file"])
    logger.info("succesfully written rainfall file")
    #temp 
    sourcePath=Path(r"C:\Users\922383\Documents\TUFLOW\bc_dbase\RFG\rain.nc")
    start = datetime.datetime(2021, 3, 15, 5, 50)
    end = datetime.datetime(2021, 3, 15, 6, 25)
    #regular
    write_netcdf_with_time_indexes(sourcePath, start, end)
    logger.info("succesfully prepared netcdf rainfall")
    if settings['run_simulation']:
        logger.info('starting TUFLOW simulation')
        system_custom(
            [settings["tuflow_executable"], settings["tcf_file"],]
        )
        if settings["use_states"]:
            shutil.copyfile(settings["trf_file"], Path(settings["trf_file"]).name)
        if settings["erf_file"]:
            shutil.copyfile(settings["erf_file"], Path(settings["erf_file"]).name)
        logger.info('Tuflow simulation finished')
    convert_flt_to_tiff(settings)
    logger.info("Tuflow results converted to tiff")
    delete_former_lizard_results(settings)
    #create_new_rastersource(settings)
    post_results_to_lizard(settings)
    logger.info("Tuflow results posted to Lizard")
    return 1


if __name__ == "__main__":
    exit(main())
