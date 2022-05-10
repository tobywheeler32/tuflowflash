import argparse
import datetime
import logging

from tuflowflash import post_processing
from tuflowflash import prepare_data
from tuflowflash import read_settings
from tuflowflash import run_tuflow
from pathlib import Path

logger = logging.getLogger(__name__)

OWN_EXCEPTIONS = (
    read_settings.MissingFileException,
    read_settings.MissingSettingException,
)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    # POSITIONAL ARGUMENTS

    parser.add_argument(
        "-s",
        "--settings",
        dest="settings_file",
        default="settings.ini",
        help=".ini settings file",
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
    return parser


def main():
    """ Call command with args from parser. """
    ## read settings
    options = get_parser().parse_args()
    if options.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    settings = read_settings.FlashSettings(options.settings_file)

    try:
        # Historical precipitation
        data_prepper = prepare_data.prepareData(settings)
        if settings.get_historical_precipitation:
            data_prepper.get_historical_precipitation()
        else:
            logger.info("not gathering historical rainfall, skipping..")

        # Future precipitation
        if settings.get_future_precipitation:
            data_prepper.get_future_precipitation()
            data_prepper.NC_to_asci(Path(r"..\bc_dbase\RFG"))
        else:
            logger.info("not gathering future rainfall data, skipping..")

        # run simulation
        if settings.run_simulation:
            tuflow_simulation = run_tuflow.TuflowSimulation(settings)
            tuflow_simulation.run()
        else:
            logger.info("Not running Tuflow simulation, skipping..")

        # uploading to Lizard
        post_processer = post_processing.ProcessFlash(settings)
        if settings.post_to_lizard:
            post_processer.process_tuflow()
            post_processer.process_bom()
        else:
            logger.info("Not uploading files to Lizard, skipping..")

        if settings.archive_simulation:
            post_processer.archive_simulation()
        else:
            logger.info("Not archiving files, skipping..")
        return 0

    except OWN_EXCEPTIONS as e:
        if options.verbose:
            logger.exception(e)
        else:
            logger.error("↓↓↓↓↓   Pass --verbose to get more information   ↓↓↓↓↓")
            logger.error(e)
        return 1  # Exit code signalling an error.

