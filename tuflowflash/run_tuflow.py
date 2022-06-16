import logging
import os
import shutil
import subprocess
from pathlib import Path
import glob
import datetime
logger = logging.getLogger(__name__)


class MissingFileException(Exception):
    pass


class TuflowSimulation:
    def __init__(self, settings):
        self.settings = settings
        self.run_command = [
            self.settings.tuflow_executable,
            "-b",
            "-x",
            "-pu2",
            str(self.settings.tcf_file),
        ]

    def run(self):
        try:
            if self.settings.manage_states:
                self.prepare_state()
            logger.info("starting TUFLOW simulation")
            if isinstance(self.run_command, list):
                cmd = " ".join(self.run_command)
            process = subprocess.Popen(
                cmd
            )  # , stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
            # )
            output, error = process.communicate()
            returncode = process.poll()
            logger.info(output)
            if returncode != 0:
                raise ValueError("Executable terminated, see log file")
            if self.settings.manage_states:
                if hasattr(self.settings,"export_states_folder"):
                    self.save_state()
            logger.info("Tuflow simulation finished")
        except (ValueError, IndexError):
            exit("Executable terminated, see log file")

    def prepare_state(self):
        search_dir=self.settings.initial_states_folder.stem
        states=list(filter(os.path.isfile, glob.glob(search_dir + "/warm_*.trf")))
        states.sort(key=lambda x: os.path.getmtime(x))
        if states:
            shutil.copyfile(states[-1], str(self.settings.tcf_file).replace(".tcf", ".trf"))
        else:
            shutil.copyfile(self.settings.initial_states_folder/"cold_state.trf", str(self.settings.tcf_file).replace(".tcf", ".trf"))

    def save_state(self):
        tcf_file = str(self.settings.tcf_file)
        trf_input_file = tcf_file.replace(".tcf", ".trf")
        trf_result_file = Path(
            os.path.join(self.settings.output_folder, trf_input_file)
        )

        if trf_result_file.exists():
            shutil.copyfile(trf_result_file, self.settings.export_states_folder/"warm_state_{}.trf".format(datetime.now().strftime("%Y%m%d%H")))
        else:
            raise MissingFileException("Source trf file %s not found", trf_result_file)

        erf_input_file = tcf_file.replace(".tcf", ".erf")
        erf_result_file = Path(
            os.path.join(self.settings.output_folder, erf_input_file)
        )

        if erf_result_file.exists():
            shutil.copyfile(erf_result_file, self.settings.export_states_folder/"warm_state_{}.erf".format(datetime.now().strftime("%Y%m%d%H")))
        else:
            logger.info("No erf file found")
