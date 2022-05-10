import logging
import os
import shutil
import subprocess
from pathlib import Path

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
            if self.settings.prepare_state_for_next_run:
                self.copy_states()
            logger.info("Tuflow simulation finished")
        except (ValueError, IndexError):
            exit("Executable terminated, see log file")

    def copy_states(self):
        tcf_file = str(self.settings.tcf_file)
        trf_input_file = tcf_file.replace(".tcf", ".trf")
        trf_result_file = Path(
            os.path.join(self.settings.output_folder, trf_input_file)
        )

        if trf_result_file.exists():
            shutil.copyfile(trf_result_file, trf_input_file)
        else:
            raise MissingFileException("Source trf file %s not found", trf_result_file)

        erf_input_file = tcf_file.replace(".tcf", ".erf")
        erf_result_file = Path(
            os.path.join(self.settings.output_folder, erf_input_file)
        )

        if erf_result_file.exists():
            shutil.copyfile(erf_result_file, erf_input_file)
        else:
            raise MissingFileException("Source erf file %s not found", erf_result_file)
