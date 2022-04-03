"""
Helper functions for dealing with external smore training command.
"""
import subprocess
import tempfile
import logging
from typing import Dict, List, Union
from pathlib import Path

from pcc.schemas.common import Model, Command

log = logging.getLogger(__name__)
SMORE_CMD_DIR = Path(__file__).parent.parent.parent.parent.parent / "smore/cli"


def run(cmd: str) -> Command:
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    return_code = process.returncode
    return Command(stdout.decode(), stderr.decode(), stdout, stderr, return_code)


def run_smore_command(
    model_name: str, parameters: Dict[str, str], file_path: Union[str, Path]
) -> Model:
    """
    Run smore internal model according to the given model_name and the parameters
    """
    with tempfile.NamedTemporaryFile() as temp_file:
        output_file = temp_file.name
        file_path = Path(file_path)
        parameters_as_string = " ".join(
            [f"-{key} {value}" for key, value in parameters.items()]
        )
        command = f"{SMORE_CMD_DIR}/{model_name} -train {file_path} -save {output_file} {parameters_as_string}"
        log.info(f"Execute SMORe Command {command}")
        result = run(command)
        if result.return_code != 0:
            raise RuntimeError(str(result.stderr))
        index_to_embedding: Dict[str, List[float]] = {}
        with open(output_file, "rt") as fin:
            fin.readline()
            for line in fin:
                idx, *emb = line.strip().split()
                index_to_embedding[idx] = [float(val) for val in emb]

    return Model(
        index_to_embedding=index_to_embedding,
        model_name=model_name,
        embedding_size=int(parameters["dimensions"]),
    )
