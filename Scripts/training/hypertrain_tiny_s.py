"""
Hyper parameter analysis
"""

import sys
from os import path
import logging, coloredlogs
import ruamel.yaml as yaml
from jinja2 import Environment, FileSystemLoader
from numpy import log
from hyperopt import hp, Trials, fmin, tpe
from training_s import run

logger = logging.getLogger("hyper_train")
logger.addHandler(logging.StreamHandler())

coloredlogs.install(
    logger=logger,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.DEBUG,
)


def objective(params):
    envPath = path.abspath(templatesDir)
    env = Environment(loader=FileSystemLoader(envPath))
    template = env.get_template(paramsFile)
    yamlStr = template.render(params)
    fullParams = yaml.safe_load(yamlStr)
    fullParams["paramsFile"] = None
    fullParams["device"] = device
    fullParams["subDeviceIdx"] = deviceIdx

    for unitGroupIdx, group in enumerate(fullParams["units"]):
        for convIdx, conv in enumerate(group["coder"]):
            fullParams["units"][unitGroupIdx]["coder"][convIdx]["conv"][
                "out_channels"
            ] = int(
                fullParams["units"][unitGroupIdx]["coder"][convIdx]["conv"][
                    "out_channels"
                ]
            )

    logger.debug(f"Running using params:\n{fullParams}")

    return -1 * run(fullParams)


if __name__ == "__main__":
    templatesDir = sys.argv[1]
    paramsFile = sys.argv[2]
    device = sys.argv[3]
    deviceIdx = sys.argv[4]

    # used for loguniform
    c = log(10.0)

    space = {
        "basisSize": hp.uniformint("basisSize", 3, 10),
        "embeddingSize": hp.uniformint("embeddingSize", 500, 5000),
        "l1_residuCoeff": hp.uniform("l1_residuCoeff", 0.1, 10),
    }

    trials = Trials()

    best = fmin(
        objective, space, algo=tpe.suggest, catch_eval_exceptions=True, max_evals=200
    )
    logger.info(f"Best configuration:\n{best}")
    logger.debug(f"Trials:\n{trials}")
