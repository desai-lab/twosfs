"""Configuration class to hold simulation parameters."""
import json
from dataclasses import dataclass
from os import PathLike
from typing import Any, Union


@dataclass
class Configuration:
    """Configuration class to hold simulation parameters."""

    # number of parallel msprime jobs
    nruns: int
    #  2 r * mean_coalescence_time
    scaled_recombination_rate: float
    msprime_parameters: dict[str, Any]
    alphas: list[float]
    growth_rates: list[float]
    end_times: list[float]
    fitted_demographies: list[str]
    fastNeutrino_maxB: int
    fastNeutrino_maxRandomRestarts: int


def configuration_from_json(config_file: Union[str, bytes, PathLike]):
    """Read a configuration from a json file."""
    with open(config_file) as f:
        data = json.load(f)
    return Configuration(**data)
