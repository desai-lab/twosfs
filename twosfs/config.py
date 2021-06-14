"""Configuration class to hold simulation parameters."""
import json
from dataclasses import dataclass
from os import PathLike
from typing import Any, Iterator, Union


@dataclass
class Configuration:
    """Configuration class to hold simulation parameters."""

    initial_spectra_file: str
    temp_sfs_file: str
    fitted_demography_file: str
    fitted_spectra_file: str
    demographic_model_file: str
    # number of parallel msprime jobs
    nruns: int
    #  2 r * mean_coalescence_time
    scaled_recombination_rate: float
    msprime_parameters: dict[str, Any]
    alphas: list[float]
    growth_rates: list[float]
    end_times: list[float]
    rec_factors: list[float]
    fitted_demographies: list[str]
    fastNeutrino_maxB: int
    fastNeutrino_maxRandomRestarts: int

    def iter_models(self) -> Iterator[tuple[str, dict]]:
        """Return an iterator over all model-parameter combinations."""
        yield "const", dict()
        for a in self.alphas:
            yield "beta", dict(alpha=a)
        for g in self.growth_rates:
            for t in self.end_times:
                yield "exp", dict(end_time=t, growth_rate=g)

    def format_initial_spectra_file(self, model: str, params: dict) -> str:
        """Get an initial spectra filename."""
        return self.initial_spectra_file.format(
            model=model, params=make_parameter_string(params), rep="all"
        )

    def format_fitted_demography_file(self, model: str, params: dict, demo: str) -> str:
        """Get a fitted demography filename."""
        return self.fitted_demography_file.format(
            model=model, params=make_parameter_string(params), demo=demo
        )

    def format_fitted_spectra_file(
        self, model: str, params: dict, demo: str, rec_factor: float
    ) -> str:
        """Get a fitted spectra filename."""
        return self.fitted_spectra_file.format(
            model=model,
            params=make_parameter_string(params),
            demo=demo,
            rec_factor=rec_factor,
            rep="all",
        )

    def iter_demos(self) -> Iterator[tuple[str, dict, str]]:
        """Return an iterator over all model-parameter-demography combinations."""
        for model, params in self.iter_models():
            for demo in self.fitted_demographies:
                yield model, params, demo

    def iter_all(self) -> Iterator[tuple[str, dict, str, float]]:
        """Return an iterator over all model-parameter-demography-rec combinations."""
        for model, params, demo in self.iter_demos():
            for rec_factor in self.rec_factors:
                yield model, params, demo, rec_factor

    def initial_spectra_files(self) -> Iterator[str]:
        """Iterate all initial spectra files."""
        return map(lambda x: self.format_initial_spectra_file(*x), self.iter_models())

    def fitted_demography_files(self) -> Iterator[str]:
        """Iterate all fitted demography files."""
        return map(lambda x: self.format_fitted_demography_file(*x), self.iter_demos())

    def fitted_spectra_files(self) -> Iterator[str]:
        """Iterate all fitted spectra files."""
        return map(lambda x: self.format_fitted_spectra_file(*x), self.iter_all())


def configuration_from_json(config_file: Union[str, bytes, PathLike]):
    """Read a configuration from a json file."""
    with open(config_file) as f:
        data = json.load(f)
    return Configuration(**data)


def make_parameter_string(params: dict) -> str:
    """Convert parameter dictionary to json string without whitespace."""
    return json.dumps(params, separators=(",", ":"))


def parse_parameter_string(parameter_string: str) -> dict:
    """Parse parameter dictionary from json string."""
    try:
        return json.loads(parameter_string)
    except json.JSONDecodeError:
        raise ValueError(f"Parameter string '{parameter_string}' is not valid JSON")
