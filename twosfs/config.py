"""Configuration class to hold simulation parameters."""
import json
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Iterator, Union


@dataclass
class Configuration:
    """Configuration class to hold simulation parameters."""

    simulation_directory: str
    initial_spectra_file: str = field(init=False)
    fitted_demography_file: str = field(init=False)
    fitted_spectra_file: str = field(init=False)
    # number of parallel msprime jobs
    nruns: int
    #  2 r * mean_coalescence_time
    scaled_recombination_rate: float
    msprime_parameters: dict[str, Any]
    alphas: list[float]
    growth_rates: list[float]
    end_times: list[float]
    rec_factors: list[float]
    # Fitting parameters
    k_max: int
    num_epochs: int

    def __post_init__(self):
        """Initialize filename templates."""
        self.initial_spectra_file = (
            self.simulation_directory
            + "/initial_spectra/model={model}.params={params}.rep={rep}.npz"
        )
        self.fitted_demography_file = (
            self.simulation_directory
            + "/fitted_demographies/model={model}.params={params}.folded={folded}.txt"
        )
        self.fitted_spectra_file = (
            self.simulation_directory
            + "/fitted_spectra/model={model}.params={params}.folded={folded}"
            + ".rec_factor={rec_factor}.rep={rep}.npz"
        )

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

    def format_fitted_demography_file(
        self, model: str, params: dict, folded: bool
    ) -> str:
        """Get a fitted demography filename."""
        return self.fitted_demography_file.format(
            model=model,
            params=make_parameter_string(params),
            folded=folded,
        )

    def format_fitted_spectra_file(
        self, model: str, params: dict, folded: bool, rec_factor: float
    ) -> str:
        """Get a fitted spectra filename."""
        return self.fitted_spectra_file.format(
            model=model,
            params=make_parameter_string(params),
            folded=folded,
            rec_factor=rec_factor,
            rep="all",
        )

    def iter_demos(self) -> Iterator[tuple[str, dict, bool]]:
        """Return an iterator over all model-parameter-demography combinations."""
        for model, params in self.iter_models():
            for folded in [True, False]:
                yield model, params, folded

    def iter_all(self) -> Iterator[tuple[str, dict, bool, float]]:
        """Return an iterator over all model-parameter-demography-rec combinations."""
        for model, params, folded in self.iter_demos():
            for rec_factor in self.rec_factors:
                yield model, params, folded, rec_factor

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
