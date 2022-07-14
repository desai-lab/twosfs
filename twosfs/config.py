"""Configuration class to hold simulation parameters."""
import json
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Dict, Iterator, List, Union


@dataclass
class Configuration:
    """Configuration class to hold simulation parameters."""

    simulation_directory: str
    initial_spectra_file: str = field(init=False)
    fitted_demography_file: str = field(init=False)
    # number of parallel msprime jobs
    nruns: int
    # number of resampling replicates for KS distance
    n_reps: int
    #  2 r * mean_coalescence_time
    scaled_recombination_rate: float
    msprime_parameters: Dict[str, Any]
    slim_parameters: Dict[str, Any]
    alphas: List[float]
    growth_rates: List[float]
    end_times: List[float]
    positive_sel_coeffs: List[float]
    positive_mut_rates: List[float]

    # Fitting parameters
    k_max: int
    num_epochs: int
    penalty_coef: float

    # Parameters for power calculations
    power_pair_densities: list[int]
    power_sequence_lengths: list[int]
    power_num_samples: int
    power_reps: int

    # Recombination search parameters
    search_r_low: float
    search_r_high: float
    search_iters: int
    search_num_replicates: int

    def __post_init__(self):
        """Initialize filename templates."""
        self.initial_spectra_file = (
            self.simulation_directory
            + "/initial_spectra/model={model}.params={params}.rep={rep}.hdf5"
        )
        self.fitted_demography_file = (
            self.simulation_directory
            + "/fitted_demographies/model={model}.params={params}.folded={folded}.txt"
        )
        self.recombination_search_file = (
            self.simulation_directory
            + "/recombination_search/model={model}.params={params}.folded={folded}."
            + "pair_density={pair_density}."
            + "sequence_length={sequence_length}."
            + "power_rep={power_rep}.hdf5"
        )
        self.fitted_spectra_file = (
            self.simulation_directory
            + "/fitted_spectra/model={model}.params={params}.folded={folded}."
            + "pair_density={pair_density}.sequence_length={sequence_length}."
            + "power_rep={power_rep}.hdf5"
        )
        self.tree_file = (
            self.simulation_directory
            + "/tree_files/model={model}.params={params}/model={model}.params={params}.rep={rep}.trees"
        )
        self.ks_distance_file = (
            self.simulation_directory
            + "/ks_distances/model={model}.params={params}."
            + "folded={folded}."
            + "pair_density={pair_density}."
            + "sequence_length={sequence_length}."
            + "power_rep={power_rep}.hdf5"
        )
        self.initial_ks_distance_file = (
            self.simulation_directory
            + "/initial_ks_distances/model={model}.params={params}."
            + "folded={folded}."
            + "pair_density={pair_density}."
            + "sequence_length={sequence_length}."
            + "power_rep={power_rep}.hdf5"
        )
        self.fitted_ks_distance_file = (
            self.simulation_directory
            + "/fitted_ks_distances/model={model}.params={params}."
            + "folded={folded}."
            + "pair_density={pair_density}."
            + "sequence_length={sequence_length}."
            + "power_rep={power_rep}.hdf5"
        )

    def iter_models(self) -> Iterator[tuple[str, dict]]:
        """Return an iterator over all model-parameter combinations."""
        yield "const", dict()
        for a in self.alphas:
            yield "beta", dict(alpha=a)
        for g in self.growth_rates:
            for t in self.end_times:
                yield "exp", dict(end_time=t, growth_rate=g)

    def iter_forward_models(self) -> Iterator[tuple[str, dict]]:
        """Return an iterator over all forward-time parameter combinations."""
        for mu in self.positive_mut_rates:
            for s in self.positive_sel_coeffs:
                for r in self.slim_parameters["rec_rates"]:
                    yield "sel", dict(s=s, rec_rate=r, mut_rate=mu)
        yield "sel", dict(s=0, rec_rate=1e-05, mut_rate = 1e-09)

    def format_initial_spectra_file(self, model: str, params: dict) -> str:
        """Get an initial spectra filename."""
        return self.initial_spectra_file.format(
            model=model, params=make_parameter_string(params), rep="all"
        )

    def format_tree_file(self, model: str, params: dict) -> str:
        """Get an initial tree filename."""
        return self.tree_file.format(
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

    def format_recombination_search_file(
        self,
        model: str,
        params: dict,
        folded: bool,
        pair_density: int,
        sequence_length: int,
        power_rep: int,
    ) -> str:
        """Get a recombination search filename."""
        return self.recombination_search_file.format(
            model=model,
            params=make_parameter_string(params),
            folded=folded,
            pair_density=pair_density,
            sequence_length=sequence_length,
            power_rep=power_rep,
        )

    def format_fitted_spectra_file(
        self,
        model: str,
        params: dict,
        folded: bool,
        pair_density: int,
        sequence_length: int,
        power_rep: int,
    ) -> str:
        """Get a fitted spectra filename."""
        return self.fitted_spectra_file.format(
            model=model,
            params=make_parameter_string(params),
            folded=folded,
            pair_density=pair_density,
            sequence_length=sequence_length,
            power_rep=power_rep,
        )

    def format_ks_distance_file(
        self,
        model: str,
        params: dict,
        folded: bool,
        pair_density: int,
        sequence_length: int,
        power_rep: int,
    ) -> str:
        return self.ks_distance_file.format(
            model=model,
            params=make_parameter_string(params),
            folded=folded,
            pair_density=pair_density,
            sequence_length=sequence_length,
            power_rep=power_rep,
        )

    def format_initial_ks_file(
        self,
        model: str,
        params: dict,
        folded: bool,
        pair_density: int,
        sequence_length: int,
        power_rep: int,
    ) -> str:
        return self.initial_ks_distance_file.format(
            model=model,
            params=make_parameter_string(params),
            folded=folded,
            pair_density=pair_density,
            sequence_length=sequence_length,
            power_rep=power_rep,
            rep="all"
        )

    def iter_demos(self) -> Iterator[tuple[str, dict, bool]]:
        """Return an iterator over all model-parameter-demography combinations."""
        for model, params in self.iter_models():
            # for folded in [True, False]:
            for folded in [True]:
                yield model, params, folded

    def iter_forward_demos(self) -> Iterator[tuple[str, dict, bool]]:
        """Return an iterator over all forward-time model-parameter-demography combinations."""
        for model, params in self.iter_forward_models():
            # for folded in [True, False]:
            for folded in [True]:
                yield model, params, folded

    def iter_rec_search(self) -> Iterator[tuple[str, dict, bool, int, int, int]]:
        """Return an iterator over all recombination search param combinations."""
        for model, params, folded in self.iter_demos():
            for density in self.power_pair_densities:
                for length in self.power_sequence_lengths:
                    for power_rep in range(self.power_reps):
                        yield model, params, folded, density, length, power_rep

    def iter_forward_rec_search(self) -> Iterator[tuple[str, dict, bool, int, int, int]]:
        """Return an iterator over all forward-time recombination search param combinations."""
        for model, params, folded in self.iter_forward_demos():
            for density in self.power_pair_densities:
                for length in self.power_sequence_lengths:
                    for power_rep in range(self.power_reps):
                        yield model, params, folded, density, length, power_rep

    def initial_spectra_files(self) -> Iterator[str]:
        """Iterate all initial spectra files."""
        return map(lambda x: self.format_initial_spectra_file(*x), self.iter_models())

    def fitted_demography_files(self) -> Iterator[str]:
        """Iterate all fitted demography files."""
        return map(lambda x: self.format_fitted_demography_file(*x), self.iter_demos())

    def recombination_search_files(self) -> Iterator[str]:
        """Iterate all recombination search files."""
        return map(
            lambda x: self.format_recombination_search_file(*x), self.iter_rec_search()
        )

    def fitted_spectra_files(self) -> Iterator[str]:
        "iterate all fitted spectra files."""
        return map(
            lambda x: self.format_fitted_spectra_file(*x), self.iter_rec_search()
        )

    def initial_forward_spectra_files(self) -> Iterator[str]:
        """Iterate all forward-time initial spectra files."""
        return map(lambda x: self.format_initial_spectra_file(*x), self.iter_forward_models())

    def fitted_forward_demography_files(self) -> Iterator[str]:
        """Iterate all forward-time fitted demography files."""
        return map(lambda x: self.format_fitted_demography_file(*x), self.iter_forward_demos())

    def fitted_forward_spectra_files(self) -> Iterator[str]:
        """Iterate all fitted spectra files."""
        return map(lambda x: self.format_fitted_spectra_file(*x), self.iter_forward_rec_search())

    def forward_recombination_search_files(self) -> Iterator[str]:
        """Iterate all forward-time recombination search files."""
        return map(
            lambda x: self.format_recombination_search_file(*x), self.iter_forward_rec_search()
        )
    def ks_distance_files(self) -> Iterator[str]:
        """Iterate all KS distance files"""
        return map(
            lambda x: self.format_ks_distance_file(*x), self.iter_rec_search()
        )

    def forward_ks_distance_files(self) -> Iterator[str]:
        """Iterate all KS distance files"""
        return map(
            lambda x: self.format_ks_distance_file(*x), self.iter_forward_rec_search()
        )


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
