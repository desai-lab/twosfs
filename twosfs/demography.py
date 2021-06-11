"""Defines a class of demographic models for translating and scaling."""
from dataclasses import dataclass

from msprime import Demography


def make_exp_demography(growth_rate: float, end_time: float, initial_size=1.0):
    """Make an msprime demography with exp growth extending to end_time in the past."""
    demography = Demography()
    demography.add_population(initial_size=initial_size, growth_rate=growth_rate)
    demography.add_population_parameters_change(time=end_time, growth_rate=0)
    return demography


def make_pwc_demography(sizes: list[float], start_times: list[float], initial_size=1.0):
    """Make an msprime demography with piecewise constant population size."""
    demography = Demography()
    demography.add_population(initial_size=initial_size)
    for size, time in zip(sizes, start_times):
        demography.add_population_parameters_change(time=time, initial_size=size)
    return demography


def expected_t2_demography(demography: Demography) -> float:
    """Compute the expected pairwise coalescence time for a demography."""
    pop = demography.populations[0].name
    debugger = demography.debug()
    return debugger.mean_coalescence_time(lineages={pop: 2}, min_pop_size=0)


@dataclass
class FastNeutrinoEpoch:
    """Epochs in the fastNeutrino demographic model."""

    size: float
    end_time: float


def _parse_line(line: str) -> FastNeutrinoEpoch:
    if line.startswith("c"):
        # Constant-N epoch
        n, t = map(float, line.split()[-2:])
    else:
        raise ValueError("Warning, bad line: " + line.strip())
    return FastNeutrinoEpoch(size=n, end_time=t)


def read_fastNeutrino_output(model_fn) -> Demography:
    """Read epochs from a fastNeutrino fitted parameters output file."""
    with open(model_fn) as modelfile:
        # Discard the header
        _ = modelfile.readline()
        n_anc = float(modelfile.readline())
        epochs = [_parse_line(line) for line in modelfile]
    # Rescale sizes because fastNeutrino and msprime use different models
    # Translate start times to end times
    initial_size = epochs[0].size / 2
    sizes = [epoch.size / 2 for epoch in epochs[1:]] + [n_anc / 2]
    start_times = [epoch.end_time for epoch in epochs]
    return make_pwc_demography(sizes, start_times, initial_size=initial_size)
