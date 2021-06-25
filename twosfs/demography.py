"""Defines a class of demographic models for translating and scaling."""

from msprime import Demography


def make_exp_demography(
    growth_rate: float, end_time: float, initial_size=1.0
) -> Demography:
    """Make an msprime demography with exp growth extending to end_time in the past."""
    demography = Demography()
    demography.add_population(initial_size=initial_size, growth_rate=growth_rate)
    demography.add_population_parameters_change(time=end_time, growth_rate=0)
    return demography


def make_pwc_demography(sizes: list[float], times: list[float]) -> Demography:
    """Make an msprime demography with piecewise constant population size."""
    demography = Demography()
    demography.add_population(initial_size=sizes[0])
    for size, time in zip(sizes[1:], times):
        demography.add_population_parameters_change(time=time, initial_size=size)
    return demography


def expected_t2_demography(demography: Demography) -> float:
    """Compute the expected pairwise coalescence time for a demography."""
    pop = demography.populations[0].name
    debugger = demography.debug()
    return debugger.mean_coalescence_time(lineages={pop: 2}, min_pop_size=0)
