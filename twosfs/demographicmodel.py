import functools
import sys

import numpy as np

from msprime import PopulationParametersChange, simulate
from twosfs.twosfs import sims2pi


class DemographicModel:
    """Stores piecewise-exponential demographic models."""
    def __init__(self, filename=None):
        # Number of epochs. Must equal the lengths of the following lists:
        self.num_epochs = 0
        # Epoch start times
        self.times = []
        # Epoch starting (i.e. most recent) sizes
        self.sizes = []
        # Epoch growth rates (forward in time)
        self.rates = []

        # If a fastNeutrino output file is specified, read it.
        if filename is not None:
            self.read_fastNeutrino_output(filename)

    def read_fastNeutrino_output(self, model_fn):
        """Read epochs from a fastNeutrino fitted parameters output file."""
        with open(model_fn) as modelfile:
            # Discard the header
            _ = modelfile.readline()
            n_anc = float(modelfile.readline())
            # First epoch implicitly starts at t=0
            start_time = 0.0
            for line in modelfile:
                # Get epoch parameters.
                if line.startswith('c'):
                    # Constant-N epoch
                    n, t = map(float, line.split()[-2:])
                    g = None
                elif line.startswith('e'):
                    # Exponential-growth epoch
                    n, t, g = map(float, line.split()[-3:])
                else:
                    raise ValueError('Warning, bad line: ' + line.strip())
                    break

                # Add epoch to model.
                # TODO: handle time order errors.
                self.add_epoch(start_time, n, g)
                # Set next epoch start time to current epoch end time
                start_time = t
        # Add ancestral population size as the last epoch
        self.add_epoch(start_time, n_anc, None)

    def add_epoch(self, time, size, rate=None):
        """Add new epoch to the demographic model."""
        if self.num_epochs == 0 or time >= self.times[-1]:
            self.num_epochs += 1
            self.times.append(time)
            self.sizes.append(size)
            self.rates.append(rate)
        else:
            raise ValueError(
                'New epoch time is less than previous epoch time.')

    def population_size(self, T):
        """Return the population size at time T."""
        if self.num_epochs == 0:
            # No epochs have been added.
            return np.empty_like(T)
        else:
            # Evalute piecewise exponential growth during each epoch.
            conditions = [T >= t0 for t0 in self.times]
            functions = [
                functools.partial(exponential_growth, *args)
                for args in zip(self.sizes, self.times, self.rates)
            ]
            return np.piecewise(T, conditions, functions)

    def rescale(self, scale=None):
        """Renormalize model parameters by a timescale (Default=N_0)."""
        if scale is None:
            # By default scale so that T2=4
            scale = self.t2() / 4
        self.sizes = [s / scale for s in self.sizes]
        self.times = [t / scale for t in self.times]
        self.rates = [None if r is None else r * scale for r in self.rates]
        return scale

    # TODO: Remove
    def print_msprime_flags(self):
        """DEPRECATED Print model params as flags simulate_joint_sfs.py."""
        # WARNING: only works for constant-time epochs (for now)
        flags = '-T ' + ' '.join(map(str, self.times))
        flags += ' -S ' + ' '.join(map(str, self.sizes))
        sys.stdout.write(flags)

    def get_demographic_events(self):
        """Get a list of demographic_events for msprime simulations."""
        events = []
        for t, s, g in zip(self.times, self.sizes, self.rates):
            events.append(
                PopulationParametersChange(t, initial_size=s, growth_rate=g))
        return events

    def t2(self):
        """
        Compute the average branch length for a pair of samples.

        Warning: Only works for constant-size epochs!
        Note: Proportional to pi.
        """
        sizes = np.array(self.sizes)
        times = np.array(self.times)
        # The lengths of the intervals, scaled by sizes
        scaled_intervals = np.empty(self.num_epochs)
        scaled_intervals[:-1] = (times[1:] - times[:-1]) / sizes[:-1]
        scaled_intervals[-1] = np.inf
        # Weights for the contribution of each size to T2
        weights = np.ones(self.num_epochs)
        weights[1:] = np.exp(-np.cumsum(scaled_intervals[:-1]))
        weights *= -np.expm1(-scaled_intervals)
        return 4 * np.sum(sizes * weights)


def scaled_demographic_events(filename):
    """
    Return a list of msprime demographic events scaled so that T2=4.

    Parameters
    ----------
    filename : str
        The name of a fastNeutrino fitted model output file.
    """
    dm = DemographicModel(filename)
    dm.rescale()
    return dm.get_demographic_events()


def exponential_growth(n0, t0, r, T):
    """Calculate N(t) for exponentially-growing population back in time."""
    if r is None:
        return n0
    else:
        return n0 * np.exp(-(T - t0) * r)
