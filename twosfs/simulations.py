import numpy as np
from scipy.special import betaln


def beta_timescale(alpha, pop_size=1.0):
    """The timescale of the beta coalescent."""
    m = 2 + np.exp(alpha * np.log(2) +
                   (1 - alpha) * np.log(3) - np.log(alpha - 1))
    N = pop_size / 2
    # The initial 2 is so that the rescaling by beta_timescale
    # gives T_2 = 4
    ret = 2 * np.exp(alpha * np.log(m) + (alpha - 1) * np.log(N) -
                     np.log(alpha) - betaln(2 - alpha, alpha))
    return ret
