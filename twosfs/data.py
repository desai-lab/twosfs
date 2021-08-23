"""Helper functions for data analysis."""
from typing import IO, Iterable


def get_allele_counts_at_sites(
    allele_count_file: IO, sites: Iterable[int], cov_cutoff: int
) -> dict[int, int]:
    """
    Read allele counts at a list of sites.

    Filter out sites without at least cov_cutoff alleles genotyped.
    """
    ac_dict = {}
    i_line = 0
    line = allele_count_file.readline()
    for pos in sites:
        while i_line < pos:
            line = allele_count_file.readline()
            i_line += 1
        nobs, mac = map(int, line.split())
        if nobs >= cov_cutoff and mac > 0:
            ac_dict[pos] = mac
    return ac_dict
