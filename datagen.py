import numpy as np
import scipy
import xarray as xr

import logging

logger = logging.getLogger(__name__)


def build_covariance_matrix(sigmas, correlations):
    """
    Builds a covariance matrix from sigmas and correlations.

    Args:
        sigmas (xarray.DataArray): Mapping of parameter name to std. dev.
        correlations (dict): Mapping of tuples to correlations, e.g.
            ``{(param1,param2):0.5}``. Default correlation for unspecified
            pairs is zero.
    Returns:
        (DataFrame): Covariance matrix.
    """
    cov = pd.DataFrame(index=sigmas.index,
                       columns=sigmas.index,
                       data=np.diag(sigmas ** 2),
                       dtype=np.float
                       )
    for param_pair, pair_corr in correlations.items():
        p1, p2 = param_pair
        pair_cov = pair_corr * sigmas[p1] * sigmas[p2]
        cov.loc[p1, p2] = cov.loc[p2, p1] = pair_cov

    try:
        scipy.linalg.cholesky(cov,
                              lower=True)
    except np.linalg.LinAlgError:
        logger.error("Provided correlation values result in a covariance"
                     " which is not positive semidefinite.")
        raise

    return cov
