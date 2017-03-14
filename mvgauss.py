import numpy as np
import scipy.linalg
import pandas as pd
import attr.validators
from copy import deepcopy
from attr import attrs, attrib

import logging

logger = logging.getLogger(__name__)

def _to_dataframe_of_float(df_like):
    """
    Ensures float datatype and a deepcopy.
    """
    return pd.DataFrame(df_like,dtype=np.float_, copy=True)


@attrs(frozen=True)
class MvGauss:
    """
    Represents a multivariate Gaussian distribution.

    A basic data-structure for hanging together $\mu$,$\sigma$ and covariance
    parameters.


    Attributes:
        pars (pandas.Dataframe): Dataframe with ``index=('mu','sigma')``,
            containing one series for each variable.
        cov (pandas.Dataframe): Labelled covariance matrix.
        corr ((pandas.Dataframe): (Property) Labelled correlation matrix.
    """

    pars = attrib(convert=_to_dataframe_of_float)
    cov = attrib(convert=_to_dataframe_of_float)

    @classmethod
    def from_correlations(cls, pars, corr):
        """
        Instantiate from mu/sigma Dataframe and dictionary of correlations.

        The ``corr`` arg should contain pair-wise tuples mapping to
        correlation values (see examples in arg-listing).

        Inter-variable correlation is assumed 0 (i.e. independent variables)
        where not supplied. The covariance is calculated from the correlations
        at instantiation.

        Args:
            pars (pandas.Dataframe): Dataframe with ``index=('mu','sigma')``,
                containing one series for each variable. (Or nested-dict
                equivalent)
            corr (dict): e.g. ``{('a','b'): 0.3}``

        Returns:
            MVGauss
        """
        cov = build_covariance_matrix(pars.T.sigma, correlations=corr)

        return cls(pars, cov)

    @property
    def corr(self):
        """
        Fetch the correlation matrix

        (Calculated from the covariance matrix)

        Returns:
            pandas.DataFrame
        """
        corr = pd.DataFrame(self.cov, copy=True)
        sigmas = self.pars.T.sigma
        for var1 in self.cov:
            for var2 in self.cov:
                corr.loc[var1, var2] /= sigmas[var1] * sigmas[var2]
        return corr

    @property
    def mu(self):
        """
        Fetch the series of parameter means.

        Returns:
            pandas.Series

        """
        return self.pars.T.mu

    @property
    def sigma(self):
        """
        Fetch the series of parameter sigmas (std. dev.).

        Returns:
            pandas.Series

        """
        return self.pars.T.sigma

def build_covariance_matrix(sigmas, correlations):
    """
    Builds a covariance matrix from sigmas and correlations.

    Args:
        sigmas (dict): Mapping of parameter name to std. dev.
        correlations (dict): Mapping of tuples to correlations, e.g.
            ``{(param1,param2):0.5}``. Default correlation for unspecified
            pairs is zero.
    Returns:
        (pandas.DataFrame): Covariance matrix.
    """
    sigmas = pd.Series(sigmas)
    cov = pd.DataFrame(index=deepcopy(sigmas.index),
                       columns=deepcopy(sigmas.index),
                       data=np.diag(sigmas ** 2),
                       dtype=np.float
                       )
    cov.columns.name = 'ax0'
    cov.index.name = 'ax1'

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
