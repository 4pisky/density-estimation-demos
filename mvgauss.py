import logging

import numpy as np
import pandas as pd
import scipy.linalg
from attr import attrib, attrs
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


def _to_dataframe_of_float(df_like):
    """
    Ensures float datatype and a deepcopy.
    """
    return pd.DataFrame(df_like, dtype=np.float_, copy=True)


@attrs(frozen=True)
class MvGauss(object):
    """
    Represents a multivariate Gaussian distribution.

    A basic data-structure for hanging together $\mu$,$\sigma$ and covariance
    parameters.


    Attributes:
        pars (pandas.Dataframe): Dataframe with ``index=('mu','sigma')``,
            containing one series for each variable.
        cov (pandas.Dataframe): Labelled covariance matrix.
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
    #
    # @classmethod
    # def from_mean_and_covariance(cls, mean, covariance):
    #     """
    #     Instantiate from mean co-ordinates and covariance matrix
    #
    #     Args:
    #         mean (iterable): Iterable containing the co-ordinates of the mean
    #         covariance (dict): e.g. ``{('a','b'): 0.3}``
    #
    #     Returns:
    #         MVGauss
    #     """

        # return cls(pars, cov)

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
    def dist(self):
        """
        Fetch a scipy.stats.multivariate_normal distribution

        (initialised to repreresent this Gaussian)
        """
        return multivariate_normal(mean=self.pars.T.mu, cov=self.cov)

    @property
    def eig(self):
        """
        Fetch the eigenvalues and eigenvectors of the covariance matrix
        
        The eigenvalues are equivalent to the diagonal values of the 
        covariance matrix in the frame with co-ordinate axis aligned to 
        the orientation of the multivariate distribution.
        
        Returns:
            tuple: eigenvalues (pandas.Series), eigenvectors (pandas.DataFrame)
        """
        e_val, e_vec = np.linalg.eigh(self.cov)

        e_val = pd.Series(e_val, name="$\lambda$")
        e_mat = pd.DataFrame(data=e_vec,
                             index=self.cov.index.copy()
                             )
        return e_val, e_mat

    @property
    def pcv(self):
        """
        Fetch the principle component vectors of the Gaussian.

        Returns:
            pandas.Dataframe
        """
        e_val, e_mat = self.eig
        # e_mat are the unit-vectors.
        # Multiply them by the std-dev in each direction to get a lengthscale
        pca = e_mat.multiply(e_val ** 0.5)
        return pca

    @property
    def mu(self):
        """
        Fetch the series of parameter means.

        Returns:
            pandas.Series

        """
        return self.pars.T.mu

    @property
    def ndim(self):
        return len(self.mu)

    @property
    def sigma(self):
        """
        Fetch the series of parameter sigmas (std. dev.).

        Returns:
            pandas.Series

        """
        return self.pars.T.sigma

    def _repr_html_(self):

        caption_pars = self.pars.T.style.set_caption('Parameters')
        caption_cov = self.cov.style.set_caption('Covariance')
        caption_corr = self.corr.style.set_caption('Correlation')
        pcv = pd.DataFrame(self.pcv, copy=True)
        # pcv=pcv.T
        # pcv.insert(0,'$\sigma$', self.eig[0].values**0.5)
        # pcv=pcv.T
        pcv.loc['$\sigma$'] = self.eig[0] ** 0.5
        caption_pcv = pcv.style.set_caption('PC-vecs')

        return """
            <style> div.output_area .rendered_html table {{float:left; margin-right:10px; }}</style>
            {pars}
            {cov}
            {corr}
            {pcv}
            """.format(
            pars=caption_pars.render(),
            cov=caption_cov.render(),
            corr=caption_corr.render(),
            pcv=caption_pcv.render(),
        )


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
    cov = pd.DataFrame(index=sigmas.index.copy(),
                       columns=sigmas.index.copy(),
                       data=np.diag(sigmas ** 2),
                       dtype=np.float
                       )
    # cov.columns.name = 'ax0'
    # cov.index.name = 'ax1'

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
