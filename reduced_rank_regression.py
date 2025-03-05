from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass
class RRR(BaseEstimator):
    """
    Reduced Rank Regression (RRR) estimator.

    This class implements the Reduced Rank Regression as described in
    Bunea et al., 2011, Annals of Statistics.

    Parameters
    ----------
    alpha : float, default=1.0
        Threshold for eigenvalues used to determine the optimal rank
        in the regression.

    Attributes
    ----------
    name : str
        Name of the estimator.
    X_mean_ : ndarray of shape (n_targets,)
        Mean of the target matrix X during fitting.
    Z_mean_ : ndarray of shape (n_features,)
        Mean of the input matrix Z during fitting.
    M_hat_ : ndarray of shape (n_targets, n_features)
        Estimated reduced-rank regression coefficients.
    k_opt_ : int
        Optimal rank chosen during fitting.
    M_ols_ : ndarray of shape (n_targets, n_features)
        Regression coefficients obtained from ordinary least squares.
    is_fitted_ : bool
        Flag indicating if the estimator has been fitted.

    """

    alpha: float = 1.0

    def __post_init__(self):
        self.name = "reduced_rank_regression"

    def fit(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        seed=None,
        check_input=True,
    ) -> None:
        """
        Fit the model to input matrix Z and target matrix X.

        Parameters
        ----------
        Z : ndarray of shape (n_samples, n_features)
            Input matrix.
        X : ndarray of shape (n_samples, n_targets)
            Target matrix.
        seed : int, default=None
            Seed for random number generator (if required).
        check_input : bool, default=True
            If True, input matrices will be validated.

        Returns
        -------
        self : object
            Returns self.
        """

        if check_input:
            # Validate matrices and ensure X is multi-output
            Z, X = check_X_y(Z, X, multi_output=True)

        # Center variables
        self.X_mean_ = np.mean(X, axis=0)
        self.Z_mean_ = np.mean(Z, axis=0)

        Z_centered = Z - self.Z_mean_
        X_centered = X - self.X_mean_

        # Compute RRR
        M_ols = np.linalg.lstsq(a=Z_centered, b=X_centered, rcond=None)[0].T

        XPzX = M_ols @ Z_centered.T @ X_centered
        V, L, Vt = la.svd(XPzX)

        k_opt = np.min([X.shape[1], Z.shape[1], np.sum(L > self.alpha)])

        A_k = V[:, 0:k_opt]
        B_k = V[:, 0:k_opt].T @ M_ols
        self.M_hat_ = A_k @ B_k
        self.k_opt_ = k_opt
        self.M_ols_ = M_ols

        self.is_fitted_ = True

    def predict(self, Z: np.ndarray, check_input=True) -> np.ndarray:
        """
        Predict using the fitted Reduced Rank Regression model.

        Parameters
        ----------
        Z : ndarray of shape (n_samples, n_features)
            Input matrix for which predictions are to be made.
        check_input : bool, default=True
            If True, input matrix Z will be validated.

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_targets)
            Predicted values.
        """

        # Check if the estimator has been fitted
        check_is_fitted(self, "is_fitted_")

        if check_input:
            Z = check_array(Z, accept_sparse=True)

        Z_centered = Z - self.Z_mean_
        prediction = Z_centered @ self.M_hat_.T

        # Adjust prediction for the mean of X
        return prediction + self.X_mean_


@dataclass
class RRR_old(BaseEstimator):
    """Implementation of Reduced Rank Regression as in Bunea et al., 2011, Annals of Statistics"""

    alpha: float = 1.0

    def __post_init__(self):
        self.name = "reduced_rank_regression"

    def fit(
        self,
        Z: np.ndarray,
        X: np.ndarray,
        seed=None,
        check_input=True,
    ) -> None:
        if check_input:
            # perform checks
            Z, X = check_X_y(Z, X, multi_output=True)

        # get dimensions
        n, p = X.shape
        n, r = Z.shape

        # compute RRR
        M_ols = np.linalg.lstsq(a=Z, b=X, rcond=None)[0].T

        XPzX = M_ols @ Z.T @ X
        V, L, Vt = la.svd(XPzX)  # type: ignore

        k_opt = np.min([p, r, np.sum(L > self.alpha)])

        A_k = V[:, 0:k_opt]
        B_k = V[:, 0:k_opt].T @ M_ols
        self.M_hat_ = A_k @ B_k
        self.k_opt_ = k_opt
        self.M_ols_ = M_ols

        self.is_fitted_ = True

    def predict(self, Z: np.ndarray, check_input=True) -> np.ndarray:
        # check whether is fitted
        check_is_fitted(self, "is_fitted_")  # type: ignore

        if check_input:
            # perform checks on Z
            Z = check_array(Z, accept_sparse=True)

        return Z @ self.M_hat_.T