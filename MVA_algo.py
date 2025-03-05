import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
import scipy


def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = np.random.randn(num, dimX)
    W = np.dot(np.random.randn(dimX, rrank), np.random.randn(rrank, dimY))
    Y = np.dot(X, W) + np.random.randn(num, dimY) * noise
    return X, Y


class ReducedRankRegressor(BaseEstimator):
    """
    kernel Reduced Rank Ridge Regression
    - X is an n-by-P matrix of features (n-time points).
    - Y is an n-by-Q matrix of targets (n-time points).
    - rank is a rank constraint.
    - reg is a regularization parameter.
    """
    
    def __init__(self, rank = None, reg = 0.1, P_rr = None, Q_fr = None, trainX = None):
        self.rank   = rank
        self.reg    = reg
        self.P_rr   = P_rr
        self.Q_fr   = Q_fr        
        self.trainX = trainX
        

    def __str__(self):
        return 'kernel Reduced Rank Ridge Regression by Mukherjee (rank = {})'.format(self.rank)


    def fit(self, X, Y):
        # use try/except blog with exceptions!
        self.rank   = int(self.rank)
        
        K_X         = np.dot(X, X.T)
        tmp_1       = self.reg * np.identity(K_X.shape[0]) + K_X 
        Q_fr        = np.linalg.solve(tmp_1, Y)
        P_fr        = scipy.linalg.eig(np.dot(Y.T, np.dot(K_X, Q_fr)))[1].real
        P_rr        = np.dot(P_fr[:,0:self.rank],P_fr[:,0:self.rank].T)
        
        self.Q_fr   = Q_fr
        self.P_rr   = P_rr
        self.trainX = X
        return self

       
    def predict(self, testX):
        # use try/except blog with exceptions!
        
        K_Xx        = np.dot(testX, self.trainX.T)
        Yhat        = np.dot(K_Xx,np.dot(self.Q_fr,self.P_rr))
        
        return Yhat

    
    def rrr_scorer(self, Yhat, Ytest):
        diag_corr   = (np.diag(np.corrcoef(Ytest,Yhat))).mean()
        return diag_corr
    
    
    
    

    
    


class OPLS:
    def __init__(self, solver='GEV', ncp=5, alpha=0):
        """
        Initializes the OPLS model.

        Parameters:
            solver (str): The solver to use for coefficient estimation ('GEV' or 'ElasticNet').
            ncp (int): Number of components to keep. If None, all components are kept.
            alpha (float): Regularization parameter for ElasticNet. If 0, Linear Regression is used.
        """
        self.solver = solver
        self.alpha = alpha
        self.ncp = ncp
        self.coef_ = None
        self.weight_ = None
        self.eig_ = None

    def transform(self, X):
        """
        Transform the input data X using the learned weight matrix.

        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            X_transformed (array-like): Transformed data of shape (n_samples, n_components).
        """
        X_transformed = X @ self.weight_
        return X_transformed

    def estimate_coef(self, X_transform, Y):
        """
        Estimate coefficients using either Linear Regression or ElasticNet.

        Parameters:
            X_transform (array-like): Transformed input data of shape (n_samples, n_components).
            Y (array-like): Target data of shape (n_samples,).

        Returns:
            None
        """
        if self.alpha != 0:
            self.coef_ = ElasticNet(alpha=self.alpha).fit(X_transform, Y).coef_.T
        else:
            self.coef_ = LinearRegression().fit(X_transform, Y).coef_.T
        if self.ncp is not None:
            self.coef_ = self.coef_[:self.ncp, :]

    def estimate_weight(self, X, Y):
        """
        Estimate the weight matrix using the given input and target data.

        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).
            Y (array-like): Target data of shape (n_samples,).

        Returns:
            None
        """
        C_tilde = X.T @ Y @ self.coef_.T
        C_XX_inv = np.linalg.pinv(X.T @ X)
        u, v = np.linalg.eig(C_XX_inv @ C_tilde @ C_tilde.T)
        u, v = u.real, v.real
        idx = np.argsort(u)[::-1]
        self.eig_ = u[idx]
        self.weight_ = v[:, idx]
        if self.ncp is not None:
            self.weight_ = self.weight_[:, :self.ncp]

    def fit(self, X, Y, max_iter=500, delta=1e-8):
        """
        Fit the OPLS model to the given input and target data.

        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).
            Y (array-like): Target data of shape (n_samples,).
            max_iter (int): Maximum number of iterations for optimization.
            delta (float): Convergence criterion.

        Returns:
            None
        """
        self.estimate_coef(X, Y)
        self.estimate_weight(X, Y)
        X_transform = self.transform(X)
        self.estimate_coef(X_transform, Y)

    def fit_transform(self, X, Y, max_iter=500, delta=1e-8):
        """
        Fit the OPLS model to the given input and target data and transform the input data.

        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).
            Y (array-like): Target data of shape (n_samples,).
            max_iter (int): Maximum number of iterations for optimization.
            delta (float): Convergence criterion.

        Returns:
            X_transformed (array-like): Transformed data of shape (n_samples, n_components).
        """
        self.fit(X, Y, max_iter=max_iter, delta=delta)
        return self.transform(X)

    def predict(self, X):
        """
        Predict target values for the input data X.

        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            Y_pred (array-like): Predicted target values of shape (n_samples,).
        """
        X_transform = self.transform(X)
        return X_transform @ self.coef_

    def score(self, X, Y):
        """
        Calculate the R^2 score of the model.

        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).
            Y (array-like): Target data of shape (n_samples,).

        Returns:
            score (float): R^2 score.
        """
        Y_pred = self.predict(X)
        return r2_score(Y, Y_pred)
