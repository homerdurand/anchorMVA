import numpy as np


class AnchorOptimalProjection:
    def __init__(self, gamma=5):
        """
        Initialize the AnchorOptimalProjection class with a specified gamma value.

        Parameters:
        - gamma (float): A parameter controlling the transformation in the Anchor optimal space.
                         Default is 5.
        """
        self.gamma = gamma
        
    def fit(self, A):
        """
        Fit the AnchorOptimalProjection model to project input data into the Anchor optimal space.

        Parameters:
        - A (numpy.ndarray): Input matrix for fitting the model.
        """
        n = A.shape[0]
        if self.gamma == 1:
            # If gamma is 1, set AOP to the identity matrix
            self.AOP = np.identity(n)
        else:
            # Calculate the projection matrix based on gamma value
            P_A = A @ np.linalg.pinv(A.T @ A) @ A.T
            if self.gamma == 'IV':
                # If gamma is 'IV', set AOP to the projection matrix P_A
                self.AOP = P_A
            else:
                # Set AOP using the specified formula
                self.AOP = np.identity(n) + (np.sqrt(self.gamma) - 1) * P_A
        
    def transform(self, X, Y=None):
        """
        Transform input matrices X and Y using the fitted Anchor optimal projection matrix.

        Parameters:
        - X (numpy.ndarray): Input matrix to be transformed.
        - Y (numpy.ndarray, optional): Second input matrix to be transformed. Default is None.

        Returns:
        - numpy.ndarray or tuple of numpy.ndarray: Transformed matrix/matrix pair in the Anchor optimal space.
        """
        if Y is None:
            return self.AOP @ X
        else:
            return self.AOP @ X, self.AOP @ Y
    
    def fit_transform(self, A, X, Y=None):
        """
        Fit the model with input matrix A and transform input matrices X and Y into the Anchor optimal space.

        Parameters:
        - A (numpy.ndarray): Input matrix for fitting the model.
        - X (numpy.ndarray): First input matrix to be transformed.
        - Y (numpy.ndarray, optional): Second input matrix to be transformed. Default is None.

        Returns:
        - numpy.ndarray or tuple of numpy.ndarray: Transformed matrix/matrix pair in the Anchor optimal space.
        """
        self.fit(A)
        return self.transform(X, Y)
