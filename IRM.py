import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import KFold

# Define the IRM Training Class with Early Stopping and Verbose Option
class IRMLinearRegression:
    def __init__(self, input_dim, output_dim, lambda_irm=1.0, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_irm = lambda_irm
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience  # Number of epochs to wait before stopping
        self.tol = tol  # Tolerance for early stopping (how much MSE must change)
        self.verbose = verbose  # Verbose flag
        self.model = nn.Linear(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def irm_loss(self, X, Y, envs):
        """
        Compute the IRM loss: risk minimization + invariance regularization.
        """
        mse_loss = nn.MSELoss()
        # Standard MSE loss
        loss = mse_loss(self.model(X), Y)
        
        # Invariance regularization: enforce invariant predictions across environments
        for env in np.unique(envs):
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]
            
            # Calculate the gradient of the loss with respect to the model predictions
            X_env.requires_grad_(True)
            predictions = self.model(X_env)
            grad = torch.autograd.grad(predictions, X_env, grad_outputs=torch.ones_like(predictions), create_graph=True)[0]
            
            # Penalize non-invariant representations
            grad_penalty = torch.norm(grad, p=2)
            loss += self.lambda_irm * grad_penalty

        return loss

    def fit(self, X_train, Y_train, envs_train):
        """
        Train the model using IRM with early stopping.
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
        
        # Early stopping parameters
        best_loss = float('inf')
        patience_counter = 0
        
        # Train the model
        for epoch in range(self.n_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Compute the IRM loss
            loss = self.irm_loss(X_tensor, Y_tensor, envs_train)
            
            # Backpropagate
            loss.backward()
            self.optimizer.step()
            
            # Calculate MSE for early stopping
            with torch.no_grad():
                Y_pred = self.model(X_tensor)
                mse = mean_squared_error(Y_tensor.numpy(), Y_pred.numpy())
            
            # Early stopping check
            if mse < best_loss - self.tol:
                best_loss = mse
                patience_counter = 0  # Reset counter if MSE improves
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}/{self.n_epochs} (MSE did not improve).")
                break

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}, Loss: {loss.item():.4f}, MSE: {mse:.4f}")

    def set_sklearn_model(self):
        """
        Set the coefficients and intercept from the learned IRM model to an sklearn LinearRegression model.
        """
        sklearn_model = LinearRegression()
        
        # Set coefficients and intercept in the sklearn model
        with torch.no_grad():
            sklearn_model.coef_ = self.model.weight.detach().numpy()
            sklearn_model.intercept_ = self.model.bias.detach().numpy()

        return sklearn_model




class IRMLinearRegression2:
    def __init__(self, input_dim, output_dim, lambda_irm=1.0, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        """
        Multi-output Invariant Risk Minimization Linear Regression.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_irm = lambda_irm
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose

        # Linear model for multi-output regression
        self.phi = nn.Parameter(torch.eye(input_dim, input_dim))  # Learnable representation
        self.w = nn.Linear(input_dim, output_dim, bias=False)  # Linear predictor
        self.optimizer = optim.Adam([self.phi, self.w.weight], lr=learning_rate)

    def irm_loss(self, X, Y, envs):
        """
        Compute the IRM loss: risk minimization + invariance penalty for multi-output regression.
        """
        mse_loss = nn.MSELoss()
        penalty = 0
        total_loss = 0

        for env in torch.unique(envs):
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]

            # Transform input with learnable phi
            X_transformed = X_env @ self.phi
            predictions = self.w(X_transformed)

            # Compute environment-specific loss
            error = mse_loss(predictions, Y_env)
            total_loss += error

            # Compute gradient penalty for invariance
            grad = torch.autograd.grad(error, self.w.weight, create_graph=True)[0]
            penalty += grad.pow(2).mean()

        # Combine loss and penalty
        return total_loss + self.lambda_irm * penalty

    def fit(self, X_train, Y_train, envs_train):
        """
        Train the IRM model using early stopping.
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
        envs_tensor = torch.tensor(envs_train, dtype=torch.int64)

        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            loss = self.irm_loss(X_tensor, Y_tensor, envs_tensor)
            loss.backward()
            self.optimizer.step()

            # Calculate mean squared error for early stopping
            with torch.no_grad():
                predictions = self.predict(X_train)
                mse = mean_squared_error(Y_train, predictions)

            # Early stopping logic
            if mse < best_loss - self.tol:
                best_loss = mse
                patience_counter = 0  # Reset patience
                best_phi = self.phi.detach().clone()
                best_w = self.w.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}/{self.n_epochs}.")
                break

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}, Loss: {loss.item():.4f}, MSE: {mse:.4f}")

        # Restore best parameters
        self.phi = nn.Parameter(best_phi)
        self.w.load_state_dict(best_w)

    def predict(self, X):
        """
        Make predictions on new data. Returns NumPy array.
        """
        self.w.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            X_transformed = X_tensor @ self.phi
            predictions = self.w(X_transformed)
        return predictions.numpy()

    def get_representation(self, X):
        """
        Get the learned representation of the input data.
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return (X_tensor @ self.phi).numpy()




class IRMLinearCV:
    def __init__(self, input_dim, output_dim, lambdas_irm=None, lambdas_ridge=None, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        """
        Multi-output Invariant Risk Minimization Linear Regression with cross-validation for hyperparameter tuning.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Hyperparameters for IRM and L2 regularization
        self.lambdas_irm = lambdas_irm if lambdas_irm is not None else [0.001, 0.01, 0.1, 1.0, 10.0]
        self.lambdas_ridge = lambdas_ridge if lambdas_ridge is not None else [0.1, 1.0, 10.0, 100.0, 1000.0]

        # Training parameters
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose

        # Best hyperparameters and weights will be stored here
        self.best_lambda_irm = None
        self.best_lambda_ridge = None
        self.best_phi = None
        self.best_w = None

    def irm_loss(self, X, Y, envs, lambda_irm, lambda_ridge):
        """
        Compute the IRM loss with L2 (ridge) regularization for the regression model.
        """
        mse_loss = nn.MSELoss()
        penalty = 0
        total_loss = 0

        for env in torch.unique(envs):
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]

            # Transform input with learnable phi
            X_transformed = X_env @ self.phi
            predictions = self.w(X_transformed)

            # Compute environment-specific loss
            error = mse_loss(predictions, Y_env)
            total_loss += error

            # Compute gradient penalty for invariance
            grad = torch.autograd.grad(error, self.w.weight, create_graph=True)[0]
            penalty += grad.pow(2).mean()

        # Add IRM regularization term
        total_loss += lambda_irm * penalty
        
        # Add L2 regularization (ridge) term for the regression weights
        l2_penalty = lambda_ridge * torch.sum(self.w.weight ** 2)
        total_loss += l2_penalty

        return total_loss

    def fit(self, X_train, Y_train, envs_train, X_val, Y_val, envs_val):
        """
        Train the IRM model using cross-validation to optimize lambda_irm and lambda_ridge.
        """
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        envs_train_tensor = torch.tensor(envs_train, dtype=torch.int64)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
        envs_val_tensor = torch.tensor(envs_val, dtype=torch.int64)

        best_val_loss = float('inf')

        # Cross-validation over lambda_irm and lambda_ridge hyperparameters
        for lambda_irm in self.lambdas_irm:
            for lambda_ridge in self.lambdas_ridge:
                # Initialize model parameters
                self.phi = nn.Parameter(torch.eye(self.input_dim, self.input_dim))  # Learnable representation
                self.w = nn.Linear(self.input_dim, self.output_dim, bias=False)  # Linear predictor
                optimizer = optim.Adam([self.phi, self.w.weight], lr=self.learning_rate)

                # Train the model for a fixed number of epochs
                best_loss = float('inf')
                patience_counter = 0

                for epoch in range(self.n_epochs):
                    optimizer.zero_grad()
                    loss = self.irm_loss(X_train_tensor, Y_train_tensor, envs_train_tensor, lambda_irm, lambda_ridge)
                    loss.backward()
                    optimizer.step()

                    # Compute validation loss
                    with torch.no_grad():
                        val_loss = self.irm_loss(X_val_tensor, Y_val_tensor, envs_val_tensor, lambda_irm, lambda_ridge)

                    # Early stopping
                    if val_loss < best_loss - self.tol:
                        best_loss = val_loss
                        patience_counter = 0
                        best_phi = self.phi.detach().clone()
                        best_w = self.w.state_dict()
                    else:
                        patience_counter += 1

                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}/{self.n_epochs}.")
                        break

                    if self.verbose and epoch % 100 == 0:
                        print(f"Epoch {epoch}/{self.n_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

                # Restore the best model parameters
                self.phi = nn.Parameter(best_phi)
                self.w.load_state_dict(best_w)

                # Update the best hyperparameters if validation loss improved
                if best_loss < best_val_loss:
                    best_val_loss = best_loss
                    self.best_lambda_irm = lambda_irm
                    self.best_lambda_ridge = lambda_ridge
                    self.best_phi = best_phi
                    self.best_w = best_w

        return self

    def predict(self, X):
        """
        Make predictions on new data. Returns NumPy array.
        """
        self.w.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            X_transformed = X_tensor @ self.best_phi
            predictions = self.best_w["weight"] @ X_transformed.T
        return predictions.numpy()

    def get_representation(self, X):
        """
        Get the learned representation of the input data.
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return (X_tensor @ self.best_phi).numpy()

class IRMLinearRegression2:
    def __init__(self, input_dim, output_dim, lambda_irm=1.0, alpha=1, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        """
        Multi-output Invariant Risk Minimization Linear Regression.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_irm = lambda_irm
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self.alpha = alpha

        # Linear model for multi-output regression
        self.phi = nn.Parameter(torch.eye(input_dim, input_dim))  # Learnable representation
        self.w = nn.Linear(input_dim, output_dim, bias=False)  # Linear predictor
        self.optimizer = optim.Adam([self.phi, self.w.weight], lr=learning_rate)

    def irm_loss(self, X, Y, envs):
        """
        Compute the IRM loss: risk minimization + invariance penalty for multi-output regression.
        """
        mse_loss = nn.MSELoss()
        penalty = 0
        total_loss = 0

        for env in torch.unique(envs):
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]

            # Transform input with learnable phi
            X_transformed = X_env @ self.phi
            predictions = self.w(X_transformed)

            # Compute environment-specific loss
            error = mse_loss(predictions, Y_env)
            total_loss += error

            # Compute gradient penalty for invariance
            grad = torch.autograd.grad(error, self.w.weight, create_graph=True)[0]
            penalty += grad.pow(2).mean()

        # Combine loss and penalty
        return total_loss + self.lambda_irm * penalty

    def irm_loss2(self, X, Y, envs):
        """
        Compute the IRM loss: risk minimization + invariance penalty for multi-output regression.
        """
        mse_loss = nn.MSELoss()
        penalty = 0
        total_loss = 0

        for env in torch.unique(envs):
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]

            # Transform input with learnable phi
            X_transformed = X_env @ self.phi
            predictions = self.w(X_transformed)

            # Compute environment-specific loss
            error = mse_loss(predictions, Y_env)
            total_loss += error

            # Compute gradient penalty for invariance
            grad = torch.autograd.grad(error, self.w.weight, create_graph=True)[0]
            penalty += grad.pow(2).mean()

        # L2 regularization term
        l2_reg = self.phi.pow(2).sum() 

        # Combine loss and penalty
        return total_loss + self.lambda_irm * penalty + self.alpha * l2_reg

    def fit(self, X_train, Y_train, envs_train):
        """
        Train the IRM model using early stopping.
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
        envs_tensor = torch.tensor(envs_train, dtype=torch.int64)

        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            loss = self.irm_loss2(X_tensor, Y_tensor, envs_tensor)
            loss.backward()
            self.optimizer.step()

            # Calculate mean squared error for early stopping
            with torch.no_grad():
                predictions = self.predict(X_train)
                mse = mean_squared_error(Y_train, predictions)

            # Early stopping logic
            if mse < best_loss - self.tol:
                best_loss = mse
                patience_counter = 0  # Reset patience
                best_phi = self.phi.detach().clone()
                best_w = self.w.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}/{self.n_epochs}.")
                break

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}, Loss: {loss.item():.4f}, MSE: {mse:.4f}")

        # Restore best parameters
        self.phi = nn.Parameter(best_phi)
        self.w.load_state_dict(best_w)

    def predict(self, X):
        """
        Make predictions on new data. Returns NumPy array.
        """
        self.w.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            X_transformed = X_tensor @ self.phi
            predictions = self.w(X_transformed)
        return predictions.numpy()

    def get_representation(self, X):
        """
        Get the learned representation of the input data.
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return (X_tensor @ self.phi).numpy()




class IRMLinearCV:
    def __init__(self, input_dim, output_dim, lambdas_irm=None, lambdas_ridge=None, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        """
        Multi-output Invariant Risk Minimization Linear Regression with cross-validation for hyperparameter tuning.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Hyperparameters for IRM and L2 regularization
        self.lambdas_irm = lambdas_irm if lambdas_irm is not None else [0.001, 0.01, 0.1, 1.0, 10.0]
        self.lambdas_ridge = lambdas_ridge if lambdas_ridge is not None else [0.1, 1.0, 10.0, 100.0, 1000.0]

        # Training parameters
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose

        # Best hyperparameters and weights will be stored here
        self.best_lambda_irm = None
        self.best_lambda_ridge = None
        self.best_phi = None
        self.best_w = None

    def irm_loss(self, X, Y, envs, lambda_irm, lambda_ridge):
        """
        Compute the IRM loss with L2 (ridge) regularization for the regression model.
        """
        mse_loss = nn.MSELoss()
        penalty = 0
        total_loss = 0

        for env in torch.unique(envs):
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]

            # Transform input with learnable phi
            X_transformed = X_env @ self.phi
            predictions = self.w(X_transformed)

            # Compute environment-specific loss
            error = mse_loss(predictions, Y_env)
            total_loss += error

            # Compute gradient penalty for invariance
            grad = torch.autograd.grad(error, self.w.weight, create_graph=True)[0]
            penalty += grad.pow(2).mean()

        # Add IRM regularization term
        total_loss += lambda_irm * penalty
        
        # Add L2 regularization (ridge) term for the regression weights
        l2_penalty = lambda_ridge * torch.sum(self.w.weight ** 2)
        total_loss += l2_penalty

        return total_loss

    def fit(self, X_train, Y_train, envs_train, X_val, Y_val, envs_val):
        """
        Train the IRM model using cross-validation to optimize lambda_irm and lambda_ridge.
        """
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        envs_train_tensor = torch.tensor(envs_train, dtype=torch.int64)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)
        envs_val_tensor = torch.tensor(envs_val, dtype=torch.int64)

        best_val_loss = float('inf')

        # Cross-validation over lambda_irm and lambda_ridge hyperparameters
        for lambda_irm in self.lambdas_irm:
            for lambda_ridge in self.lambdas_ridge:
                # Initialize model parameters
                self.phi = nn.Parameter(torch.eye(self.input_dim, self.input_dim))  # Learnable representation
                self.w = nn.Linear(self.input_dim, self.output_dim, bias=False)  # Linear predictor
                optimizer = optim.Adam([self.phi, self.w.weight], lr=self.learning_rate)

                # Train the model for a fixed number of epochs
                best_loss = float('inf')
                patience_counter = 0

                for epoch in range(self.n_epochs):
                    optimizer.zero_grad()
                    loss = self.irm_loss(X_train_tensor, Y_train_tensor, envs_train_tensor, lambda_irm, lambda_ridge)
                    loss.backward()
                    optimizer.step()

                    # Compute validation loss
                    with torch.no_grad():
                        val_loss = self.irm_loss(X_val_tensor, Y_val_tensor, envs_val_tensor, lambda_irm, lambda_ridge)

                    # Early stopping
                    if val_loss < best_loss - self.tol:
                        best_loss = val_loss
                        patience_counter = 0
                        best_phi = self.phi.detach().clone()
                        best_w = self.w.state_dict()
                    else:
                        patience_counter += 1

                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}/{self.n_epochs}.")
                        break

                    if self.verbose and epoch % 100 == 0:
                        print(f"Epoch {epoch}/{self.n_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

                # Restore the best model parameters
                self.phi = nn.Parameter(best_phi)
                self.w.load_state_dict(best_w)

                # Update the best hyperparameters if validation loss improved
                if best_loss < best_val_loss:
                    best_val_loss = best_loss
                    self.best_lambda_irm = lambda_irm
                    self.best_lambda_ridge = lambda_ridge
                    self.best_phi = best_phi
                    self.best_w = best_w

        return self

    def predict(self, X):
        """
        Make predictions on new data. Returns NumPy array.
        """
        self.w.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            X_transformed = X_tensor @ self.best_phi
            predictions = self.best_w["weight"] @ X_transformed.T
        return predictions.numpy()

    def get_representation(self, X):
        """
        Get the learned representation of the input data.
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return (X_tensor @ self.best_phi).numpy()



class IRMLinearRegression3:
    def __init__(self, input_dim, output_dim, lambda_irm=1.0, alpha=1, n_epochs=100, learning_rate=0.01, patience=20, tol=1e-2, verbose=False):
        """
        Multi-output Invariant Risk Minimization Linear Regression.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_irm = lambda_irm
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self.alpha = alpha 

        # Linear model for multi-output regression
        self.phi = nn.Parameter(torch.eye(input_dim, input_dim))  # Learnable representation
        self.w = nn.Parameter(torch.eye(input_dim, output_dim))  # Linear predictor
        self.optimizer = optim.Adam([self.phi], lr=learning_rate)

    def irm_loss(self, X, Y, envs):
        """
        Compute the IRM loss: risk minimization + invariance penalty for multi-output regression.
        """
        mse_loss = nn.MSELoss()
        penalty = 0
        total_loss = 0

        for env in torch.unique(envs):
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]

            # Transform input with learnable phi
            X_transformed = X_env @ self.phi
            predictions = X_transformed @ self.w

            # Compute environment-specific loss
            error = mse_loss(predictions, Y_env)
            total_loss += error

            # Compute gradient penalty for invariance
            grad = torch.autograd.grad(error, self.w, create_graph=True)[0]
            penalty += grad.pow(2).mean()

        # L2 regularization term
        l2_reg = self.phi.pow(2).sum() 

        # Combine loss and penalty
        return total_loss + self.lambda_irm * penalty + self.alpha * l2_reg

    def fit(self, X_train, Y_train, envs_train):
        """
        Train the IRM model using early stopping.
        """
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32)
        envs_tensor = torch.tensor(envs_train, dtype=torch.int64)

        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            loss = self.irm_loss(X_tensor, Y_tensor, envs_tensor)
            loss.backward()
            self.optimizer.step()

            # Calculate mean squared error for early stopping
            with torch.no_grad():
                predictions = self.predict(X_train)
                mse = mean_squared_error(Y_train, predictions)

            # Early stopping logic
            if mse < best_loss - self.tol:
                # print(mse)
                best_loss = mse
                patience_counter = 0  # Reset patience
                best_phi = self.phi.detach().clone()
                best_w = self.w#.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}/{self.n_epochs}.")
                break

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}, Loss: {loss.item():.4f}, MSE: {mse:.4f}")

        # Restore best parameters
        self.phi = nn.Parameter(best_phi)
        self.w = nn.Parameter(best_w)#.load_state_dict(best_w)

    def predict(self, X):
        """
        Make predictions on new data. Returns NumPy array.
        """
        # self.w.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            X_transformed = X_tensor @ self.phi
            predictions = X_transformed @ self.w
        return predictions.numpy()

    def get_representation(self, X):
        """
        Get the learned representation of the input data.
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return (X_tensor @ self.phi).numpy()


class IRMCV:
    def __init__(self, input_dim, output_dim, lambda_grid, alpha_grid, n_epochs=100, learning_rate=0.01, patience=20, tol=1e-2, verbose=False):
        """
        Hyperparameter optimization for IRMLinearRegression3 using cross-validation.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_grid = lambda_grid
        self.alpha_grid = alpha_grid
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self.best_model = None

    def fit(self, X_train, Y_train, envs_train, X_val, Y_val, metric="r2"):
        """
        Fit the model by optimizing the hyperparameters `lambda_irm` and `alpha` over the grid of possible values.

        Parameters:
            X_train (np.array): Training features.
            Y_train (np.array): Training targets.
            envs_train (np.array): Training environments.
            X_val (np.array): Validation features.
            Y_val (np.array): Validation targets.
            metric (str): Metric to optimize, either "r2" for R² score or "mse" for Mean Squared Error.

        Returns:
            self: Trained IRMCV instance.
        """
        best_lambda = None
        best_alpha = None
        best_score = -float('inf') if metric == "r2" else float('inf')

        for lambda_irm in self.lambda_grid:
            for alpha in self.alpha_grid:
                # Initialize and train model
                model = IRMLinearRegression3(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    lambda_irm=lambda_irm,
                    alpha=alpha,
                    n_epochs=self.n_epochs,
                    learning_rate=self.learning_rate,
                    patience=self.patience,
                    tol=self.tol,
                    verbose=self.verbose
                )
                model.fit(X_train, Y_train, envs_train)

                # Evaluate on validation set
                predictions = model.predict(X_val)
                if metric == "r2":
                    score = r2_score(Y_val, predictions)
                elif metric == "mse":
                    score = -mean_squared_error(Y_val, predictions)  # Negative for minimization
                else:
                    raise ValueError("Unsupported metric. Use 'r2' or 'mse'.")

                # Update best model if applicable
                if (metric == "r2" and score > best_score) or (metric == "mse" and score < best_score):
                    best_score = score
                    best_lambda = lambda_irm
                    best_alpha = alpha
                    self.best_model = model

                if self.verbose:
                    print(f"Lambda: {lambda_irm:.4f}, Alpha: {alpha:.4f}, Validation Score ({metric}): {score:.4f}")

        if self.verbose:
            print(f"Best Lambda: {best_lambda}, Best Alpha: {best_alpha}, Best Validation Score: {best_score:.4f}")

        return self

    def predict(self, X_test):
        """
        Make predictions using the best model found during fitting.

        Parameters:
            X_test (np.array): Test features.

        Returns:
            predictions (np.array): Predicted values.
        """
        if self.best_model is None:
            raise RuntimeError("You must call fit() before predict().")

        return self.best_model.predict(X_test)

# Example usage:
# irm_cv = IRMCV(input_dim=10, output_dim=1, lambda_grid=[0.1, 1.0, 10.0], alpha_grid=[0.01, 0.1, 1.0], n_epochs=100, verbose=True)
# irm_cv.fit(X_train, Y_train, envs_train, X_val, Y_val, metric="r2")
# predictions = irm_cv.predict(X_test)