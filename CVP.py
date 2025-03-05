import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.cluster import KMeans


# Define the Conditional Variance Penalty model
class CVPLinearRegression:
    def __init__(self, input_dim, output_dim, lambda_cvp=1.0, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_cvp = lambda_cvp  # The regularization strength for conditional variance penalty
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self.model = nn.Linear(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def cvp_loss(self, X, Y, envs):
        """
        Compute the Conditional Variance Penalty Loss: risk minimization + conditional variance regularization.
        """
        mse_loss = nn.MSELoss()
        # Standard MSE loss
        loss = mse_loss(self.model(X), Y)
        
        # Conditional variance regularization: penalize high variance across environments
        unique_envs = np.unique(envs)
        for env in unique_envs:
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]
            
            # Predictions for this environment
            predictions = self.model(X_env)
            
            # Calculate variance of predictions across different environments for the same input
            prediction_variance = torch.var(predictions, dim=0)
            loss += self.lambda_cvp * torch.sum(prediction_variance)

        return loss

    def fit(self, X_train, Y_train, envs_train):
        """
        Train the model using Conditional Variance Penalty with early stopping.
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
            
            # Compute the CVP loss
            loss = self.cvp_loss(X_tensor, Y_tensor, envs_train)
            
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
        Set the coefficients and intercept from the learned CVP model to an sklearn LinearRegression model.
        """
        sklearn_model = LinearRegression()
        
        # Set coefficients and intercept in the sklearn model
        with torch.no_grad():
            sklearn_model.coef_ = self.model.weight.detach().numpy()
            sklearn_model.intercept_ = self.model.bias.detach().numpy()

        return sklearn_model


# Define the Conditional Variance Penalty model
class CVPLinearRegression2:
    def __init__(self, input_dim, output_dim, lambda_cvp=1.0, n_epochs=1000, learning_rate=0.01, patience=20, tol=1e-4, verbose=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_cvp = lambda_cvp  # The regularization strength for conditional variance penalty
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tol = tol
        self.verbose = verbose
        self.model = nn.Linear(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def cvp_loss(self, X, Y, envs):
        """
        Compute the Conditional Variance Penalty Loss: risk minimization + conditional variance regularization.
        """
        mse_loss = nn.MSELoss()
        # Standard MSE loss
        loss = mse_loss(self.model(X), Y)
        
        # Conditional variance regularization: penalize high variance across environments
        unique_envs = np.unique(envs)
        for env in unique_envs:
            mask = envs == env
            X_env, Y_env = X[mask], Y[mask]
            
            # Predictions for this environment
            predictions = self.model(X_env)
            
            # Calculate variance of predictions across different environments for the same input
            prediction_variance = torch.var(predictions, dim=0)
            loss += self.lambda_cvp * torch.sum(prediction_variance)

        return loss

    def cvp_loss2(self, X, Y, envs, n_groups=5):
        """
        Compute the Conditional Variance Penalty Loss: risk minimization + conditional variance regularization.
        The regularization is conditioned on discretized Y values.
        
        Args:
            X: Input features (Tensor, shape [batch_size, input_dim]).
            Y: Multi-output targets (Tensor, shape [batch_size, output_dim]).
            envs: Environment labels (Tensor, shape [batch_size]).
            n_groups: Number of groups to discretize Y into.
        
        Returns:
            Total loss with conditional variance penalty.
        """
        mse_loss = nn.MSELoss()
        # Standard MSE loss
        loss = mse_loss(self.model(X), Y)
        
        # Discretize Y into groups
        Y_flat = Y.view(-1, Y.shape[-1])  # Flatten Y for multi-output processing
        Y_groups = torch.zeros_like(Y_flat[:, 0], dtype=torch.long)  # Initialize group indices
        for i in range(Y.shape[1]):  # Iterate over each output dimension
            quantiles = torch.quantile(Y_flat[:, i], torch.linspace(0, 1, n_groups + 1))
            Y_groups += torch.bucketize(Y_flat[:, i], quantiles[:-1]) * (i + 1)
        
        # Unique (Y_group, env) combinations
        unique_combinations = torch.unique(torch.stack([Y_groups, envs]), dim=1)
        
        # Conditional variance regularization
        for group, env in unique_combinations.t():
            mask = (Y_groups == group) & (envs == env)
            X_group, Y_group = X[mask], Y[mask]
            
            if X_group.shape[0] > 1:  # Variance requires at least two samples
                predictions = self.model(X_group)
                prediction_variance = torch.var(predictions, dim=0)
                loss += self.lambda_cvp * torch.sum(prediction_variance)

        return loss

    def cvp_loss_kmeans(self, X, Y, envs, n_groups=5):
        """
        Compute the Conditional Variance Penalty Loss: risk minimization + conditional variance regularization.
        The regularization is conditioned on groups created using k-means clustering on Y.

        Args:
            X: Input features (Tensor, shape [batch_size, input_dim]).
            Y: Multi-output targets (Tensor, shape [batch_size, output_dim]).
            envs: Environment labels (Tensor, shape [batch_size]).
            n_groups: Number of groups for k-means clustering.

        Returns:
            Total loss with conditional variance penalty.
        """
        mse_loss = nn.MSELoss()
        # Standard MSE loss
        loss = mse_loss(self.model(X), Y)

        # Convert Y to numpy for k-means clustering
        Y_numpy = Y.detach().cpu().numpy()

        # Perform k-means clustering on Y
        kmeans = KMeans(n_clusters=n_groups, random_state=42)
        group_labels = kmeans.fit_predict(Y_numpy)  # Cluster assignments for each sample
        group_labels = torch.tensor(group_labels, dtype=torch.long, device=Y.device)  # Convert back to tensor

        # Unique (group, env) combinations
        unique_combinations = torch.unique(torch.stack([group_labels, envs]), dim=1)

        # Conditional variance regularization
        for group, env in unique_combinations.t():
            mask = (group_labels == group) & (envs == env)
            X_group, Y_group = X[mask], Y[mask]

            if X_group.shape[0] > 1:  # Variance requires at least two samples
                predictions = self.model(X_group)
                prediction_variance = torch.var(predictions, dim=0)
                loss += self.lambda_cvp * torch.sum(prediction_variance)

        return loss

    def fit(self, X_train, Y_train, envs_train):
        """
        Train the model using Conditional Variance Penalty with early stopping.
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
            
            # Compute the CVP loss
            loss = self.cvp_loss2(X_tensor, Y_tensor, envs_train)
            
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
        Set the coefficients and intercept from the learned CVP model to an sklearn LinearRegression model.
        """
        sklearn_model = LinearRegression()
        
        # Set coefficients and intercept in the sklearn model
        with torch.no_grad():
            sklearn_model.coef_ = self.model.weight.detach().numpy()
            sklearn_model.intercept_ = self.model.bias.detach().numpy()

        return sklearn_model