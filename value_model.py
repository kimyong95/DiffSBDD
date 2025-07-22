import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import gpytorch
import torch.nn as nn
import einops
import time

def with_1d_support(transform_func):
    """Decorator to add 1D input support to transform methods."""
    def wrapper(self, data):
        is_1d = data.ndim == 1
        if is_1d:
            data = data.unsqueeze(0)
        
        result = transform_func(self, data)
        
        if is_1d:
            result = result.squeeze(0)
        
        return result
    return wrapper

class BaseScaler(nn.Module):
    """
    Base class for scalers. It's an "empty" scaler that does nothing.
    `fit`, `transform`, and `inverse_transform` can be overridden by subclasses.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def fit(self, data):
        """Fits the scaler to the data. For this base class, it does nothing."""
        pass

    @with_1d_support
    def transform(self, data):
        """Transforms the data. For this base class, it returns the data as is."""
        return data

    @with_1d_support
    def inverse_transform(self, data):
        """Inverse transforms the data. For this base class, it returns the data as is."""
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class StandardScaler(BaseScaler):
    def __init__(self, feature_dim):
        super().__init__(feature_dim)
        self.register_buffer('mean', torch.zeros(feature_dim))
        self.register_buffer('std', torch.ones(feature_dim))

    def fit(self, data):
        """Computes the mean and standard deviation for scaling."""
        if data.ndim == 1:
            self.mean[:] = data.mean()
            self.std[:] = data.std().clamp(min=1e-8)
        else:
            self.mean[:] = data.mean(dim=0)
            self.std[:] = data.std(dim=0).clamp(min=1e-8)

    @with_1d_support
    def transform(self, data):
        """Standardizes the data."""
        device = data.device
        return (data - self.mean.to(device)) / self.std.to(device)

    @with_1d_support
    def inverse_transform(self, data):
        """Reverts the standardization."""
        device = data.device
        return data * self.std.to(device) + self.mean.to(device)

class ExactGpModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, x_dim):
        super(ExactGpModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = (x_dim) ** 0.5

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ValueModel(nn.Module):
    def __init__(self, dimension, noise_level = 1e-2) -> None:
        super().__init__()
        self.dim = dimension
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = noise_level
        self.likelihood.eval()

        model = ExactGpModel(None, None, self.likelihood, x_dim=dimension)

        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

        self.x_scaler = BaseScaler(dimension)
        self.y_scaler = BaseScaler(1)

        self.register_buffer("all_x", torch.empty(0, dimension, dtype=torch.float32))
        self.register_buffer("all_y", torch.empty(0, dtype=torch.float32))

    @torch.no_grad()
    def predict(self, x):
        device = x.device
        self.model.to(device)
        self.likelihood.to(device)
        
        with gpytorch.settings.fast_pred_var():
            y_preds = self.likelihood(self.model(self.x_scaler.transform(x)))
        
        y_preds_mean = self.y_scaler.inverse_transform(y_preds.mean.to(device))
        y_preds_var = y_preds.variance.to(device)

        # torch.cuda.empty_cache()

        return y_preds_mean.to(device), y_preds_var.to(device)

    # x: data points
    # y: lower is better
    def add_model_data(self, x, y):
        device = x.device

        # self.all_x = torch.cat([self.all_x, x], dim=0)
        # self.all_y = torch.cat([self.all_y, y], dim=0)

        self.all_x = x
        self.all_y = y

        

        self.x_scaler.fit(self.all_x)
        self.y_scaler.fit(self.all_y)
        
        self.model.set_train_data(
            inputs=self.x_scaler.transform(self.all_x),
            targets=self.y_scaler.transform(self.all_y),
            strict=False
        )

