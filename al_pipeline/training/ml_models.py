import gpytorch
import torch
from torch import nn

class GPRegressionModel(gpytorch.models.ExactGP):
    """
    A Gaussian Process Regression (GPR) model using GPyTorch.
    
    This class defines a GPR model with a flexible kernel.
    """
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        """
        Parameters:
        -----------
        train_x: torch.Tensor
            Training input data.
        train_y: torch.Tensor
            Training target data.
        likelihood: gpytorch.likelihoods.Likelihood
            The likelihood for the GP model.
        kernel: gpytorch.kernels.Kernel, optional
            The kernel to use in the GP model. Defaults to Matérn kernel.
        """
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Set default kernel to Matérn if none is provided
        if kernel is None:
            kernel = gpytorch.kernels.MaternKernel(nu=3./2)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
    
    def forward(self, x):
        """
        Forward pass through the GP model.
        
        Parameters:
        -----------
        x: torch.Tensor
            Input data.
        
        Returns:
        -----------
        gpytorch.distributions.MultivariateNormal
            The output distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultitaskGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super().__init__(train_x, train_y, likelihood)
        """
        Parameters:
        -----------
        train_x: torch.Tensor
            Training input data.
        train_y: torch.Tensor
            Training target data.
        likelihood: gpytorch.likelihoods.Likelihood
            The likelihood for the GP model.
        num_tasks: int
            Number of tasks for multitask learning.
        """
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(nu=3./2), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        """
        Forward pass through the multitask GP model.
        
        Parameters:
        -----------
        x: torch.Tensor
            Input data.
        
        Returns:
        -----------
        gpytorch.distributions.MultitaskMultivariateNormal
            The output distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    

# TODO: felsh out 
class DNN(torch.nn.Module):
    def __init__(self, dim_list, output_dim, activation = nn.Tanh(), dropout=0.0, batch_norm=False):
        super(DNN, self).__init__()
        """
        Parameters:
        -----------
        dim_list: list of int
            List of dimensions for each hidden layer.
        output_dim: int
            Dimension of the output layer.
        """
        self.linear     = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i+1]) for i in range(len(dim_list)-1)])
        self.out        = nn.Linear(dim_list[-1], output_dim)
        self.activation = activation
    def forward(self, x):
        for layer in self.linear:
            x = self.activation(layer(x))
        x = self.out(x)
        return x