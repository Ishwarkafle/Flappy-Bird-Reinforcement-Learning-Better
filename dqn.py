import torch
import torch.nn as nn


class DQN(nn.Module):
    """Deep Q-Network (DQN) policy model for estimating action-values."""

    def __init__(self, state_dim: int = 12, action_dim: int = 2, hidden_dim: int = 256):
        """Initialize the DQN network architecture.
        
        Args:
            state_dim (int): Dimensionality of the input state vector.
            action_dim (int): Number of discrete actions available.
            hidden_dim (int): Size of the hidden layer.
        """
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        return self.model(x)