from collections import deque
import random
from typing import Any, List, Tuple, Optional, Union
import torch

# Type alias for RL transitions (state, action, next_state, reward, terminated)
Transition = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]


class ReplayMemory:
    """Experience Replay Buffer for storing and sampling reinforcement learning transitions."""

    def __init__(self, maxlen: int, seed: Optional[int] = None) -> None:
        """Initialize the replay buffer queue with a maximum capacity.
        
        Args:
            maxlen (int): Maximum number of transitions to store in buffer.
            seed (Optional[int]): Random seed for reproducible sampling.
        """
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, new_exp: Transition) -> None:
        """Add a new experience transition tuple to the buffer.
        
        Args:
            new_exp (Transition): Tuple containing (state, action, next_state, reward, terminated).
        """
        self.memory.append(new_exp)

    def sample(self, sample_size: int) -> List[Transition]:
        """Randomly sample a batch of transitions from memory.
        
        Args:
            sample_size (int): Number of transitions to sample.
            
        Returns:
            List[Transition]: List of sampled transition tuples.
        """
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        """Return the current number of stored experience transitions."""
        return len(self.memory)