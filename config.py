from dataclasses import dataclass
import os
import yaml


@dataclass
class DQNConfig:
    """Hyperparameter configuration schema for DQN training."""
    env_id: str
    epsilon_init: float
    epsilon_min: float
    epsilon_decay: float
    replay_memory_size: int
    mini_batch_size: int
    network_sync_rate: int
    alpha: float
    gamma: float
    reward_threshold: float

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "parameters.yaml", param_set: str = "flappybirdv0") -> "DQNConfig":
        """Load and validate configuration from YAML file.
        
        Args:
            yaml_path (str): Path to parameters YAML file.
            param_set (str): Hyperparameter set key name.
            
        Returns:
            DQNConfig: Configured dataclass instance.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file '{yaml_path}' not found.")

        with open(yaml_path, "r") as f:
            all_params = yaml.safe_load(f)

        if param_set not in all_params:
            raise KeyError(f"Parameter set '{param_set}' not found in '{yaml_path}'. Available: {list(all_params.keys())}")

        params = all_params[param_set]
        return cls(
            env_id=params.get("env_id", "FlappyBird-v0"),
            epsilon_init=float(params["epsilon_init"]),
            epsilon_min=float(params["epsilon_min"]),
            epsilon_decay=float(params["epsilon_decay"]),
            replay_memory_size=int(params["replay_memory_size"]),
            mini_batch_size=int(params["mini_batch_size"]),
            network_sync_rate=int(params["network_sync_rate"]),
            alpha=float(params["alpha"]),
            gamma=float(params["gamma"]),
            reward_threshold=float(params["reward_threshold"]),
        )
