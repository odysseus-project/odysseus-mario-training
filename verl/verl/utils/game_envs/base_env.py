from typing import Any, Optional, Union, Tuple

import gymnasium as gym
from abc import ABC, abstractmethod
from dataclasses import dataclass
from omegaconf import DictConfig
from PIL import Image
@dataclass
class Obs(ABC):
    pass

@dataclass
class Action(ABC):
    pass


class BaseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # Need to define these:
        # self.observation_space = spaces.YourSpace(...)
        # self.action_space = spaces.YourSpace(...)

    def reset(self, seed) -> Tuple[Any, dict]:
        super().reset(seed=seed)
        obs = self.initial_obs()
        return obs, {}

    def initial_obs(self) -> Obs:
        """Should return an object compatible with observation_space."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self, action: Action
    ) -> Tuple[Obs, float, bool, bool, dict]:
        pass

    def render(self):
        """Optional rendering logic."""
        pass

    def close(self):
        """Cleanup logic if needed."""
        pass
    
    def compute_reward(self) -> float:
        """Compute and return the reward for the current state."""
        pass
    
    def save_state(self, state_path: str):
        self.pyboy.save_state(open(state_path, "wb"))

    # def resize_image(self, image: Image.Image, resize_ratio: float = 1.0) -> Image.Image:
    #     if resize_ratio != 1.0:
    #         """Resize the image to the specified ratio."""
    #         new_size = (int(image.width * resize_ratio), int(image.height * resize_ratio))
    #         resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    #         return resized_image
    #     else:   
    #         """Return the original image."""
    #         return image
