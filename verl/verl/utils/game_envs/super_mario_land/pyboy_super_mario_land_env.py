from pyboy import PyBoy
from typing import Tuple, Any, Dict, List
from PIL import Image
import random
from verl.utils.game_envs.base_env import BaseEnv

class PyBoySuperMarioLandEnv(BaseEnv):
    """Mario training with screenshots, grayscale emulation, and progress rewards."""

    metadata = {"render_modes": []}
    BUTTONS = ['a', 'b', 'left', 'right', 'up', 'down', 'noop']
    MAX_STEP = 100000

    def __init__(
        self,
        rom_path: str,
        init_state: str = None,
        world: int = 1,
        level: int = 1,
        headless: bool = True,
    ):
        super().__init__()
        if headless:
            self.pyboy = PyBoy(
                rom_path,
                window="null",
                cgb=False,
            )
        else:
            self.pyboy = PyBoy(
                rom_path,
                cgb=False,
            )
        self.pyboy.set_emulation_speed(0)
        self.world = world
        self.level = level
        self.mario = self.pyboy.game_wrapper
        self.mario.start_game(world_level = (world, level))
        if init_state:
            self.init_state = init_state
            with open(init_state, "rb") as f:
                self.pyboy.load_state(f)
        else:
            self.init_state = None
        self.tick(1, True)
        self.step_count = 0

    def initial_obs(self) -> Dict:
        return self.get_observation()

    def step(self, action: List[str]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action = [str(btn).lower() for btn in action]
        # Preserve the per-step RNG draw used in training; level sampling shares this RNG.
        random.random()
        invalid_buttons = []
        if not all(btn in self.BUTTONS for btn in action):
            invalid_buttons = [btn for btn in action if btn not in self.BUTTONS]
            action = [btn for btn in action if btn in self.BUTTONS]
        self.press_buttons(action)

        if len(action) > 0:
            self.step_count += 1
        # Here, self.pyboy.memory[0xFFA6] accesses the byte in Game Boy memory
        # at address 0xFFA6, which is used by Super Mario Land to track the
        # 'death animation' timer. When this value exceeds 0x80 (i.e., 128 in decimal),
        # it indicates that Mario is dead or the death sequence is active, so the episode should terminate.
        done = (
            self.step_count >= self.MAX_STEP
            or self.mario.game_over()
            or self.pyboy.memory[0xFFA6] > 0x80  # 0xFFA6 > 0x80 means Mario is dead or in death animation
            or (self.mario.lives_left < 2 if self.init_lives_left == 2 else False)
        )
        reward = self.compute_reward()

        obs = self.get_observation()
        self.set_prev_state()
        return obs, reward, done, False, {"invalid_buttons": invalid_buttons, "level_progress": float(self.mario.level_progress), "game_score": float(self.mario.score)}

    def compute_reward(self):
        if self.pyboy.memory[0xFFA6] > 0x80:
            lives_diff = -1
        else:
            lives_diff = self.mario.lives_left - self.prev_lives_left
        level_progress_diff = (self.mario.level_progress - self.prev_level_progress) * float(lives_diff == 0)
        return 10.0 * level_progress_diff

    def set_prev_state(self):
        self.prev_lives_left = self.mario.lives_left
        self.prev_level_progress = self.mario.level_progress

    def reset(self, seed) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.mario.reset_game()
        if self.init_state:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
        self.tick(1, True)
        self.step_count = 0
        self.set_prev_state()
        self.init_lives_left = self.mario.lives_left
        obs = self.get_observation()
        return obs, {}

    def close(self):
        self.pyboy.stop(save=False)

    # Helper functions
    def get_screenshot(self):
        """Get the current screenshot."""
        return Image.fromarray(self.pyboy.screen.ndarray)

    def get_observation(self):
        """Return the screenshot and the empty state text used by training."""
        return {"image": self.get_screenshot(), "state": ""}

    def tick(self, *args, **kwargs):
        """Advance the emulator by the specified number of frames."""
        self.pyboy.tick(*args, **kwargs)

    def press_buttons(self, buttons):
        """Hold jump actions for 15 frames and other actions for 5, then release."""
        for button in buttons:
            if button == "noop":
                continue
            self.pyboy.button_press(button)
        if "a" in buttons:
            self.tick(15, False)
        else:
            self.tick(5, False)
        for button in buttons:
            if button == "noop":
                continue
            self.pyboy.button_release(button)
        self.tick(1, True) # Immediate for real time games
