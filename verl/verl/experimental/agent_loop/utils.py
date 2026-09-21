# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tokenizer.apply_chat_template is not working properly for gpt-oss model.
# Because the chat template requires tool call messages to parse tool response messages
# so we need to format the tool response manually.

from typing import List

from PIL import Image

def format_gpt_oss_tool_response_manually(tool_response: str, tool_call_name: str) -> str:
    """Format tool response for gpt-oss model.
    Args:
        tool_response: Tool response string
        tool_call_name: Name of the tool that was called

    Returns:
        Formatted tool response string
    """
    return f"<|start|>functions.{tool_call_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"


def add_generation_prompt_for_gpt_oss(message_content: str) -> str:
    """Add generation prompt for gpt-oss model.
    Args:
        message_content: Message content string

    Returns:
        Message content string with generation prompt
    """
    return message_content + "<|start|>assistant"


def resize_image(image: Image.Image, resize_ratio: float = 1.0, resize_width: int = None, resize_height: int = None) -> Image.Image:
    if resize_ratio != 1.0:
        """Resize the image to the specified ratio."""
        new_size = (int(image.width * resize_ratio), int(image.height * resize_ratio))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        return resized_image
    if resize_width is not None and resize_height is not None:
        """Resize the image to the specified height and width."""
        resized_image = image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        return resized_image
    return image


def compute_return_to_go(reward_list: List[float], discount_factor: float) -> List[float]:
    """Compute return-to-go for each turn."""
    running_return = 0
    return_to_go = []
    for reward in reversed(reward_list):
        running_return = reward + discount_factor * running_return
        return_to_go.append(running_return)
    return_to_go.reverse()
    return return_to_go
