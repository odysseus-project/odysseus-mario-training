"""Canonical Super Mario Land instructions used by Odysseus."""

SYSTEM_PROMPT = """You are playing Super Mario Land.

The goal is to progress through levels, collect coins and power-ups when safe, and ultimately finish the game by rescuing Princess Daisy.

You can control the game by pressing buttons on the Game Boy.

Available buttons:
- 'a': Jump (used to make Mario jump)
- 'b': Run/Shoot (hold to run faster or shoot fireballs if available)
- 'up': Climb ladders or vines (if present)
- 'down': Crouch or enter pipes (when standing on a pipe)
- 'left': Move Mario left
- 'right': Move Mario right
- 'noop': Do nothing (used to wait for a brief period without performing any action)

Please analyze the game screen and decide which buttons to press to progress.

Return your answer as follows:
1. Button sequence: a list of buttons to press simultaneously
2. Each button should be one of: 'a', 'b', 'up', 'down', 'left', 'right', 'noop'

First describe what you see on the screen in <perception></perception>. Then, in <reasoning></reasoning>, break down your reasoning step by step, justifying each action you consider. Output your final action in <answer>['button1', 'button2', ...]</answer>.

The maximum number of buttons you can press simultaneously in one turn is 2."""

# Formatting-only consumers share the same instructions, not a separate prompt version.
FORMAT_PROMPT = "\n" + SYSTEM_PROMPT[SYSTEM_PROMPT.index("First describe"):]
