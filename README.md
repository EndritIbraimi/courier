# Courier

A delivery robot learns to pick up and deliver packages (reinforcement learning).
Semester project for Introduction to Artificial Intelligence.

## Run (macOS)
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python3 main.py             # the game: play yourself or watch the AI
    python3 play_random.py      # baseline: average score of 100 random games
    python3 train.py            # week 2: teach the robot (Q-learning), makes training_curve.png

## Files
- main.py        menu, play mode, watch-the-AI mode
- courier_env.py game rules (no graphics)
- renderer.py    all drawing and animation
- agents.py      the AI brains: random robot and Q-learning robot
- train.py       trains the robot and draws the learning chart
- models/        the saved, trained robot (the "notebook")
- levels.py      level list, unlocked week by week

## Progress
- Week 1: level 1, random robot, baseline score, menu and graphics
- Week 2: Q-learning robot trained on level 1, learning chart, Watch mode uses the trained robot (T switches to random)
