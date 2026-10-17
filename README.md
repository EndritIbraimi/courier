# Courier

A delivery robot learns to pick up and deliver packages (reinforcement learning).
Semester project for Introduction to Artificial Intelligence.

## Run (macOS)
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python3 main.py             # the game: play yourself or watch the AI
    python3 play_random.py      # baseline: average score of 100 random games

## Files
- main.py        menu, play mode, watch-the-AI mode
- courier_env.py game rules (no graphics)
- renderer.py    all drawing and animation
- agents.py      the AI brains (random for now)
- levels.py      level list, unlocked week by week

## Progress
- Week 1: level 1, random robot, baseline score, menu and graphics
