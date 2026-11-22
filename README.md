# Courier

A delivery robot learns to pick up and deliver packages (reinforcement learning).
Semester project for Introduction to Artificial Intelligence.

## Run (macOS)
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python3 main.py             # the game: play yourself or watch the AI
    python3 play_random.py      # baseline: average score of 100 random games
    python3 train.py --level 3  # train Q-learning AND SARSA and compare them (levels 1, 2 or 3)

## Files
- main.py        menu, play mode, watch-the-AI mode
- courier_env.py game rules (no graphics)
- renderer.py    all drawing and animation
- agents.py      the AI brains: random robot, Q-learning robot, SARSA robot
- train.py       trains both algorithms several times and makes compare_level{N}.png + results_level{N}.txt
- models/        the saved, trained robot (the "notebook")
- assets/music.wav  background music (made by make_music.py); press M to mute
- levels.py      level list, unlocked week by week

## Progress
- Week 1: level 1, random robot, baseline score, menu and graphics
- Week 2: Q-learning robot trained on level 1, learning chart, Watch mode uses the trained robot (T switches to random)
- Week 2 bonus: background music
- Week 3: level 2 (7x7 map with walls), second algorithm (SARSA), Q-learning vs SARSA comparison
- Week 4: level 3 (battery + charging pads), Q-learning vs SARSA again, battery meter in the game
