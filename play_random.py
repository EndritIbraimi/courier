"""
play_random.py -- measure the BASELINE: how well does a random robot score?

    python3 play_random.py        -> level 1
    python3 play_random.py 2      -> level 2
"""
import sys
import numpy as np
from courier_env import CourierEnv

level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
env = CourierEnv(level=level)
scores, delivered = [], 0
for game in range(100):
    env.reset(seed=game)              # a seed makes the starting squares repeatable
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    scores.append(info["total_reward"])
    delivered += info["delivered"]

print(f"Level {level}")
print(f"Games played    : {len(scores)}")
print(f"Average score   : {np.mean(scores):.1f}")
print(f"Best / worst    : {max(scores):.0f} / {min(scores):.0f}")
print(f"Deliveries made : {delivered} of {len(scores)}")
