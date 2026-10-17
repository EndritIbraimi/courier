"""
play_random.py -- measure the BASELINE: how well does a random robot score?

    python3 play_random.py     -> plays 100 games with no graphics

This average is the number the learning robot must beat from week 2.
"""
import numpy as np
from courier_env import CourierEnv

env = CourierEnv()
scores, delivered = [], 0
for game in range(100):
    env.reset(seed=game)              # a seed makes the results repeatable
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    scores.append(info["total_reward"])
    delivered += info["delivered"]

print(f"Games played    : {len(scores)}")
print(f"Average score   : {np.mean(scores):.1f}")
print(f"Best / worst    : {max(scores):.0f} / {min(scores):.0f}")
print(f"Deliveries made : {delivered} of {len(scores)}")
