"""
train.py -- WEEK 2: teach the robot with Q-learning.

    python3 train.py

What it does:
  1. Plays thousands of games. At first the robot explores (random moves),
     later it uses what it learned (epsilon goes down).
  2. Saves the learned notebook to models/level1_q.npy  (Watch mode uses it).
  3. Draws training_curve.png: score going up = the robot is learning.
  4. Tests the trained robot against the random robot on 100 NEW games.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # draw to a file, no window needed
import matplotlib.pyplot as plt
from courier_env import CourierEnv
from agents import QLearningAgent, model_path

LEVEL = 1
EPISODES = 3000                        # how many games the robot plays while learning
EPS_START, EPS_END = 1.0, 0.05         # exploration: from 100% random down to 5%
EPS_DECAY_FRACTION = 0.7               # reach the minimum after 70% of the games

env = CourierEnv(level=LEVEL)
agent = QLearningAgent(env.observation_space.n, env.action_space.n)

# ---------------------------------------------------------------- training
scores = []
for ep in range(EPISODES):
    # epsilon shrinks linearly: explore a lot early, exploit later
    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / (EPISODES * EPS_DECAY_FRACTION))
    obs, _ = env.reset(seed=ep)
    done = False
    while not done:
        action = agent.act(obs, eps)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.learn(obs, action, reward, next_obs, terminated)   # <- the learning
        obs = next_obs
        done = terminated or truncated
    scores.append(info["total_reward"])
    if (ep + 1) % 500 == 0:
        print(f"Games {ep + 1:>5}: average score of the last 500 games = {np.mean(scores[-500:]):6.1f}")

agent.save(model_path(LEVEL))
print(f"\nSaved the learned notebook to {model_path(LEVEL)}")


# ---------------------------------------------------------------- testing
def evaluate(choose_action, games=100):
    """Play `games` NEW games (seeds the robot never trained on)."""
    results = []
    for g in range(games):
        obs, _ = env.reset(seed=100000 + g)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(choose_action(obs))
            done = terminated or truncated
        results.append(info)
    return (np.mean([r["total_reward"] for r in results]),
            sum(r["delivered"] for r in results), games)

rand_avg, rand_del, n = evaluate(lambda o: env.action_space.sample())
ai_avg, ai_del, _ = evaluate(lambda o: agent.act(o))       # epsilon = 0: no exploring
print("\n--- RESULT (100 new games each) ---")
print(f"Random robot  : average score {rand_avg:6.1f}, deliveries {rand_del}/{n}")
print(f"Trained robot : average score {ai_avg:6.1f}, deliveries {ai_del}/{n}")

# ---------------------------------------------------------------- chart
window = 50
smooth = np.convolve(scores, np.ones(window) / window, mode="valid")  # moving average
fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0B1220")
ax.set_facecolor("#111B2E")
ax.plot(range(window, EPISODES + 1), smooth, color="#38BDF8", linewidth=2.5, label="Learning robot")
ax.axhline(rand_avg, color="#F87171", linestyle="--", linewidth=2, label=f"Random robot ({rand_avg:.0f})")
ax.set_title("The robot learns to deliver", color="white", fontsize=18, fontweight="bold")
ax.set_xlabel("Games played while learning", color="#CBD5E1")
ax.set_ylabel("Score (moving average)", color="#CBD5E1")
ax.tick_params(colors="#94A3B8")
for side in ax.spines.values():
    side.set_color("#334155")
ax.grid(color="#1E293B")
ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")
fig.tight_layout()
fig.savefig("training_curve.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved the chart to training_curve.png")
