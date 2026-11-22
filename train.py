"""
train.py -- WEEK 3: train TWO algorithms and compare them.

    python3 train.py --level 2            # train Q-learning and SARSA on level 2
    python3 train.py --level 1 --runs 5   # same on level 1, 5 runs each

WHY SEVERAL RUNS?  Learning has luck in it (random exploring). One run of each
algorithm could be lucky or unlucky, so we train each one RUNS times and
compare the AVERAGES. That makes the comparison fair.

It produces:
  models/level{N}_qlearning.npy and models/level{N}_sarsa.npy  (Watch mode uses them)
  compare_level{N}.png        the learning curves of both algorithms
  results_level{N}.txt        the comparison table
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # draw to a file, no window needed
import matplotlib.pyplot as plt
from courier_env import CourierEnv
from agents import QLearningAgent, SarsaAgent, model_path

EPISODES = {1: 3000, 2: 6000, 3: 30000}          # practice games per run (bigger map = more games)
EPS_START, EPS_END, EPS_DECAY_FRACTION = 1.0, 0.05, 0.7
NAMES = {"qlearning": "Q-learning", "sarsa": "SARSA"}
COLORS = {"qlearning": "#38BDF8", "sarsa": "#FBBF24"}


def train_one(algo, level, episodes, seed):
    """Train one robot for `episodes` games. Returns (robot, score of every game)."""
    np.random.seed(seed)
    env = CourierEnv(level=level)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    agent = QLearningAgent(n_states, n_actions) if algo == "qlearning" else SarsaAgent(n_states, n_actions)
    scores = []
    for ep in range(episodes):
        # epsilon shrinks linearly: explore a lot early, use the notebook later
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / (episodes * EPS_DECAY_FRACTION))
        obs, _ = env.reset(seed=seed * 1_000_000 + ep)
        done = False
        action = agent.act(obs, eps)               # SARSA needs the first move chosen in advance
        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_action = agent.act(next_obs, eps)
            if algo == "qlearning":
                agent.learn(obs, action, reward, next_obs, terminated)
            else:
                agent.learn(obs, action, reward, next_obs, next_action, terminated)
            obs, action = next_obs, next_action
            done = terminated or truncated
        scores.append(info["total_reward"])
    return agent, np.array(scores)


def evaluate(level, choose_action, games=100):
    """Play `games` NEW games (seeds nobody trained on) with exploring switched off."""
    env = CourierEnv(level=level)
    res = []
    for g in range(games):
        obs, _ = env.reset(seed=9_000_000 + g)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(choose_action(obs, env))
            done = terminated or truncated
        res.append(info)
    return (np.mean([r["total_reward"] for r in res]),
            sum(r["delivered"] for r in res),
            np.mean([r["steps"] for r in res]),
            sum(r["dead"] for r in res))             # games that ended with an empty battery


def games_to_profit(scores, window=100):
    """First game where the moving average score is above 0 (robot 'earns' points)."""
    avg = np.convolve(scores, np.ones(window) / window, mode="valid")
    above = np.flatnonzero(avg >= 0)
    return int(above[0]) + window if len(above) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5, help="training runs per algorithm")
    args = ap.parse_args()
    level, runs = args.level, args.runs
    episodes = EPISODES[level]

    curves = {a: [] for a in NAMES}            # score of every game, for every run
    final = {a: [] for a in NAMES}             # test result of every run
    profit = {a: [] for a in NAMES}
    online = {a: [] for a in NAMES}            # score during the last 500 practice games
    for run in range(runs):
        for algo in NAMES:
            agent, scores = train_one(algo, level, episodes, seed=run)
            curves[algo].append(scores)
            final[algo].append(evaluate(level, lambda o, e: agent.act(o)))
            profit[algo].append(games_to_profit(scores))
            online[algo].append(scores[-500:].mean())
            if run == 0:
                agent.save(model_path(level, algo))   # run 0 is the one Watch mode shows
        print(f"run {run + 1}/{runs} done")

    rand = evaluate(level, lambda o, e: e.action_space.sample())

    # ------------------------------------------------------------ table
    battery = CourierEnv(level=level).battery_max is not None      # extra column on battery levels
    lines = [f"LEVEL {level}  ({episodes} practice games per run, {runs} runs per algorithm)",
             f"Test = 100 new games with exploring switched off, averaged over the {runs} runs",
             "Learning score = average score of the last 500 practice games (exploring still ON)", "",
             f"{'':<12}{'Avg score':>12}{'(+/-)':>8}{'Delivered':>12}{'Avg moves':>11}{'Games to profit':>18}{'Learning score':>16}" + (f"{'Battery empty':>15}" if battery else "")]
    for algo in NAMES:
        f = np.array(final[algo])
        p = [x for x in profit[algo] if x is not None]
        ptxt = f"{np.mean(p):.0f}" + ("" if len(p) == runs else f" ({len(p)}/{runs} runs)") if p else "never"
        lines.append(f"{NAMES[algo]:<12}{f[:, 0].mean():>12.1f}{f[:, 0].std():>8.1f}"
                     f"{f[:, 1].mean():>11.0f}%{f[:, 2].mean():>11.1f}{ptxt:>18}{np.mean(online[algo]):>16.1f}"
                     + (f"{f[:, 3].mean():>14.0f}%" if battery else ""))
    lines.append(f"{'Random':<12}{rand[0]:>12.1f}{'':>8}{rand[1]:>11.0f}%{rand[2]:>11.1f}{'-':>18}{'-':>16}" + (f"{rand[3]:>14.0f}%" if battery else ""))
    text = "\n".join(lines)
    print("\n" + text)
    open(f"results_level{level}.txt", "w").write(text + "\n")

    # ------------------------------------------------------------ chart
    window = 100
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0B1220")
    ax.set_facecolor("#111B2E")
    for algo in NAMES:
        sm = np.array([np.convolve(c, np.ones(window) / window, mode="valid") for c in curves[algo]])
        x = np.arange(window, episodes + 1)
        ax.plot(x, sm.mean(0), color=COLORS[algo], linewidth=2.5, label=NAMES[algo])
        ax.fill_between(x, sm.mean(0) - sm.std(0), sm.mean(0) + sm.std(0), color=COLORS[algo], alpha=0.2)
    ax.axhline(rand[0], color="#F87171", linestyle="--", linewidth=2, label=f"Random robot ({rand[0]:.0f})")
    ax.set_title(f"Q-learning vs SARSA  -  Level {level}", color="white", fontsize=18, fontweight="bold")
    ax.set_xlabel("Games played while learning", color="#CBD5E1")
    ax.set_ylabel(f"Score (average of {runs} runs)", color="#CBD5E1")
    ax.tick_params(colors="#94A3B8")
    for side in ax.spines.values():
        side.set_color("#334155")
    ax.grid(color="#1E293B")
    ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")
    fig.tight_layout()
    fig.savefig(f"compare_level{level}.png", dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved compare_level{level}.png and results_level{level}.txt")


if __name__ == "__main__":
    main()
