import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("notebooks/plots", exist_ok=True)
df = pd.read_csv("logs/doodler_eval.csv")

# Learning Curve
plt.plot(df["episode"], df["reward"], label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Learning Curve: Reward Per Episode")
plt.legend()
plt.savefig("notebooks/plots/reward_curve.png", dpi=300)
plt.close()

# Distribution of Scores
plt.hist(df["score"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Distribution of Scores")
plt.savefig("notebooks/plots/score_distribution.png", dpi=300)
plt.close()

# Death vs Time Alive Scatter
plt.scatter(df["time_alive"], df["terminated"], alpha=0.7)
plt.xlabel("Time Alive (s)")
plt.ylabel("Death (1=yes, 0=no)")
plt.title("Deaths vs Time Alive")
plt.savefig("notebooks/plots/death_vs_time_alive.png", dpi=300)
plt.close()

# Coverage of Unique Platforms
plt.bar(df["episode"], df["platforms_visited"])
plt.xlabel("Episode")
plt.ylabel("Unique Platforms Visited")
plt.title("Platform Coverage per Episode")
plt.savefig("notebooks/plots/coverage_platforms.png", dpi=300)
plt.close()
