"""
eval_doodler.py
----------------
Evaluation script for trained PPO models on the Doodler Jump environment.
- Loads a trained model
- Runs multiple episodes and collects performance metrics
- Optionally renders gameplay
- Saves results to a CSV for analysis

Usage Example:
    python eval_doodler.py --model_path models/doodler/aggressive_doodler_final.zip --episodes 10
"""

import argparse, os, csv, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
from stable_baselines3 import PPO
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.doodler_env import DoodlerEnv

# Reward Configs
REWARD_PERSONAS = {
    "survivor": {
        "type": "survivor",
        "weights": {
            "height_progress": 10.0,
            "death": -100.0,
            "landing": 2.0,
        }
    },
    "aggressive": {
        "type": "aggressive",
        "weights": {
            "height_progress": 20.0,
            "death": -50.0,
            "landing": 1.0,
        }
    }
}

def run_episode(model, render=False, seed=None, reward_config=None):
    '''Runs a single evaluation episode.

        Returns:
            dict: A dictionary containing episode metrics like reward, score, height climbed, etc.
    '''
    env = DoodlerEnv(render_mode="human" if render else None, seed=seed, reward_config=reward_config, frame_skip=1)
    obs, info = env.reset()

    done = trunc = False
    episode_reward, steps, score = 0.0, 0, 0
    max_height = env.HEIGHT

    while not(done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(int(action))
        episode_reward += reward
        score = info.get("score", 0)
        max_height = min(max_height, info.get("max_height", 600))

        if render:
            env.render()
        steps += 1

    env.close()
    return {
        "epsiode": 0,
        "reward": episode_reward, 
        "steps": steps, 
        "score": score,
        "max_height": max_height,
        "height_climbed": env.HEIGHT - max_height, 
        "terminated": int(done), 
        "truncated": int(trunc)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/doodler/aggressive_doodler_final.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--csv_out", type=str, default="logs/doodler_eval.csv")
    parser.add_argument("--reward_persona", type=str, default="aggressive", choices=REWARD_PERSONAS.keys())
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    reward_config = REWARD_PERSONAS[args.reward_persona]
    print(f"Evaluating model with reward persona: {args.reward_persona}")
    
    try:
        model = PPO.load(args.model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model. Ensure the file is valid. Details: {e}")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)

    results = []

    for i in range(1, args.episodes + 1):
        print(f"Running Episode {i}/{args.episodes}...")
        metrics = run_episode(model, render=bool(args.render), reward_config=reward_config)
        metrics["episode"] = i
        results.append(metrics)
    
    mean_reward = np.mean([r["reward"] for r in results])
    std_reward = np.std([r["reward"] for r in results])
    mean_score = np.mean([r["score"] for r in results])
    mean_height = np.mean([r["height_climbed"] for r in results])
    max_score = max([r["score"] for r in results])
    max_height = max([r["height_climbed"] for r in results])

    print(f"Episodes: {len(results)}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean score: {mean_score:.2f}")
    print(f"Max score: {max_score}")
    print(f"Mean height: {mean_height:.0f}px")
    print(f"Max height: {max_height:.0f}px")

    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Metrics saved to {args.csv_out}")

if __name__ == "__main__":
    main()