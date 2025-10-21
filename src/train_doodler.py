"""
train_doodler.py
-----------------
Training script for PPO or DQN agent in the Doodler Jump environment.
- Select algorithm with --algo ppo or --algo dqn
- Parses CLI arguments for configuration
- Builds training and evaluation environments
- Configures reward shaping and callbacks
- Saves the trained model and logs metrics

Usage Examples:
    python train_doodler.py --algo ppo --timesteps 1000000 --reward_persona survivor
    python train_doodler.py --algo dqn --timesteps 500000 --reward_persona aggressive
"""
import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from envs.doodler_env import DoodlerEnv

# CONSTANTS
DEFAULT_TIMESTEPS = 100_000
DEFAULT_SAVE_FREQ = 100_00
DEFAULT_EVAL_FREQ = 25_000
DEFAULT_ENVS_COUNT = 4
DEFAULT_SEED = 42
DEFAULT_LOGDIR = "logs/doodler"
DEFAULT_MODELDIR = "models/doodler"
DEFAULT_REWARD_PERSONA = "aggressive"

# Simplified reward configs focusing on what matters
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

def make_env(seed=42, render_mode=None, reward_config=None):
    '''Factory function for creating a monitored DoodlerEnv instance.'''

    def _init():
        env = DoodlerEnv(
            render_mode=render_mode, 
            seed=seed, 
            reward_config=reward_config,
            frame_skip=1  # Increased for better exploration
        )
        return Monitor(env)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn"])
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)  # Increased
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--logdir", type=str, default=DEFAULT_LOGDIR)
    parser.add_argument("--modeldir", type=str, default=DEFAULT_MODELDIR)
    parser.add_argument("--save_freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--eval_freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--n_envs", type=int, default=DEFAULT_ENVS_COUNT)  # Parallel environments
    parser.add_argument("--reward_persona", type=str, default=DEFAULT_REWARD_PERSONA, choices=REWARD_PERSONAS.keys())
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    reward_config = REWARD_PERSONAS[args.reward_persona]

    print(f"Training with algorithm: {args.algo.upper()}")
    print(f"Training with reward persona: {args.reward_persona}")
    print(f"Reward weights: {reward_config['weights']}")

    # Create vectorized environments for faster training
    env = DummyVecEnv([make_env(seed=args.seed + i, reward_config=reward_config) 
                    for i in range(args.n_envs)])
    
    eval_env = DoodlerEnv(seed=args.seed + 1000, reward_config=reward_config, render_mode=None)
    eval_env = Monitor(eval_env)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
            n_steps=2048,  # Increased for better value estimates
            batch_size=256,
            gamma=0.97,  # Slightly reduced for more immediate rewards
            gae_lambda=0.95,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.03,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])  # Larger network
            )
        )
    else: #DQN
        model = DQN(
            "MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=args.logdir,
            seed=args.seed,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=64,
            tau=0.1,
            gamma=0.98,
            train_freq=4,
            target_update_interval=10_000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[256, 256])
        )

    logger = configure(args.logdir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,  # Adjust for parallel envs
        save_path=args.modeldir,
        name_prefix=f"{args.algo}_{args.reward_persona}_doodler"
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.modeldir,
        log_path=args.logdir,
        eval_freq=args.eval_freq // args.n_envs,  # Adjust for parallel envs
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    
    try:
        model.learn(
            total_timesteps=args.timesteps, 
            callback=[checkpoint_cb, eval_cb], 
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        model.save(os.path.join(args.modeldir, f"{args.algo}_{args.reward_persona}_checkpoint"))
        
    save_path = os.path.join(args.modeldir, f"{args.algo}_{args.reward_persona}_final")
    model.save(save_path)
    print(f"\n✓ Training Complete! Model saved to {save_path}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()