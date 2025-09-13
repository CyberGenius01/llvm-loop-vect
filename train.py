# train.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import LoopVectorizeEnv
import os

ir_path = "benchmark.ll"  # produce with clang -S -emit-llvm
env = DummyVecEnv([lambda: LoopVectorizeEnv(ir_path, single_loop_mode=True)])
model = PPO("MlpPolicy", env, verbose=1, n_steps=64)

# simple training loop: step through each loop and train for short episodes
model.learn(total_timesteps=5000)
model.save("ppo_loop_vectorize")
