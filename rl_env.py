# rl_env.py
import gym
import json
import subprocess
import os
import time
import numpy as np

class LoopVectorizeEnv(gym.Env):
    """
    Action space: for each loop, choose discrete action:
      0 = disable vectorization
      1 = width 2
      2 = width 4
      3 = width 8
    This simplified environment handles one loop at a time or whole-program depending on mode.
    """

    def __init__(self, ir_path, binary_path="./a.out", single_loop_mode=True):
        super().__init__()
        self.ir_path = ir_path
        self.binary_path = binary_path
        self.single_loop_mode = single_loop_mode

        # Load features
        with open("loop_features.json","r") as f:
            self.loops = json.load(f)

        # Flatten features and create observation space
        # For simplicity, use a fixed-size vector per loop (num_blocks, loads, stores, arith, calls, trip_est)
        self.n_loops = len(self.loops)
        self.obs_dim = 6
        # If single_loop_mode, env steps through loops sequentially; else one action per loop together
        if single_loop_mode:
            self.action_space = gym.spaces.Discrete(4)
            self.observation_space = gym.spaces.Box(low=-1e9, high=1e9, shape=(self.obs_dim,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.MultiDiscrete([4]*self.n_loops)
            self.observation_space = gym.spaces.Box(low=-1e9, high=1e9, shape=(self.n_loops*self.obs_dim,), dtype=np.float32)

        # Baseline run for reward normalization
        self.baseline_time = self.run_binary(self.ir_path, "baseline")
        self.current_loop_idx = 0

    def _feature_to_obs(self, loop):
        tri = loop.get("trip_count_est", -1)
        return np.array([loop.get("num_blocks",0), loop.get("num_loads",0), loop.get("num_stores",0),
                         loop.get("num_arith",0), loop.get("num_calls",0), float(tri)], dtype=np.float32)

    def reset(self):
        self.current_loop_idx = 0
        if self.single_loop_mode:
            obs = self._feature_to_obs(self.loops[self.current_loop_idx])
        else:
            allobs = []
            for loop in self.loops:
                allobs.append(self._feature_to_obs(loop))
            obs = np.concatenate(allobs)
        return obs

    def step(self, action):
        # Build actions map
        if self.single_loop_mode:
            idx = self.current_loop_idx
            act_map = {}
            loop_id = self.loops[idx]["loop_id"]
            act_map[loop_id] = self.action_to_metadata(action)
            # write loop_actions.json
            with open("loop_actions.json","w") as f:
                json.dump(act_map, f)
            # Run opt + compile + run
            t = self.run_binary(self.ir_path, "trial")
            reward = self.baseline_time - t  # higher is better (speedup)
            done = True  # one-step episode per loop; you can change to multi-step
            info = {"runtime": t}
            self.current_loop_idx += 1
            obs = self.reset() if not done else np.zeros(self.obs_dim, dtype=np.float32)
            return obs, reward, done, info
        else:
            # multiple loops in one step: action is vector
            act_map = {}
            for idx, a in enumerate(action):
                loop_id = self.loops[idx]["loop_id"]
                act_map[loop_id] = self.action_to_metadata(int(a))
            with open("loop_actions.json","w") as f:
                json.dump(act_map, f)
            t = self.run_binary(self.ir_path, "trial")
            reward = self.baseline_time - t
            done = True
            info = {"runtime": t}
            obs = np.zeros(self.n_loops*self.obs_dim, dtype=np.float32)
            return obs, reward, done, info

    def action_to_metadata(self, action):
        if action == 0:
            return {"disable": True}
        elif action == 1:
            return {"width": 2}
        elif action == 2:
            return {"width": 4}
        elif action == 3:
            return {"width": 8}
        else:
            return {"disable": True}

    def run_binary(self, ir_path, tag):
        """
        Runs: 1) apply LoopRLOpt reading loop_actions.json (if present),
              2) run opt -loop-vectorize,
              3) compile to native binary,
              4) run binary and measure time.

        Requires: clang/opt in PATH.
        """
        # 1) run our pass to (re)generate features + apply actions
        # Note: pass plugin path may vary.
        cmd_pass = ["opt", "-load-pass-plugin=./LoopRLOpt.so", "-passes=loop-rl-opt", ir_path, "-S", "-o", "tmp_opt.ll"]
        # If opt doesn't support plugin loading like this on your LLVM, adjust command to use legacy pass manager:
        try:
            subprocess.check_call(cmd_pass)
        except subprocess.CalledProcessError as e:
            print("opt pass failed:", e)
            raise

        # 2) run loop-vectorize
        cmd_vec = ["opt", "-loop-vectorize", "tmp_opt.ll", "-S", "-o", "tmp_vec.ll"]
        subprocess.check_call(cmd_vec)

        # 3) compile to native binary (clang)
        subprocess.check_call(["clang", "tmp_vec.ll", "-O3", "-o", self.binary_path])

        # 4) run binary and measure time (simple)
        t0 = time.time()
        subprocess.check_call([f"./{self.binary_path}"])
        t1 = time.time()
        elapsed = t1 - t0
        print(f"[{tag}] runtime: {elapsed:.4f}s")
        return elapsed
