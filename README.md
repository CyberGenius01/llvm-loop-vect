# 🚀 LLVM + Reinforcement Learning for Loop Vectorization

## 🌟 Overview
Modern compilers like LLVM rely on **fixed heuristics** to optimize code, including loop vectorization. While these heuristics work in many cases, they often **miss optimal opportunities** for diverse program structures.  

This project demonstrates how **Reinforcement Learning (RL)** can dynamically guide loop vectorization, enabling **adaptive, data-driven compiler optimizations**.  
The system integrates a **custom LLVM pass** with a **Python-based RL environment**, where the agent learns to make decisions based on **loop features** and **runtime feedback**.

---

## ✨ Features
- 📝 **LLVM Pass for Loop Analysis**: Extracts key loop features like depth, instruction count, and memory operations.  
- 🤖 **RL Environment**: Gym-style Python environment modeling loop optimization as a reinforcement learning problem.  
- 🧠 **PPO-based Agent**: Learns policies to vectorize loops efficiently using Stable Baselines3.  
- 🔄 **End-to-End Pipeline**: From source code → LLVM IR → RL decisions → optimized executable → performance feedback.

---

## 🛠 Pipeline
```
+------------------+
|  C / C++ Source  |
+------------------+
          │
          ▼
+------------------+
|   LLVM IR (.ll)  |
|  clang -emit-llvm|
+------------------+
          │
          ▼
+---------------------------+
|   LoopRLOpt LLVM Pass     |
|  (feature extraction &    |
|   action application)     |
+---------------------------+
          │
          ▼
+---------------------------+
| Python RL Environment     |
|  (Gym, loads features)    |
+---------------------------+
          │
          ▼
+---------------------------+
|      RL Agent             |
|  (PPO decides vectorize) |
+---------------------------+
          │
          ▼
+---------------------------+
| Optimized IR (.ll)        |
| Compile → Executable      |
+---------------------------+
          │
          ▼
+---------------------------+
| Run & Measure Performance |
|  (feedback → reward)     |
+---------------------------+
          │
          ▼
+---------------------------+
|  RL Agent updates policy  |
+---------------------------+
```
- The RL agent **closes the loop**, learning continuously from execution feedback.  

---

## ⚡ Installation

1. **Clone the repository**  
`git clone <repository_url>`  
`cd llvm-rl-vectorize-project`

2. **Build LLVM Pass**  
`mkdir build && cd build`  
`cmake -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm ..`  
`make`

3. **Install Python dependencies**  
`pip install -r requirements.txt`

✅ `requirements.txt` includes:  
`gym==0.26.3`  
`stable-baselines3==2.2.0`  
`torch==2.1.0`  
`torchvision==0.16.1`  
`numpy==1.26.2`  
`pandas==2.1.0`  
`matplotlib==3.8.0`  
`seaborn==0.12.3`  
`pyyaml==6.1`  
`tqdm==4.68.0`

---

## 🚀 Usage

1. **Generate LLVM IR from your benchmark**  
`clang -O0 -emit-llvm -S ../benchmark.c -o benchmark.ll`

2. **Run the custom LLVM pass**  
`opt -load-pass-plugin=./LoopRLOpt.so -passes=loop-rl-opt benchmark.ll -S -o out.ll`

3. **Train RL Agent**  
`python train.py`

4. **Compile optimized IR and run**  
`clang out.ll -o a.out`  
`./a.out`

---

## 📊 Results
- The RL agent learns **adaptive loop vectorization policies**.  
- Benchmarks show **improved runtime** compared to LLVM default heuristics by avoiding unnecessary vectorization.  
- Demonstrates the **feasibility of integrating machine learning into compiler optimization pipelines**.  

---

## 🔮 Future Work
- Extend RL guidance to **other compiler passes** (e.g., inlining, unrolling).  
- Incorporate **advanced RL techniques** (A2C, AlphaZero-style search).  
- Test on **large-scale, real-world workloads** to improve generalization.  

---

## 📫 Contact
- **Author:** Aditya Kumar  
- **GitHub:** [https://github.com/CyberGenius01](https://github.com/CyberGenius01)  
- **Email:** aditya.kumar@example.com
