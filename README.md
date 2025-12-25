# Code-for-Adaptive-Function-Service-Auto-Scaling
This repository contains the  implementation of the paper **"Adaptive Function Service Auto-Scaling for Serverless Computing via Deep Recurrent Reinforcement Learning"**.

We propose a **Deep Recurrent Reinforcement Learning (DRRL)** framework, specifically **LSTM-PPO+**, to address the auto-scaling challenges in Serverless Computing (e.g., OpenFaaS). This repository also includes the implementation of baseline methods (LSTM-PPO from *Agarwal et al.* and standard PPO) for comparative analysis.

The codebase is organized as follows:

```text
FsCode-master/
├── agents/                       # Core algorithms implementation
│   ├── LSTM-PPO+/                # [Proposed Method] Our enhanced DRRL algorithm
│   │   ├── agent.py              # Training script (Tianshou framework)
│   │   ├── env.py                # Training environment (Gaussian Reward & P95 Latency)
│   │   ├── test_agent.py         # Testing/Evaluation script
│   │   └── test_env.py           # Testing environment
│   │   
│   ├── LSTM-PPO/                 # [Baseline] Reproduction of Agarwal et al. (IEEE TSC 2024)
│   │   ├── agent_Agarwal.py      # Baseline training script
│   │   ├── env_Agarwal.py        # Baseline environment (Linear Reward, No P95)
│   │   ├── test_agent_Agarwal.py # Baseline testing script
│   │   └── test_env_Agarwal.py   # Baseline testing environment
│   │   
│   └── PPO/                      # [Baseline] Standard PPO algorithm
│       └── ...                   # Standard PPO agent and environment
│
├── workload-generator/           # Workload generation tools
│   ├── request_generator.py      # Script to send HTTP requests to the gateway
│   ├── request_count_raw.json    # Original Azure Trace data
│   └── request_count_boosted.json# Augmented Trace data (used for training)
│
└── kube/                         # Kubernetes configuration storage
    └── config                    # (Downloaded automatically from the master node)

```

🛠️ Prerequisites
Before running the experiments, ensure your infrastructure meets the following requirements:

1.Kubernetes Cluster: Version v1.23+ recommended.

2.OpenFaaS: Deployed in the Kubernetes Cluster.

3.Prometheus (Dual Setup):

 - Prometheus A: For function-level metrics (RPS, Latency).

 - Prometheus B: For Pod/Container-level resource metrics (CPU, Memory).

4.Target Function: Deploy a function named StressFunc (used for stress testing) on OpenFaaS.
