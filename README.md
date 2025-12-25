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

1. Infrastructure
Kubernetes Cluster: Version v1.23+ recommended.

OpenFaaS: Deployed in the Kubernetes Cluster.

Prometheus (Dual Setup):

Prometheus A: For function-level metrics (RPS, Latency).

Prometheus B: For Pod/Container-level resource metrics (CPU, Memory).

Target Function: Deploy a function named StressFunc (used for stress testing) on OpenFaaS.


2. Python Environment
We recommend using Anaconda to create a virtual environment:

Bash

conda create -n faas-rl python=3.9
conda activate faas-rl
Install the required dependencies:

Bash

pip install numpy pandas gymnasium torch tianshou tensorflow kubernetes prometheus-api-client paramiko requests
(Note: tensorflow is used primarily for TensorBoard logging).

⚙️ Configuration (Crucial Step)
To run this code in your environment, you MUST update the hardcoded configuration in the environment files.

Please verify and modify agents/LSTM-PPO+/env.py, agents/LSTM-PPO+/test_env.py, and the corresponding files in the baseline folders.

Parameters to Update:
Cluster Connection (getConfig function): Update the SSH credentials to allow the script to fetch the Kubernetes config file from your master node.

Python

vm_ip = "192.168.70.31"      # Your K8s Master Node IP
username = "root"            # SSH Username
password = "YOUR_PASSWORD"   # SSH Password
OpenFaaS & Prometheus URLs:

Python

gateway_url = "[http://192.168.70.31:31112](http://192.168.70.31:31112)"        # OpenFaaS Gateway
self.prom_a_url = "[http://192.168.70.31:30090](http://192.168.70.31:30090)..." # Prometheus A (Function Metrics)
self.prom_b_url = "[http://192.168.70.32:30090](http://192.168.70.32:30090)..." # Prometheus B (Resource Metrics)
Deployment Name:

Python

deployment_name = "StressFunc" # Must match your OpenFaaS function name
🚀 How to Run
Step 1: Start Workload Generation
You need to simulate traffic to drive the environment state changes.

Bash

cd workload-generator
# This script reads request_count_boosted.json and sends requests to the gateway
python request_generator.py
Tip: Keep this script running in a separate terminal window.

Step 2: Train the Agent (LSTM-PPO+)
To train our proposed method:

Bash

cd agents/LSTM-PPO+
python agent.py
Logs: Training logs are saved in logs/.

Models: Checkpoints are saved in models/RPPO_Tianshou/.

Monitoring: Use TensorBoard to visualize training progress.

Bash

tensorboard --logdir=logs/
Step 3: Evaluation & Testing
Load a trained model and evaluate it in the test environment:

Open test_agent.py and ensure MODEL_STEP points to your desired checkpoint (e.g., 100000).

Run the test script:

Bash

python test_agent.py
Step 4: Run Baselines
To reproduce the comparative experiments:

Agarwal et al. (LSTM-PPO Baseline):

Bash

cd agents/LSTM-PPO
python agent_Agarwal.py       # Train
python test_agent_Agarwal.py  # Test
Standard PPO:

Bash

cd agents/PPO
python agent.py
