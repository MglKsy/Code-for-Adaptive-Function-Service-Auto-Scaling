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

Python Dependencies:
```bash
conda create -n faas-rl python=3.9
conda activate faas-rl
pip install numpy pandas gymnasium torch tianshou tensorflow kubernetes prometheus-api-client paramiko requests
```
⚙️ Configuration (Crucial)
You MUST update the hardcoded configuration in the environment files before running any code. Check agents/LSTM-PPO+/env.py (and similar files in other agent folders):
1. Cluster Connection (getConfig function):
```python
vm_ip = "192.168.70.31"      # REPLACE with your K8s Master Node IP
username = "root"            # REPLACE with your SSH Username
password = "YOUR_PASSWORD"   # REPLACE with your SSH Password
```
2.OpenFaaS & Prometheus URLs:
```python
gateway_url = "[http://192.168.70.31:31112](http://192.168.70.31:31112)"        # OpenFaaS Gateway IP:Port
self.prom_a_url = "[http://192.168.70.31:30090](http://192.168.70.31:30090)..." # Prometheus A (Function Metrics)
self.prom_b_url = "[http://192.168.70.32:30090](http://192.168.70.32:30090)..." # Prometheus B (Resource Metrics)
```
🚀 How to Run
Step 1: Start Workload Generation
You need to simulate traffic to drive the environment. Keep this running in a separate terminal.

```bash
cd workload-generator
python request_generator.py
```

Step 2: Train the Agents
Option A: Train Proposed Method (LSTM-PPO+)
```bash
cd agents/LSTM-PPO+
python agent.py
```
 - Models are saved in models/RPPO_Tianshou/.
 - Logs are saved in logs/.

Option B: Train Baselines
```bash
# Agarwal et al. (IEEE TSC 2024)
cd agents/LSTM-PPO
python agent_Agarwal.py

# Standard PPO
cd agents/PPO
python agent.py
```

Step 3: Evaluation & Testing
Load the trained model and evaluate it.

1.Open test_agent.py and ensure MODEL_STEP points to your desired checkpoint (e.g., 100000).

2.Run the test:
```bash
cd agents/LSTM-PPO+
python test_agent.py
```
