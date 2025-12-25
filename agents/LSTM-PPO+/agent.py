import os
import time
import torch
import numpy as np
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import onpolicy_trainer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from env import Environment as env  # 你的自定义环境

# ===================== 环境与目录 =====================
model_name = "RPPO_Tianshou"
en = env()

# Tianshou 建议使用向量化环境
train_envs = DummyVectorEnv([lambda: env() for _ in range(1)])  # 单环境向量化
test_envs = DummyVectorEnv([lambda: env() for _ in range(1)])

logdir = en.get_logdir()
models_dir = f"models/{model_name}"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

writer = SummaryWriter(log_dir=logdir)

# ===================== 网络定义 =====================
# 假设环境状态为 env.observation_space.shape，动作空间为 env.action_space.n 或 shape
obs_shape = en.observation_space.shape or en.observation_space.n
action_shape = en.action_space.n if hasattr(en.action_space, 'n') else en.action_space.shape[0]

# 定义带 LSTM 的策略网络
class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_shape, action_shape, lstm_hidden_size=256, net_arch=[64,64]):
        super().__init__()
        self.net_arch = net_arch
        self.lstm_hidden_size = lstm_hidden_size

        # 策略和价值网络共享前置 MLP
        layers = []
        input_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
        for size in net_arch:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.ReLU())
            input_dim = size
        self.mlp = nn.Sequential(*layers)

        # LSTM 层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_size, batch_first=True)

        # Actor & Critic heads
        self.actor = nn.Linear(lstm_hidden_size, action_shape)
        self.critic = nn.Linear(lstm_hidden_size, 1)

    def forward(self, obs, hidden=None, mask=None):
        # obs shape: [batch, obs_dim]
        x = self.mlp(obs)
        x = x.unsqueeze(1)  # [batch, seq_len=1, feature]
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.squeeze(1)
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        return logits, value, hidden

# ===================== PPO Policy =====================
net = ActorCriticLSTM(obs_shape, action_shape, lstm_hidden_size=256, net_arch=[64,64])
optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

policy = PPOPolicy(
    net,
    optimizer,
    dist_fn=torch.distributions.Categorical if hasattr(en.action_space, 'n') else torch.distributions.Normal,
    discount_factor=0.99,
    max_grad_norm=0.5,
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    reward_normalization=False,
    gae_lambda=0.95,
    recompute_advantage=True
)

# ===================== Collector =====================
train_collector = Collector(policy, train_envs, VectorReplayBuffer(size=2000, buffer_num=1))
test_collector = Collector(policy, test_envs)

# ===================== 训练循环 =====================
TIMESTEPS = 2000
total_rounds = 50
start_time = time.time()
global_step = 0

for i in range(1, total_rounds + 1):
    print(f"第{i}个回合")
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=1,  # 每轮训练一次
        step_per_epoch=TIMESTEPS,
        repeat_per_collect=1,
        batch_size=128,
        episode_per_test=1,
        writer=writer,
        log_interval=1
    )
    global_step += TIMESTEPS
    torch.save(policy.state_dict(), f'{models_dir}/{global_step}.pth')
    print(f"Model saved at {models_dir}/{global_step}")

end_time = time.time()
elapsed_time_hours = (end_time - start_time) / 3600
print(f"程序运行时间：{elapsed_time_hours:.6f} 小时")
