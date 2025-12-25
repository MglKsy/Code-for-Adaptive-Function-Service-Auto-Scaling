import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer

from env import Environment as env  # 你的环境

model_name = "PPO"

en = env()
logdir = en.get_logdir()
models_dir = f"models/{model_name}"

# 创建目录
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

writer = SummaryWriter(log_dir=logdir)

# 向量化环境，这里只用 1 个环境
train_envs = DummyVectorEnv([lambda: env()])

# 定义策略网络，尽量保留 SB3 MlpPolicy 风格
class Net(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, action_shape)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs, state=None, info=None):
        x = self.fc(obs)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value, state

obs_shape = en.observation_space.shape
action_shape = en.action_space.n
net = Net(obs_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=3e-4)

policy = PPOPolicy(
    actor=net,
    critic=net,
    optim=optim,
    dist_fn=torch.distributions.Categorical,
    discount_factor=0.99,
    max_grad_norm=0.5,
    eps_clip=0.2,
    vf_coef=0.5,
    ent_coef=0.0
)

# 定义回放缓冲和收集器
buffer = VectorReplayBuffer(buffer_num=1, total_size=20000, buffer_size=2000)
collector = Collector(policy, train_envs, buffer)

TIMESTEPS = 2000
total_rounds = 50
start_time = time.time()
global_step = 0

for i in range(1, total_rounds+1):
    print(f"第{i}个回合")
    # 每轮训练 TIMESTEPS 步
    result = onpolicy_trainer(
        policy=policy,
        train_collector=collector,
        test_collector=None,
        max_epoch=1,
        step_per_epoch=TIMESTEPS,
        repeat_per_collect=1,
        batch_size=128,
        writer=writer,
        log_interval=1
    )
    global_step += TIMESTEPS
    # 保存模型
    torch.save(policy.state_dict(), f'{models_dir}/{global_step}.pth')
    print(f"Model saved at {models_dir}/{global_step}')

end_time = time.time()
elapsed_time_hours = (end_time - start_time) / 3600.0
print(f"程序运行时间：{elapsed_time_hours:.6f} 小时")
