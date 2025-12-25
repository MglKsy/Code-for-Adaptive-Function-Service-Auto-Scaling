import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector
from tianshou.policy import PPOPolicy
from test_env import Environment as env  # 你的自定义环境
from tianshou.utils.net.common import Net
from torch import nn

# ===================== 环境与目录 =====================
MODEL_NAME = "RPPO"
MODEL_STEP = 100000  # 加载模型
en = env()
num_envs = 1

logdir = en.get_logdir()
models_dir = f"models/{MODEL_NAME}"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

writer = SummaryWriter(log_dir=logdir)

# 向量化环境
test_envs = DummyVectorEnv([lambda: env() for _ in range(num_envs)])

# ===================== 网络定义 =====================
obs_shape = en.observation_space.shape or en.observation_space.n
action_shape = en.action_space.n if hasattr(en.action_space, 'n') else en.action_space.shape[0]

class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_shape, action_shape, lstm_hidden_size=256, net_arch=[64,64]):
        super().__init__()
        self.net_arch = net_arch
        self.lstm_hidden_size = lstm_hidden_size

        layers = []
        input_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
        for size in net_arch:
            layers.append(nn.Linear(input_dim, size))
            layers.append(nn.ReLU())
            input_dim = size
        self.mlp = nn.Sequential(*layers)

        self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)
        self.actor = nn.Linear(lstm_hidden_size, action_shape)
        self.critic = nn.Linear(lstm_hidden_size, 1)

    def forward(self, obs, state=None, mask=None):
        x = self.mlp(obs)
        x = x.unsqueeze(1)  # [batch, seq_len=1, feature]
        if state is None:
            lstm_out, state = self.lstm(x)
        else:
            lstm_out, state = self.lstm(x, state)
        lstm_out = lstm_out.squeeze(1)
        logits = self.actor(lstm_out)
        value = self.critic(lstm_out)
        return logits, value, state

# ===================== 加载模型 =====================
net = ActorCriticLSTM(obs_shape, action_shape, lstm_hidden_size=256, net_arch=[64,64])
policy = PPOPolicy(
    net,
    optim=None,  # 测试阶段不需要优化器
    dist_fn=torch.distributions.Categorical if hasattr(en.action_space, 'n') else torch.distributions.Normal,
    is_train=False
)
# 加载权重
policy.net.load_state_dict(torch.load(f"{models_dir}/{MODEL_STEP}.pth"))

# ===================== Collector =====================
test_collector = Collector(policy, test_envs)

# ===================== 评估循环 =====================
episodes = 500
rewards_history = []

start_time = time.time()
for ep in range(episodes):
    print(f"第{ep+1}个episode")
    obs = test_envs.reset()
    done = [False for _ in range(num_envs)]
    ep_rew = 0
    hidden_state = None
    while not all(done):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            logits, _, hidden_state = policy.net(obs_tensor, state=hidden_state)
            if hasattr(en.action_space, 'n'):
                dist = torch.distributions.Categorical(logits=logits)
            else:
                dist = torch.distributions.Normal(logits, 1.0)
            action = dist.sample()
        obs, rewards, done, infos = test_envs.step(action.numpy())
        ep_rew += rewards[0]  # 单环境时使用 rewards[0]

    rewards_history.append(ep_rew)
    avg_score = np.mean(rewards_history[-100:])
    writer.add_scalar('reward_summary', avg_score, ep)
    writer.add_scalar('episodic_reward', ep_rew, ep)
    writer.flush()

end_time = time.time()
elapsed_time_hours = (end_time - start_time) / 3600
print(f"程序运行时间：{elapsed_time_hours:.6f} 小时")
