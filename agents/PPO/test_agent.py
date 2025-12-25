import os
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector
from tianshou.policy import PPOPolicy

from test_env import Environment as env

MODEL_NAME = "PPO"

single_env = env()
logdir = single_env.get_logdir()
models_dir = f"models/{MODEL_NAME}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

en = DummyVectorEnv([lambda: env()])
en.reset()

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

obs_shape = single_env.observation_space.shape
action_shape = single_env.action_space.n
net = Net(obs_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=3e-4)

model = PPOPolicy(
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

MODEL = 100000
model_path = f'models/{MODEL_NAME}/{MODEL}.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

collector = Collector(policy=model, env=en)

episodes = 500
rewards_history = []
writer = tf.summary.create_file_writer(logdir)

for ep in range(episodes):
    print(f"第{ep}个episode")
    obs = en.reset()
    done = False
    ep_rew = 0

    while not done:
        action = model.forward(torch.tensor(obs, dtype=torch.float32))[0]
        action = torch.argmax(action, dim=1).numpy()
        obs, rewards, done, infos = en.step(action)
        ep_rew += rewards[0]

    rewards_history.append(ep_rew)
    avg_score = np.mean(rewards_history[-100:])

    with writer.as_default():
        tf.summary.scalar('reward summary', data=avg_score, step=ep)
        tf.summary.scalar('episodic_reward', data=ep_rew, step=ep)
        writer.flush()
