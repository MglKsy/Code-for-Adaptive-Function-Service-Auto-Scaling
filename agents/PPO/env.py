import time
import math
import json
import numpy as np
import gymnasium as gym
import requests
from gymnasium import spaces
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
from requests.auth import HTTPBasicAuth
import tensorflow as tf
from datetime import datetime
import os

deployment_name = "StressFunc"
namespace = "openfaas-fn"
gateway_url = "http://192.168.70.31:31112"

model_name = "PPO_MODEL"

# Kubernetes API
config.load_kube_config(config_file="../../kube/config")
scale_api = client.AppsV1Api()


def getConfig():
    vm_ip = "192.168.70.31"  # 替换为你的虚拟机 IP
    username = "root"  # 使用 root 用户
    password = "030228"

    # 远程 Kubeconfig 文件路径
    remote_file = "/root/.kube/config"
    local_file = "../../kube/config"  # 下载到当前目录

    try:
        # 创建 SSH 客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 自动接受密钥
        ssh.connect(vm_ip, username=username, password=password)

        # 打开 SFTP 传输
        sftp = ssh.open_sftp()
        sftp.get(remote_file, local_file)  # 下载 kubeconfig
        sftp.close()

        print(f"文件已下载到本地: {local_file}")

        ssh.close()
    except Exception as e:
        print("SSH 连接失败:", str(e))

class Environment(gym.Env):
    metadata = {'render_modes': ['human', None]}

    def __init__(self, rew_range=(-100, 10000), min_pods=1, max_pods=24) -> None:
        super(Environment, self).__init__()

        self.MIN_PODS = min_pods
        self.MAX_PODS = max_pods
        self.reward_range = rew_range
        self.timestep = 0
        self.episode = 0
        self.loop = 0
        self._last_obs = None
        self._stats_window = 100
        self.reward_history = []
        self.score = 0
        self.sampling_window = 30  # 秒

        # ---------------- Observation / Action ----------------
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, self.MIN_PODS, 0, 0, 0]),
            high=np.array([60, 100, 100, self.MAX_PODS, 2, 2, 2000]),
            shape=(7,),
            dtype=np.float64
        )
        self.action_space = spaces.Discrete(5)
        self._action_to_scale = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}

        # ---------------- 双 Prometheus 初始化 ----------------
        self.prom_a_url = "http://192.168.70.31:30090/api/v1/query"  # 函数级指标
        self.prom_b_url = "http://192.168.70.32:30090/api/v1/query"  # Pod/资源指标
        self.prom_a = PrometheusConnect(url=self.prom_a_url, disable_ssl=True)
        self.prom_b = PrometheusConnect(url=self.prom_b_url, disable_ssl=True)

        self._initial_setup()

    def _initial_setup(self):
        self.func_cpu = 0.1  # 核
        self.func_mem = 256 / 1024  # Gi
        logdir = f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}/{model_name}"
        os.makedirs(logdir, exist_ok=True)
        self.file_writer = tf.summary.create_file_writer(logdir)
        self.file_writer.set_as_default()
        self._reward_file = f'reward_history_{model_name}.json'
        self.logdir = logdir

    def _take_action(self, action):
        try:
            current_pods = scale_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            ).status.ready_replicas
            if current_pods is None:
                current_pods = 0
        except Exception as e:
            current_pods = 0
            print(f"读取 ready pods 出错: {e}")

        scale_value = current_pods + action
        action_feedback = False

        if action < 0:
            if scale_value >= self.MIN_PODS:
                action_feedback = True
                body = {'spec': {'replicas': scale_value}}
                try:
                    _ = scale_api.patch_namespaced_deployment_scale(
                        name=deployment_name,
                        namespace=namespace,
                        body=body
                    ).spec.replicas
                except Exception as e:
                    action_feedback = False
                    print(e)
        elif action == 0:
            action_feedback = True
        else:
            if self.MIN_PODS <= scale_value <= self.MAX_PODS:
                action_feedback = True
                body = {'spec': {'replicas': scale_value}}
                try:
                    _ = scale_api.patch_namespaced_deployment_scale(
                        name=deployment_name,
                        namespace=namespace,
                        body=body
                    ).spec.replicas
                except Exception as e:
                    action_feedback = False
                    print(e)

        return {'action': action, 'action_feedback': action_feedback, 'pods': current_pods, 'scale_value': scale_value}

    # ---------------- prom_query 支持 A/B ----------------
    def prom_query(self, query, prom_type='A'):
        try:
            if prom_type == 'A':
                return self.prom_a.custom_query(query=query)
            else:
                return self.prom_b.custom_query(query=query)
        except Exception as e:
            print(f"Prometheus {prom_type} 查询失败: {e}")
            return []

    def _get_replicas(self):
        """获取当前 Deployment 的副本数"""
        try:
            url = f"{gateway_url}/system/functions"
            response = requests.get(url, auth=HTTPBasicAuth("admin", "RYDbE2zykFAm"))
            if response.status_code == 200:
                for f in response.json():
                    if f["name"] == deployment_name:
                        return f["replicas"]
        except Exception:
            pass
        return 0

    def _get_obs(self):
        avg_execution = 0.0
        throughput = 0.0
        total_requests = 0
        replicas = 0
        avg_cpu = 0.0
        avg_mem = 0.0
        latency = 0.0

        # ---------------- Prometheus A: 函数级指标 ----------------
        try:
            query_avg_exec = f"""
            rate(gateway_functions_seconds_sum{{function_name='{deployment_name}.{namespace}', code='200'}}[30s])
            /
            rate(gateway_functions_seconds_count{{function_name='{deployment_name}.{namespace}', code='200'}}[30s])
            """
            data = self.prom_query(query_avg_exec, prom_type='A')
            if data:
                avg_execution = float(data[0]['value'][1])
                if math.isnan(avg_execution):
                    avg_execution = 0.0

            query_req = f"increase(gateway_function_invocation_total{{function_name='{deployment_name}.{namespace}'}}[30s])"
            data = self.prom_query(query_req, prom_type='A')
            total_requests = sum(int(float(d['value'][1])) for d in data) if data else 0
            throughput = total_requests / 30
        except Exception:
            avg_execution, total_requests, throughput = 0.0, 0, 0.0

        # ---------------- Prometheus B: Pod/资源指标 ----------------
        try:
            cpu_query = f'rate(container_cpu_usage_seconds_total{{pod=~"{deployment_name}.*", namespace="{namespace}"}}[30s])'
            mem_query = f'container_memory_usage_bytes{{pod=~"{deployment_name}.*", namespace="{namespace}"}}'

            cpu_result = self.prom_query(cpu_query, prom_type='B')
            mem_result = self.prom_query(mem_query, prom_type='B')

            replicas = max(1, self._get_replicas())
            cpu_total = sum(float(r['value'][1]) for r in cpu_result if r['metric'].get('container') == deployment_name)
            mem_total_bytes = sum(float(r['value'][1]) for r in mem_result if r['metric'].get('container') == deployment_name)
            mem_total_gib = mem_total_bytes / (1024 ** 3)

            avg_cpu = round((cpu_total / replicas) / self.func_cpu, 4)
            avg_mem = round((mem_total_gib / replicas) / self.func_mem, 4)
        except Exception:
            avg_cpu, avg_mem = 0.0, 0.0

        # ---------------- 延迟 (Prometheus A) ----------------
        try:
            query_latency = f"""
            histogram_quantile(0.95, sum(rate(gateway_functions_seconds_bucket{{function_name='{deployment_name}.{namespace}'}}[30s])) by (le))
            """
            data = self.prom_query(query_latency, prom_type='A')
            if data:
                latency = float(data[0]['value'][1])
                if math.isnan(latency):
                    latency = 200.0
            else:
                latency = 200.0
        except Exception:
            latency = 200.0

        return np.array([avg_execution, throughput, total_requests, replicas, avg_cpu, avg_mem, latency])


    def _calculate_reward(self, obs, metadata={}):
        throughput, requests, replicas, avg_cpu, avg_mem, latency = obs[1], obs[2], obs[3], obs[4], obs[5], obs[6]
        meta_scale_value = metadata.get('scale_value', -1)

        alpha, beta, gamma, delta, zeta = 0.2, 0.8, 0.5, 0.2, 0.1
        optimal_usage = 0.7

        r_th = alpha * (throughput ** 2)

        def resource_reward(usage, weight):
            if usage > 1.0:
                return -weight * (1 + 2.0 * (usage - 1.0))
            elif usage < 0.3:
                return -weight * (optimal_usage - usage)
            else:
                return weight * math.exp(-4 * (usage - optimal_usage) ** 2)

        r_cpu = resource_reward(avg_cpu, beta)
        r_mem = resource_reward(avg_mem, gamma)
        r_latency = delta * (1 - min(1, latency / 200.0))

        cpu_based = math.ceil(avg_cpu / optimal_usage * replicas)
        mem_based = math.ceil(avg_mem / optimal_usage * replicas)
        target_replicas = max(cpu_based, mem_based)

        r_rep = -zeta * abs(replicas - target_replicas)
        if meta_scale_value != replicas:
            r_rep += -0.5

        reward = round(r_th + r_cpu + r_mem + r_latency + r_rep, 3)
        return reward

    def _write_to_board(self, obs, action, rew, info, step, episode):
        with self.file_writer.as_default():
            tf.summary.scalar('avg_execution_time', obs[0], step)
            tf.summary.scalar('throughput', obs[1], step)
            tf.summary.scalar('requests', obs[2], step)
            tf.summary.scalar('replicas', obs[3], step)
            tf.summary.scalar('cpu', obs[4], step)
            tf.summary.scalar('mem', obs[5], step)
            tf.summary.scalar('latency', obs[6], step)
            tf.summary.scalar('episode', episode, step)
            tf.summary.scalar('action', action, step)
            tf.summary.scalar('action_feedback', int(info['action_feedback']), step)
            tf.summary.scalar('n-step_reward', rew, step)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0
        self.loop = 0
        observation = self._get_obs()
        self._last_obs = observation
        return observation, {}

    def step(self, action):
        done = False
        action = self._action_to_scale[action]
        info = self._take_action(action)

        if not info['action_feedback']:
            self._write_to_board(self._last_obs, action, -100, info, self.timestep, self.episode)
            self.timestep += 1
            self.loop += 1
            self.score += -100
            if self.loop >= 10:
                done = True
            next_obs = self._get_obs()
        else:
            time.sleep(self.sampling_window)
            next_obs = self._get_obs()
            reward = self._calculate_reward(next_obs, info)
            self.score += reward
            self._write_to_board(next_obs, action, reward, info, self.timestep, self.episode)

            self.timestep += 1
            if self.timestep % 10 == 0:
                done = True
                self.episode += 1
                self.loop = 0
                self.reward_history.append(self.score)
                with self.file_writer.as_default():
                    tf.summary.scalar('episodic_reward', self.score, self.episode)
                    tf.summary.scalar('mean_reward', np.mean(self.reward_history[-self._stats_window:]), self.episode)
                self.score = 0
                with open(self._reward_file, "w") as outfile:
                    json.dump({'reward_history': self.reward_history, 'last_episode': self.episode}, outfile)

        self._last_obs = next_obs
        if info['action_feedback']:
            reward = self._calculate_reward(next_obs, info)
        else:
            reward = -100
        return next_obs, reward, done, False, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
