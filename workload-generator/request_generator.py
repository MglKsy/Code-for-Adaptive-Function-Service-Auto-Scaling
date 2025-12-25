import json
import subprocess
import numpy as np
from time import sleep
from concurrent.futures import ThreadPoolExecutor

FUNCTION_URL = "http://192.168.70.31:31112/function/StressFunc"
hey_path = r"C:\Users\ta\hey.exe"


def func(a=1):
    command = [hey_path, "-n", str(a), "-c", str(a), "-o", "csv", FUNCTION_URL]  # Replace with your target URL
    subprocess.run(command, shell=True)


# 读取不稳定的调用数据并为我们的容量进行归一化
try:
    with open('request_count-boosted.json') as f:
        data = [int(i) for i in json.load(f)]
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading data: {e}")
    data = []  # 提供默认值
d = [int(i) for i in data]

# 设置随机种子以复现结果
seed = 29
np.random.seed(seed)

# 使用线程池来并发请求
max_threads = 10
executor = ThreadPoolExecutor(max_threads)

# 运行长时间模拟大于14天的测试
for _ in range(20):
    for i in d:  # 每个qq 时间间隔的请求数量，即 30 秒内的请求
        total_time = 30
        try:
            # 平均到达时间（lambda）
            average_inter_arrival_time = total_time / i
            # 生成到达时间间隔，即它们将在哪些秒数到达
            times = np.random.poisson(lam=average_inter_arrival_time, size=i)
            print("times:", times)
            # 每个请求生成并等待，如果需要
            for k in range(0, i):
                inter = times[k]
                executor.submit(func, 1)  # 提交任务给线程池
                sleep(inter)
        except Exception as e:
            # i = 0，即此时间段没有请求
            average_inter_arrival_time = 0
            times = [0]

        times = sum(times)
        if times < 30:
            sleep(30 - times)  # 确保每个周期总时间为 30 秒
