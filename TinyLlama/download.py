import os
import subprocess

# 设置镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    subprocess.run([
        "huggingface-cli", 
        "download", 
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", #下载TinyLlama模型
        "--local-dir", # 下载到本地
        "./models" # 本地路径
    ])
except Exception as e:
    print(f"下载模型时出错: {e}")
