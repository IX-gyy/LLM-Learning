import os
import subprocess

# 设置镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用 hf-mirror.com 镜像

try:
    subprocess.run([
        "huggingface-cli", 
        "download", 
        "Qwen/Qwen3-4B", #选择Qwen3-4B
        "--local-dir", 
        "./Qwen"
    ])
except Exception as e:
    print(f"下载模型时出错: {e}")
