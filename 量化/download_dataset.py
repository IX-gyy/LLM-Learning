
from datasets import load_dataset

# 设置本地缓存路径（可选）
custom_cache_dir1 = "./datasets/mmlu"
ds = load_dataset("cais/mmlu", "all", cache_dir=custom_cache_dir1)

# 设置本地缓存路径
custom_cache_dir2 = "./datasets/humaneval"
ds = load_dataset("openai/openai_humaneval", cache_dir=custom_cache_dir2)

custom_cache_dir3 = "./datasets/medmcqa"
ds = load_dataset("openlifescienceai/medmcqa", cache_dir=custom_cache_dir3)

custom_cache_dir4 = "./datasets/commonsense_qa"
ds = load_dataset("tau/commonsense_qa", cache_dir=custom_cache_dir4)

custom_cache_dir4 = "./datasets/race"
ds = load_dataset("ehovy/race", "all", cache_dir=custom_cache_dir4)

'''

from datasets import load_dataset

# 设置本地缓存路径
custom_cache_dir = "./datasets/gsm8k"
ds = load_dataset("openai/gsm8k", "main", cache_dir=custom_cache_dir)
'''
