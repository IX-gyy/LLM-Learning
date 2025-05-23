from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot


# 选择量化算法
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8), # 平滑算法 平滑度设为0.8
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]), # GPTQ量化
]

# 应用量化
oneshot(
    model="models", # 模型的路径
    dataset="open_platypus", # 数据集(网上加载)
    recipe=recipe,
    output_dir="TinyLlama-1.1B-Chat-v1.0-INT8", # 量化结果输出路径
    max_seq_length=2048,
    num_calibration_samples=512,
)
