from llmcompressor.modifiers.smoothquant import SmoothQuantModifier #执行SmoothQuant量化操作的包
from llmcompressor.modifiers.quantization import GPTQModifier # GPTQ量化
from llmcompressor import oneshot # 一次性对模型进行一系列修改的包


# 选择量化算法
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8), # 创建一个SmoothQuant量化修改器实例 平滑强度0.8
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]), #量化方案W8A8(权重量化为8位，激活量化为8位) 量化目标为线性层 忽略对语言模型输出层的量化操作
]

# 应用量化
oneshot(
    model="Qwen",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="Qwen-INT8",
    max_seq_length=2048, #限制输入长度
    num_calibration_samples=512, #指定用于校准的样本数量
)
