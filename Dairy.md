# 2025/4/3

### 在跑readme里的实例代码时出现了第一个环境问题
>cannot import name 'oneshot' from 'llmcompressor'

### 我们可以用下面的代码检查库里的函数
``` python
import llmcompressor
print(dir(llmcompressor))
```
### 发现确实是没有对应的函数
### 我们可以查阅这段评论里尝试找到解决方法
>https://github.com/vllm-project/llm-compressor/issues/1257


---
2025/4/20
### 文件传输经验
> 本机Windows，从本地传输大量文件到云服务器的经验  
> *LordShark*一早上快被文件传输整疯了，希望自己的经验对uu们有帮助，不用再浪费时间在查找文件传输办法上了/(ㄒoㄒ)/  

打开PowerShell，输入`scp -r -P 1234 "C:\...\...\" username@cloud_ip:/.../` <br>
-r 代表递归复制文件和文件夹 <br>
-P 1234代表指定端口1234【如果租的机子名称一样，则一定要指定端口号，否则连接失败】<br>
"C:\...\"代表本地要传输的文件夹位置 <br>
username 云服务器用户名（一般为root）<br>
cloud_ip 云服务器ip（可以在ssh登录处获取）<br>
后面是云服务器上要存放的位置（如果文件夹不存在，需要手动创建文件夹）<br>
如果是第一次连接会向你确认，输入"yes"即可<br>
然后输入密码，就是ssh连接时的密码（密码隐形，屏幕上不可见，输入后直接回车即可）<br>
就可以愉快地传输数据了[]~ (￣▽￣) ~* <br>

---
2025/4/21
### conda管理Python虚拟环境
> 在开发不同的项目时，不同的项目可能依赖于不同版本的库。通过创建虚拟环境，可以为每个项目创建一个独立的环境  
> 在虚拟环境中安装/删除包 不会影响全局环境(可以随便乱造bushi)  

打开终端，输入`conda create -n myenv` 可以创建一个名为myenv的虚拟环境(体现为一个文件夹)<br>
想要指定具体python版本可以在后面跟python=3.8<br>
Linux激活虚拟环境 输入`source activate myenv`<br>
Windows激活虚拟环境 输入`conda activate myenv`<br>
看见终端前面会有`env`出现，代表已经进入虚拟环境 <br>
然后就可以随便造 : ) <br>
关闭环境`conda deactivate`<br>
查看已创建的环境`conda env list` <br>
删除已创建的环境`conda remove -n myenv --all`<br>
查询环境中有哪些包`conda list`<br>

---
2025/4/25
### screen实现后台运行
> 为了解决跑代码的时候无法关闭页面狂玩的问题(bushi) *LordShark*查了一下如何在后台运行程序  
> 大模型的训练和评测可能花费大量时间 而且as we all know 屏幕黑了或者关闭Jupyter都会触发终端自动挂断  
> 那么如何在服务器开着、电脑关闭的情况下运行代码呢？（可以趁此机会狂玩手机bushi） 

打开终端，输入`conda list`检查一下有没有下载screen<br>
没有下载就输入`pip install screen`<br>
输入`screen`看到弹出页面/终端页面更换，就是已经打开screen，按下空格或回车跳过阅读文档<br>
已经打开screen工具，程序在里面运行就是在后台运行的<br>
要回到原终端的话，按下`Ctrl+A+D`<br>
要回到程序：输入`screen -r`<br>
结束后台程序：screen页面输入`exit`<br>


---
2025/4/27
### github上面的实例代码完整跑通的流程


*环境和包*
```bash
!pip install llmcompressor
!pip install vllm
!pip install huggingface_hub
```

*实际运行代码*
>从镜像网站上下载模型到本地，解决直接跑代码时远程获取模型链接超时的问题。
```python
import os
import subprocess

# 设置镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    subprocess.run([
        "huggingface-cli", 
        "download", 
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        "--local-dir", 
        "./models"
    ])
except Exception as e:
    print(f"下载模型时出错: {e}")
```
>运用本地的模型进行量化
```python
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot


# 选择量化算法
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]

# 应用量化
oneshot(
    model="models",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="TinyLlama-1.1B-Chat-v1.0-INT8",
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

---
2025/4/22
## 评估库的使用参考知乎：
>https://zhuanlan.zhihu.com/p/671235487
### 最主要的问题是评估使用的数据集无法被直接下载
~（在量化时的量化数据集其实也无法直接使用镜像下载，但是后来莫名的能够通过直连下载了）~

## 将模型更换为Qwen-7B时出现量化到最后一步突然崩塌的情况

### 目前定性为因为qwen -7B所需的内存过大，最后需要生成并保存量化后的文件，但是系统内存只有45G，内存溢出后jupyter崩溃。
### 可以采用分片储存的方式来解决这个问题

2025/4/24
## 在用终端运行评估时，如果出现一整屏的warning，那么很有可能是远程的数据集链接不上了，可以输入一下设置镜像的指令
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---
4.25
## nohup使用方法
类似screen 但是screen是切换终端页面 nohup在后台运行结束后生成output.log日志  
nohup运行python代码：
```
nohup python test.py > output.log 2>&1 &
```
查看output.log：  
`tail -f output.log`
`less output.log`

---
5.10
# 关于部分量化方法
## SmoothQuant（平滑量化）：为激活值设计的前处理方法
1. *背景问题*：激活值分布不均导致量化误差大  
在标准量化中，模型中某些 激活值分布跨度特别大，而量化操作必须把整个分布压缩进一个有限的离散数值空间（如 int8 的 [-128, 127]），这会导致信息丢失严重，精度下降。  

2. *核心思想*：激活-权重平衡（Activation-Weight Balancing）  
SmoothQuant 的目标是：将激活值的跨度压缩，使其更适合 int8 量化，同时通过权重的反向调整保持输出不变。  
```
原始矩阵乘法：
Y = X * W^T

SmoothQuant重写：
Y = (X / s) * (W * s)^T
```

## GPTQ
1. *背景问题*：大模型无法微调怎么办？  
对于 GPT-3、LLaMA 之类的千亿参数级别模型，我们很难用传统方法进行量化微调（QAT），这就需要一种不依赖训练的量化方法（Post-Training Quantization, PTQ）。  

2. *核心思想*：最小化量化误差（Error-aware Quantization）  
GPTQ 的创新点在于：不是直接量化整个矩阵，而是以列为单位，逐列选择最佳量化方式，并最小化误差对下游的影响。对每个列块逐个选择量化值，同时补偿误差。
