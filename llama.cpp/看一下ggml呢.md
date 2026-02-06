# ggml
据说要了解llama.cpp的底层原理，需要先从GGML库开始，那么我们直接开搞  
[ggml简介传送门](https://huggingface.co/blog/zh/introduction-to-ggml)  
- `ggml_context`: 一个装载各个对象（如张量、计算图、其他数据）的容器
- `ggml_cgraph`: 计算图的表示，将要传给后端的“计算执行顺序”
- `ggml_backend`: 执行计算图的接口，有多种类型：CPU（默认）、CUDA、Metal等等
- `ggml_backend_buffer_type`: 表示一种缓存，可以理解为连接到每个`ggml_backend`的一个*内存分配器*。假设需要在GPU上执行计算，则需要通过一个`buffer_type`（通常缩写为`buft`）去在GPU上分配内存
- `ggml_backend_buffer`: 表示一个通过`buft`分配的缓存，一个缓存可以存储多个张量
- `ggml_gallocr`: 表示一个给计算图分配内存的分配器，可以给计算图中的张量进行高效的内存分配
- `ggml_backend_sched`: 一个调度器，使得多种后端可以并发使用，在处理大模型火多GPU推理是，实现跨硬件平台地分配计算任务（如CPU+GPU混合运算）。它还能自动将GPU不支持的算子转移到CPU上，确保最优的资源调用和兼容性。

## 一个ggml流程
1. 分配一个 ggml_context 来存储张量数据
2. 分配张量并赋值
3. 为矩阵乘法运算创建一个 ggml_cgraph
4. 执行计算
5. 获取计算结果
6. 释放内存并退出

## 使用后端的流程
1. 初始化 ggml_backend
2. 分配 ggml_context 以保存张量的 metadata (此时还不需要直接分配张量的数据)
3. 为张量创建 metadata (也就是形状和数据类型)
4. 分配一个 ggml_backend_buffer 用来存储所有的张量
5. 从内存 (RAM) 中复制张量的具体数据到后端缓存
6. 为矩阵乘法创建一个 ggml_cgraph
7. 创建一个 ggml_gallocr 用以分配计算图
8. 可选: 用 ggml_backend_sched 调度计算图
9. 运行计算图
10. 获取结果，即计算图的输出
11. 释放内存并退出
