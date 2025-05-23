同文件夹下终端输入:   
其中 output_path 为评估结果输出路径，在评估完成后会产生一个`results.json`文件，标记了评估日期  
打开`result.json`，再打开里面的`result`条目，可以看见对应下面`task`项目的结果的评分  
```
lm_eval --model vllm \
  --model_args "pretrained=TinyLlama-1.1B-Chat-v1.0-INT8/,tokenizer=TinyLlama-1.1B-Chat-v1.0-INT8,dtype=auto" \
  --tasks toxigen,squadv2 \
  --batch_size auto \
  --output_path eval_out/openbuddy13b \
  --use_cache eval_cache
```
