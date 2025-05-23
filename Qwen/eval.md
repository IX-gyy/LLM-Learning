
```
lm_eval --model vllm \
  --model_args "pretrained=Qwen-INT8/,tokenizer=Qwen-INT8,dtype=auto" \
  --tasks toxigen,squadv2 \
  --device cuda:0 \
  --batch_size auto \
  --output_path eval_out/openbuddy13b \
  --use_cache eval_cache
```
