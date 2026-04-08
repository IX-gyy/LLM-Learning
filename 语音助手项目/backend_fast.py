import os
import io
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from funasr import AutoModel

app = FastAPI(title="语音转写工具")

# 允许跨域（方便前端调试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# 1. 全局加载模型（启动时只加载一次，保持待命）
# --------------------------
print("="*50)
print("正在加载 paraformer-zh 模型，请稍候...")
start_load_time = time.time()

# 模型配置：非流式模式，关闭不必要的功能以提速
model = AutoModel(
    model="paraformer-zh",
    disable_update=True,  # 禁用自动检查更新
    use_itn=True          # 开启文本逆规范化（可选，让结果更自然）
)

end_load_time = time.time()
print(f"✅ 模型加载完成！耗时：{end_load_time - start_load_time:.2f}秒")
print("="*50)

# --------------------------
# 2. 转写接口：接收音频，立即返回结果
# --------------------------
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    start_time = time.time()
    
    temp_file_path = None
    try:
        audio_bytes = await file.read()
        
        if len(audio_bytes) < 100:
            print(f"⚠️ 音频文件太小: {len(audio_bytes)} 字节")
            return {
                "text": "录音时间太短，请重新录制",
                "process_time": 0
            }
        
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'wav'
        temp_file_path = f"temp_audio_{int(time.time())}.{file_extension}"
        
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_bytes)
        
        file_size = os.path.getsize(temp_file_path)
        print(f"📁 音频文件大小: {file_size} 字节")
        
        if file_size < 100:
            print(f"⚠️ 音频文件太小: {file_size} 字节")
            return {
                "text": "录音时间太短，请重新录制",
                "process_time": 0
            }
        
        result = model.generate(input=temp_file_path)
        
        text = ""
        if result and "text" in result[0]:
            text = result[0]["text"]
        
        end_time = time.time()
        process_time = end_time - start_time
        
        print(f"📝 转写完成 | 耗时: {process_time:.2f}秒 | 结果: {text}")
        
        return {
            "text": text,
            "process_time": round(process_time, 2)
        }
        
    except Exception as e:
        print(f"❌ 转写失败: {str(e)}")
        return {
            "text": "转写失败，请检查音频格式",
            "process_time": 0
        }
        
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)