<template>
  <div class="container">
    <a-card title="🎤 语音转写工具" :bordered="false" class="card">
      <template #extra>
        <a-badge :status="isRecording ? 'processing' : (isLoading ? 'warning' : 'success')" :text="statusText" />
      </template>

      <a-spin :spinning="isLoading" tip="正在转写中...">
        <div class="content">
          <div class="status-info">
            <a-alert
              v-if="isRecording"
              message="正在录音中..."
              description="请清晰地说出您要转写的内容"
              type="warning"
              show-icon
              :closable="false"
            />
            <a-alert
              v-else-if="isLoading"
              message="正在转写..."
              description="请稍候，AI正在处理您的音频"
              type="info"
              show-icon
              :closable="false"
            />
            <a-alert
              v-else
              message="准备就绪"
              description="按住下方按钮开始录音，也可以使用空格键进行录音"
              type="success"
              show-icon
              :closable="false"
            />
          </div>

          <div class="controls">
            <a-button
              @mousedown="startRecording"
              @mouseup="stopRecording"
              @mouseleave="stopRecording"
              @touchstart.prevent="startRecording"
              @touchend.prevent="stopRecording"
              :disabled="isLoading"
              :type="isRecording ? 'primary' : 'default'"
              :danger="isRecording"
              size="large"
              class="record-btn"
            >
              <template #icon>
                <template v-if="isLoading">
                  <loading-outlined />
                </template>
                <template v-else-if="isRecording">
                  <sound-outlined />
                </template>
                <template v-else>
                  <audio-outlined />
                </template>
              </template>
              {{ isLoading ? '转写中...' : (isRecording ? '松开结束' : '按住说话') }}
            </a-button>
          </div>

          <div v-if="resultText" class="result-box">
            <a-divider orientation="left">转写结果</a-divider>
            <a-result
              status="success"
              :sub-title="`处理耗时：${processTime}秒`"
            >
              <template #extra>
                <div class="result-text">{{ resultText }}</div>
              </template>
            </a-result>
          </div>
        </div>
      </a-spin>
    </a-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import axios from 'axios';
import { message } from 'ant-design-vue';
import { LoadingOutlined, SoundOutlined, AudioOutlined } from '@ant-design/icons-vue';

const isRecording = ref(false);
const isLoading = ref(false);
const statusText = ref('准备就绪');
const resultText = ref('');
const processTime = ref<number | null>(null);

let audioContext: AudioContext | null = null;
let mediaStream: MediaStream | null = null;
let audioData: Float32Array[] = [];
let scriptProcessor: ScriptProcessorNode | null = null;
let sourceNode: MediaStreamAudioSourceNode | null = null;
let startTime: number = 0;

const API_URL = 'http://localhost:8000/transcribe';

const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array) => {
  for (let i = 0; i < input.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
};

const writeString = (view: DataView, offset: number, string: string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

const encodeWAV = (samples: Float32Array, sampleRate: number) => {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPCM(view, 44, samples);

  return view;
};

const mergeBuffers = (audioDataArray: Float32Array[]) => {
  let totalLength = 0;
  audioDataArray.forEach(buffer => {
    totalLength += buffer.length;
  });
  
  const result = new Float32Array(totalLength);
  let offset = 0;
  audioDataArray.forEach(buffer => {
    result.set(buffer, offset);
    offset += buffer.length;
  });
  
  return result;
};

const handleKeyDown = (e: KeyboardEvent) => {
  // 防止在输入框等元素中触发空格键录音
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
    return;
  }
  
  if (e.code === 'Space' && !e.repeat && !isRecording.value && !isLoading.value) {
    e.preventDefault(); // 防止页面滚动
    startRecording();
  }
};

const handleKeyUp = (e: KeyboardEvent) => {
  if (e.code === 'Space' && isRecording.value) {
    e.preventDefault();
    stopRecording();
  }
};

onMounted(() => {
  document.addEventListener('keydown', handleKeyDown);
  document.addEventListener('keyup', handleKeyUp);
});

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyDown);
  document.removeEventListener('keyup', handleKeyUp);
});

const startRecording = async () => {
  if (isRecording.value || isLoading.value) return;
  
  isRecording.value = true;
  statusText.value = '正在录音...';
  resultText.value = '';
  processTime.value = null;
  audioData = [];
  startTime = Date.now();

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      } 
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    
    scriptProcessor.onaudioprocess = (e) => {
      const channelData = e.inputBuffer.getChannelData(0);
      audioData.push(new Float32Array(channelData));
    };
    
    sourceNode.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);
    
    console.log('录音已开始 (16kHz WAV)');

  } catch (err) {
    console.error('录音失败:', err);
    message.error('无法访问麦克风，请检查权限');
    isRecording.value = false;
    statusText.value = '无法访问麦克风';
    cleanupMedia();
  }
};

const stopRecording = async () => {
  if (!isRecording.value) return;
  
  const recordDuration = Date.now() - startTime;
  
  if (recordDuration < 1000) {
    message.warning('录音时间太短，请至少录制 1 秒');
    isRecording.value = false;
    cleanupMedia();
    return;
  }
  
  isRecording.value = false;
  statusText.value = '正在处理...';
  isLoading.value = true;

  try {
    console.log('audioData 数量:', audioData.length);
    
    if (audioData.length === 0) {
      message.warning('录音失败，请重试');
      statusText.value = '准备就绪';
      cleanupMedia();
      return;
    }

    const mergedAudio = mergeBuffers(audioData);
    console.log('合并后的音频长度:', mergedAudio.length);
    
    const wavData = encodeWAV(mergedAudio, 16000);
    const audioBlob = new Blob([wavData], { type: 'audio/wav' });
    
    console.log('audioBlob 大小:', audioBlob.size, '字节');
    console.log('MIME 类型: audio/wav');
    
    if (audioBlob.size < 10000) {
      message.warning('录音时间太短，请重试');
      statusText.value = '准备就绪';
      cleanupMedia();
      return;
    }
    
    statusText.value = '正在转写...';
    
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.wav');
    
    const response = await axios.post(API_URL, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    
    resultText.value = response.data.text;
    processTime.value = response.data.process_time;
    statusText.value = '转写完成！';
    message.success('转写完成！');
    
  } catch (error) {
    console.error('转写失败:', error);
    message.error('转写失败，请检查后端是否启动');
    statusText.value = '转写失败';
  } finally {
    isLoading.value = false;
    cleanupMedia();
  }
};

const cleanupMedia = () => {
  if (sourceNode && audioContext) {
    try {
      sourceNode.disconnect();
    } catch (e) {}
  }
  if (scriptProcessor && audioContext) {
    try {
      scriptProcessor.disconnect();
    } catch (e) {}
  }
  if (audioContext) {
    try {
      audioContext.close();
    } catch (e) {}
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
  }
  audioContext = null;
  mediaStream = null;
  scriptProcessor = null;
  sourceNode = null;
  audioData = [];
};
</script>

<style scoped>
.container {
  max-width: 700px;
  margin: 50px auto;
  padding: 0 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.card {
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}
.content {
  padding: 20px 0;
}
.status-info {
  margin-bottom: 30px;
}
.controls {
  text-align: center;
  margin: 30px 0;
}
.record-btn {
  height: 60px;
  padding: 0 50px;
  font-size: 18px;
  border-radius: 30px;
  transition: all 0.3s;
}
.result-box {
  margin-top: 30px;
}
.result-text {
  font-size: 18px;
  line-height: 1.8;
  color: #333;
  padding: 20px;
  background-color: #f5f5f5;
  border-radius: 8px;
  text-align: left;
}
</style>