import pyaudio
import torch
import whisper
import numpy as np
import threading

class ASRService:
    def __init__(self, model_size="medium", rate=16000, channels=1, format=pyaudio.paInt16, frames_per_buffer=1024):
        self.audio = pyaudio.PyAudio()
        self.frames = []  # 用来存放音频数据
        self.recording = False  # 用于标记是否正在录音
        self.stream = None  # 用于存储音频流对象
        self.rate = rate  # 采样率
        self.channels = channels  # 声道数
        self.format = format  # 音频格式
        self.frames_per_buffer = frames_per_buffer  # 每个缓冲区的帧数
        # 检查是否有可用的 GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model on {self.device}...")
        # 在 GPU 上加载 Whisper 模型
        self.model = whisper.load_model(model_size, device=self.device)

    def start_recording(self):
        """开始录音。"""
        self.frames = []
        try:
            self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.frames_per_buffer)
            self.recording = True
            print("Recording started...")
            self._record_in_background()
        except Exception as e:
            print(f"Error starting recording: {e}")

    def _record_in_background(self):
        """启动后台线程录音。"""
        def record():
            while self.recording:
                try:
                    data = self.stream.read(self.frames_per_buffer)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Error during recording: {e}")
                    break

        self.recording_thread = threading.Thread(target=record)
        self.recording_thread.start()

    def stop_recording(self):
        """停止录音。"""
        self.recording = False
        if self.recording_thread.is_alive():
            self.recording_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("Recording stopped.")

    def audio_to_text(self):
        """将录制的音频转录为文本。"""
        try:
            print("Transcribing audio...")
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max  # Normalize to float32

            # 使用 Whisper 模型进行转录
            result = self.model.transcribe(audio_data, fp16=True)  # Enable fp16 for GPU inference
            return result["text"]
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def close(self):
        """关闭 PyAudio 实例，释放资源。"""
        self.audio.terminate()

if __name__ == "__main__":
    asr_service = ASRService()

    try:
        while True:
            command = input("Enter command('f': start recording, 's': stop recording and transcribe, 'q': quit): ")
            
            # 开始录音
            if command == 'q':
                break
            
            elif command == 'f':
                print("Listening...")
                asr_service.start_recording()
            
            elif command == 's':
                asr_service.stop_recording()
                transcribed_text = asr_service.audio_to_text()
                print(f"User:{transcribed_text}")
                
                # # Agent response
                # if transcribed_text:
                #     response = 
            
            else:
                print("Unknown command, please try again.")

    finally:
        # 关闭 PyAudio 实例
        asr_service.close()