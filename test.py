from ASR.ASR_component import ASRService
from RAG.rag_llm import NewsSearcherRAG
from TTS.ChatTTS.tts import TTSService
from playsound import playsound  # 添加播放功能

def main():
    asr_service = ASRService()
    rag_service = NewsSearcherRAG()
    tts_service = TTSService()
    
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
                
                # Agent response
                if transcribed_text:
                    response = rag_service.rag_response(transcribed_text)
                    print("Response:", response)
                    tts_service.text_to_audio(response)
                    
                    # 播放生成的音频文件
                    print("Playing response audio...")
                    playsound(f"output[{tts_service.cnt-1}].wav")
            else:
                print("Unknown command, please try again.")
            

    finally:
        # 关闭 PyAudio 实例
        asr_service.close()

if __name__ == "__main__":
    main()
