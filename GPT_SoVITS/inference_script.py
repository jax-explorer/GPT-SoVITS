from inference_base import get_tts_wav, change_sovits_weights, change_gpt_weights
import wave
import numpy as np
import soundfile as sf

def create_wav_file(file_path, audio_data, sampling_rate):
    sf.write(file_path, audio_data, sampling_rate)

    # # 打开WAV文件
    # with wave.open(file_path, 'w') as wav_file:
    #     # 设置WAV文件的参数
    #     wav_file.setnchannels(1)  # 单声道
    #     wav_file.setsampwidth(2)  # 16位
    #     wav_file.setframerate(sampling_rate)
    #     wav_file.setnframes(len(audio_data))
    #     # 将音频数据写入WAV文件
    #     wav_file.writeframes(np.array(audio_data, dtype=np.int16).tobytes())

def inference(ref_wav_path, prompt_text, prompt_language, text, text_language, result_path, sovits_path, gpt_path):
    change_sovits_weights(sovits_path)
    change_gpt_weights(gpt_path)
    getTTSWavGenerator = get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, "Slice once every 4 sentences", top_k=5, top_p=1, temperature=1)
    for simple_rate, audio_data in getTTSWavGenerator:
        create_wav_file(file_path=result_path, audio_data=audio_data, sampling_rate=simple_rate)