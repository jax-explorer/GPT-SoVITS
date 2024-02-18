from interence_base import get_tts_wav
import wave
import os
import numpy as np

SoVITS_weight_root="SoVITS_weights"
GPT_weight_root="GPT_weights"
exp_name = "jax_clone_voice"
gpt_path = GPT_weight_root + "/" + exp_name + "-e15.ckpt"
sovits_path = SoVITS_weight_root + "/" + exp_name + "_e8_s120.pth"

def create_wav_file(file_path, audio_data, sampling_rate):
    # 打开WAV文件
    with wave.open(file_path, 'w') as wav_file:
        # 设置WAV文件的参数
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sampling_rate)
        wav_file.setnframes(len(audio_data))

        # 将音频数据写入WAV文件
        wav_file.writeframes(np.array(audio_data, dtype=np.int16).tobytes())

def interence(srt_content, audio_list_dir):
    subs = pysrt.from_string(srt_content)
    for i, sub in enumerate(subs):
        wav_name = f"{i+1}.wav"
        ref_wav_path = audio_list_dir + "/" + wav_name
        chinese_text = sub.text.split('\n')[1]
        english_text = sub.text.split('\n')[0]
        prompt_text = chinese_text
        prompt_language = "中文"
        text = english_text
        text_language = "英文"
        getTTSWavGenerator = get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language)
        for simple_rate, audio_data in getTTSWavGenerator:
            wav_output_dir = "data/output_list"
            file_path = wav_output_dir + "/" + wav_name
            os.makedirs(wav_output_dir, exist_ok=True)
            create_wav_file(file_path=file_path, audio_data=audio_data, sampling_rate=simple_rate)


current_working_directory = os.getcwd()
inp_wav_dir = current_working_directory + "/" + "data/list"


srt_path = "data/subtitle.srt"
with open(srt_path, 'r') as file:
      data_dict = json.load(file)
srt_content = data_dict['srt_content']


interence(srt_content=srt_content, audio_list_dir=inp_wav_dir)