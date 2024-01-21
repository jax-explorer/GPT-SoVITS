import pysrt
from pydub import AudioSegment
import json


def merge_audio_with_srt(audio_list_dir, srt_content):
    # Load srt file
    subs = pysrt.from_string(srt_content)

    # Create a new empty audio segment
    combined_audio = AudioSegment.empty()

    last_end_time = 0

    # Merge audio files based on srt timeline
    for i, sub in enumerate(subs):
        # Load the audio file corresponding to the subtitle index
        audio_file = f"{audio_list_dir}/{i+1}.wav"
        audio_segment = AudioSegment.from_wav(audio_file)

        gap_time = start_time - last_end_time 

        if gap_time > 0:
            combined_audio += AudioSegment.silent(duration=gap_time)

        # Calculate the duration of the subtitle
        start_time = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
        end_time = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
        last_end_time = end_time
        duration = end_time - start_time

        # Append silence if the audio segment is shorter than the subtitle duration
        # if len(audio_segment) > duration:
        #     audio_segment = audio_segment.speedup(playback_speed=len(audio_segment) / duration)
        # else:
        #     audio_segment = audio_segment.speeddown(playback_speed=duration / len(audio_segment))
        # combined_audio += audio_segment

        # Append the audio segment to the combined audio
        combined_audio += audio_segment

    # Export the combined audio to a file
    combined_audio.export(f"{audio_list_dir}/combined_audio.wav", format="wav")    


wav_output_dir = "data/output_list"
srt_path = "data/subtitle.srt"
with open(srt_path, 'r') as file:
      data_dict = json.load(file)
srt_content = data_dict['srt_content']

merge_audio_with_srt(audio_list_dir=wav_output_dir, srt_content=srt_content)