import av
import numpy as np


def extract_keyframes(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    frame_index = 0
    keyframe_indices, keyframes = [], []
    for frame in container.decode(stream):
        if frame_index == 0 or frame.key_frame:
            frame_data = frame.to_ndarray(format="rgb24")
            keyframes.append(frame_data)
            keyframe_indices.append(frame_index)
        frame_index += 1
    container.close()
    return keyframe_indices, np.array(keyframes)


def sample_keyframes(keyframe_indices, keyframes, sampling_ratio):
    num_samples = int(len(keyframe_indices) * sampling_ratio)
    samples = np.linspace(0, len(keyframe_indices) - 1, num_samples, dtype=int)
    sampled_keyframes = keyframes[samples]
    sampled_keyframe_indices = [keyframe_indices[i] for i in samples]
    return sampled_keyframe_indices, sampled_keyframes
