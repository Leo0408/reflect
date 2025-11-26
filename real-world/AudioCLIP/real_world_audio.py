import os
import sys
# 获取当前文件所在目录（AudioCLIP目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 AudioCLIP 目录添加到 Python 路径
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# 将 real-world 目录添加到 Python 路径（用于导入 constants）
# 需要确保它在 main 目录之前，所以使用 insert(0) 并确保优先级
real_world_dir = os.path.dirname(current_dir)  # 上一级目录就是 real-world
if real_world_dir not in sys.path:
    sys.path.insert(0, real_world_dir)
import numpy as np
import torch
from model import AudioCLIP
from utils.transforms import ToTensor1D
from moviepy.editor import AudioFileClip
import itertools

# 直接从 real-world/constants.py 导入，使用绝对路径避免冲突
import importlib.util
constants_path = os.path.join(real_world_dir, 'constants.py')
spec = importlib.util.spec_from_file_location("real_world_constants", constants_path)
real_world_constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_world_constants)
real_world_sound_map = real_world_constants.real_world_sound_map

torch.set_grad_enabled(False)

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
# derived from ESResNeXt
SAMPLE_RATE = 44100
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

LABELS = ['gas stove burner turns on', "water runs in sink", "cracking sound"]

# 构建模型文件的绝对路径
model_path = os.path.join(current_dir, 'assets', MODEL_FILENAME)
# 如果模型文件不存在，尝试其他可能的位置
if not os.path.exists(model_path):
    # 尝试在 AudioCLIP 目录下查找
    alt_path = os.path.join(real_world_dir, 'AudioCLIP', 'assets', MODEL_FILENAME)
    if os.path.exists(alt_path):
        model_path = alt_path
    else:
        # 如果都不存在，设置为 None（将在首次使用时加载或跳过）
        model_path = None
        print(f"⚠️  警告: 模型文件 {MODEL_FILENAME} 未找到")
        print(f"   预期路径: {os.path.join(current_dir, 'assets', MODEL_FILENAME)}")
        print(f"   如果不需要音频处理功能，可以忽略此警告")

# 延迟初始化 AudioCLIP（仅在需要时加载）
aclp = None
audio_transforms = ToTensor1D()

def _get_audioclip():
    """延迟加载 AudioCLIP 模型"""
    global aclp
    if aclp is None:
        if model_path and os.path.exists(model_path):
            aclp = AudioCLIP(pretrained=model_path)
            print(f"✅ AudioCLIP 模型已加载: {model_path}")
        else:
            # 如果模型文件不存在，使用未预训练的模型
            aclp = AudioCLIP(pretrained=False)
            print("⚠️  使用未预训练的 AudioCLIP 模型（功能可能受限）")
    return aclp

def extract_audio_from_video(audio_path, volume_thresh):
    def to_ranges(iterable):
        iterable = sorted(set(iterable))
        for _, group in itertools.groupby(enumerate(iterable),
                                            lambda t: t[1] - t[0]):
            group = list(group)
            yield group[0][1], group[-1][1]

    pred_sounds = {}
    input_audio = AudioFileClip(audio_path).set_fps(SAMPLE_RATE)
    duration = int(input_audio.duration)
    frames_w_sound = []
    for cur_time in range(0, duration, 4):
        if cur_time+4 > duration:
            break
        subaudio = input_audio.subclip(cur_time, cur_time+4)
        max_volume = subaudio.max_volume()
        if max_volume > volume_thresh:
            frames_w_sound += range(cur_time, cur_time+4)

    sound_ranges = list(to_ranges(frames_w_sound))
    print("sound ranges:", sound_ranges)
    
    filtered_sound_ranges = []
    for sound_range in sound_ranges:
        if sound_range[1] - sound_range[0] < 4:
            subaudio = input_audio.subclip(sound_range[0], sound_range[1])
            max_volume = subaudio.max_volume()
            print("max volume:", max_volume)
            if max_volume > 0.5:
                filtered_sound_ranges.append(sound_range)
        else:
            filtered_sound_ranges.append(sound_range)
    print("filtered sound ranges:", filtered_sound_ranges)

    # plt.figure()
    # f, axarr = plt.subplots(len(sound_ranges), 1, figsize=(20, 16))
    tracks, max_volumes = [], []
    for idx, frame_range in enumerate(filtered_sound_ranges):
        print(f"FRAME {frame_range}")
        if frame_range[0] + 5 > duration:
            break
        sub_audio = input_audio.subclip(frame_range[0], frame_range[0]+5)
        max_volumes.append(sub_audio.max_volume())
        signal = sub_audio.to_soundarray().astype(np.float32)
        tracks.append(signal)

    return tracks, filtered_sound_ranges, max_volumes

def format_time(total_seconds):
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f'{minutes:.0f}:{seconds:02.0f}'

def get_sound_events(audio_path, volume_thresh=0.05):
    # 延迟加载模型
    model = _get_audioclip()
    
    tracks, sound_ranges, max_volumes = extract_audio_from_video(audio_path, volume_thresh)
    if len(tracks) == 0:
        return {}
    elif len(tracks) == 1:
        tracks = [tracks[0], tracks[0]]
    audios = torch.stack([audio_transforms(track.reshape(1, -1)) for track in tracks])
    if "makeCoffee" in audio_path:
        labels = LABELS + ["coffee machine turns on"]
    else:
        labels = LABELS
    texts = [[label] for label in labels]

    ((audio_features, _, _), _), _ = model(audio=audios)
    ((_, _, text_features), _), _ = model(text=texts)

    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

    scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
    logits_audio_text = scale_audio_text * audio_features @ text_features.T

    sound_events = {}
    confidence = logits_audio_text.softmax(dim=1)
    for idx in range(len(sound_ranges)):
        max_volume = max_volumes[idx]
        if sound_ranges[idx][1] - sound_ranges[idx][0] < 8 and max_volume > 0.8:
            sound_events[tuple(sound_ranges[idx])] = "something drops on the ground"
            continue
        conf_values, ids = confidence[idx].topk(1)
        if labels[ids] in real_world_sound_map:
            sound_events[tuple(sound_ranges[idx])] = real_world_sound_map[labels[ids]]
        else:
            sound_events[tuple(sound_ranges[idx])] = labels[ids]
        # print(conf_values, (format_time(sound_ranges[idx][0]), format_time(sound_ranges[idx][1])), LABELS[ids])

    print(sound_events)
    return sound_events

if __name__ == '__main__':
    audio_path = 'real_world/data/makeCoffee3/videos/0/0/audio.wav'
    get_sound_events(audio_path, volume_thresh=0.03)