import os
import sys

CheckpointsDir = os.path.join("/workspace/MuseTalk", "models")
musetalk_path = os.path.abspath("/workspace/MuseTalk")
if musetalk_path not in sys.path:
    sys.path.append(musetalk_path)
import time
import pdb
import re
import random

import gradio as gr
import numpy as np
import subprocess

from huggingface_hub import snapshot_download
import requests

import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import shutil
import gdown
import imageio
import ffmpeg
from moviepy.editor import *
from transformers import WhisperModel


from MuseTalk.musetalk.utils.blending import get_image
from MuseTalk.musetalk.utils.face_parsing import FaceParsing
from MuseTalk.musetalk.utils.audio_processor import AudioProcessor
from MuseTalk.musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
from MuseTalk.musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder, get_bbox_range

import subprocess

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)

def download_model():
    # 检查必需的模型文件是否存在
    required_models = {
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/unet.pth",
        "MuseTalk": f"{CheckpointsDir}/musetalkV15/musetalk.json",
        "SD VAE": f"{CheckpointsDir}/sd-vae/config.json",
        "Whisper": f"{CheckpointsDir}/whisper/config.json",
        "DWPose": f"{CheckpointsDir}/dwpose/dw-ll_ucoco_384.pth",
        "SyncNet": f"{CheckpointsDir}/syncnet/latentsync_syncnet.pt",
        "Face Parse": f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth",
        "ResNet": f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
    }
    
    missing_models = []
    for model_name, model_path in required_models.items():
        if not os.path.exists(model_path):
            missing_models.append(model_name)
    
    if missing_models:
        # 全用英文
        print("The following required model files are missing:")
        for model in missing_models:
            print(f"- {model}")
        print("\nPlease run the download script to download the missing models:")
        if sys.platform == "win32":
            print("Windows: Run download_weights.bat")
        else:
            print("Linux/Mac: Run ./download_weights.sh")
        sys.exit(1)
    else:
        print("All required model files exist.")

download_model()
def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


@torch.no_grad()
def inference(audio_path, video_path, bbox_shift, extra_margin=10, parsing_mode="jaw", 
              left_cheek_width=90, right_cheek_width=90, progress=gr.Progress(track_tqdm=True)):
    # Set default parameters, aligned with inference.py
    args_dict = {
        "result_dir": './results/output', 
        "fps": 25, 
        "batch_size": 8, 
        "output_vid_name": '', 
        "use_saved_coord": False,
        "audio_padding_length_left": 2,
        "audio_padding_length_right": 2,
        "version": "v15",  # Fixed use v15 version
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width
    }
    args = Namespace(**args_dict)

    # Check ffmpeg
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    
    # Create temporary directory
    temp_dir = os.path.join(args.result_dir, f"{args.version}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set result save path
    result_img_save_path = os.path.join(temp_dir, output_basename)
    crop_coord_save_path = os.path.join(args.result_dir, "../", input_basename+".pkl")
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name == "":
        output_vid_name = os.path.join("/workspace/brain_rot_video_gen", "output"+".mp4")
    else:
        output_vid_name = os.path.join(temp_dir, args.output_vid_name)
        
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(temp_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        # Read video
        reader = imageio.get_reader(video_path)

        # Save images
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
        
    ############################################## extract audio feature ##############################################
    # Extract audio features
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, 
        device, 
        weight_dtype, 
        whisper, 
        librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
        
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    bbox_shift_text = get_bbox_range(input_img_list, bbox_shift)
    
    # Initialize face parser
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    ############################################## inference batch by batch ##############################################
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )
    res_frame_list = []
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        audio_feature_batch = pe(whisper_batch)
        # Ensure latent_batch is consistent with model weight type
        latent_batch = latent_batch.to(dtype=weight_dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        y2 = y2 + args.extra_margin
        y2 = min(y2, frame.shape[0])
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
            continue
        
        # Use v15 version blending
        combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=args.parsing_mode, fp=fp)
            
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        
    # Frame rate
    fps = 25
    # Output video path
    output_video = 'temp.mp4'

    # Read images
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
        

    # Save video
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

    input_video = './temp.mp4'
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Read video
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # Get original video frame rate
    reader.close() # Otherwise, error on win11: PermissionError: [WinError 32] Another program is using this file, process cannot access. : 'temp.mp4'
    # Store frames in list
    frames = images
    
    print(len(frames))

    # Load the video
    video_clip = VideoFileClip(input_video)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)
    audio_clip_duration = audio_clip.duration

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip).subclip(0, audio_clip_duration)

    # Write the output video
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)

    os.remove("temp.mp4")
    #shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")
    return output_vid_name,bbox_shift_text


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae, unet, pe = load_all_model(
    unet_model_path=os.path.join(CheckpointsDir, "musetalkV15/unet.pth"), 
    vae_type="sd-vae",
    unet_config=os.path.join(CheckpointsDir, "musetalkV15/musetalk.json"),
    device=device
)


ffmpeg_path = "/workspace/miniconda3/envs/MuseTalk/bin/ffmpeg"
weight_dtype = torch.float32
# Move models to specified device
pe = pe.to(device)
vae.vae = vae.vae.to(device)
unet.model = unet.model.to(device)

timesteps = torch.tensor([0], device=device)

# Initialize audio processor and Whisper model
audio_processor = AudioProcessor(feature_extractor_path=os.path.join(CheckpointsDir, "whisper"))
whisper = WhisperModel.from_pretrained(os.path.join(CheckpointsDir, "whisper"))
whisper = whisper.to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)

def check_video(video):
    if not isinstance(video, str):
        return video # in case of none type
    # Define the output video file name
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # Add the output prefix to the file name
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # Combine the directory path and the new file name
    output_video = os.path.join('./results/input', output_file_name)


    # read video
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # get fps from original video

    # conver fps to 25
    frames = [im for im in reader]
    target_fps = 25
    
    L = len(frames)
    L_target = int(L / fps * target_fps)
    original_t = [x / fps for x in range(1, L+1)]
    t_idx = 0
    target_frames = []
    for target_t in range(1, L_target+1):
        while target_t / target_fps > original_t[t_idx]:
            t_idx += 1      # find the first t_idx so that target_t / target_fps <= original_t[t_idx]
            if t_idx >= L:
                break
        target_frames.append(frames[t_idx])

    # save video
    imageio.mimwrite(output_video, target_frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video

def gen_video_with_audio(character, text_to_audio):
    if character == "sydney":
        ref_audio_path = "/workspace/brain_rot_db/sydney-sweeney_old/sydsweeney-2.mp3"
        ref_audio_transcript_path = "/workspace/brain_rot_db/sydney-sweeney_old/transcript-sydsweeney-2.txt"
        with open(ref_audio_transcript_path, 'r') as f: ref_audio_transcript = f.read()

        cli_command_voice_gen = f"""f5-tts_infer-cli --model F5TTS_v1_Base --ref_audio "{ref_audio_path}" --ref_text "{ref_audio_transcript}" --gen_text "{text_to_audio}" --output_file /workspace/brain_rot_audio_gen/output.wav"""
        subprocess.run(cli_command_voice_gen, shell=True)
        audio_path = "/workspace/brain_rot_audio_gen/output.wav"
        # input_video_path = "/workspace/brain_rot_db/sydney-sweeney_old/sydsweeney-2.mp4"
        input_video_path = random.choice(os.listdir("/workspace/brain_rot_db/syd-sweeny"))
        processed_input_video_path = check_video(os.path.join("/workspace/brain_rot_db/syd-sweeny", input_video_path))
        output_video_path, _ = inference(audio_path, processed_input_video_path, bbox_shift=0, extra_margin=10, parsing_mode="jaw", 
                left_cheek_width=30, right_cheek_width=30, progress=gr.Progress(track_tqdm=True))
    
    if character == "ronaldo":
        ref_audio_path = "/workspace/brain_rot_db/ronaldo_old/ronaldo.mp3"
        ref_audio_transcript_path = "/workspace/brain_rot_db/ronaldo_old/transcript-ronaldo.txt"
        with open(ref_audio_transcript_path, 'r') as f: ref_audio_transcript = f.read()

        cli_command_voice_gen = f"""f5-tts_infer-cli --model F5TTS_v1_Base --ref_audio "{ref_audio_path}" --ref_text "{ref_audio_transcript}" --gen_text "{text_to_audio}" --output_file /workspace/brain_rot_audio_gen/output.wav"""
        subprocess.run(cli_command_voice_gen, shell=True)
        audio_path = "/workspace/brain_rot_audio_gen/output.wav"
        # input_video_path = "/workspace/brain_rot_db/ronaldo_old/ronaldo.mp4"
        input_video_path = random.choice(os.listdir("/workspace/brain_rot_db/ronaldo"))
        processed_input_video_path = check_video(os.path.join("/workspace/brain_rot_db/ronaldo", input_video_path))
        output_video_path, _ = inference(audio_path, processed_input_video_path, bbox_shift=0, extra_margin=10, parsing_mode="jaw", 
                left_cheek_width=10, right_cheek_width=10, progress=gr.Progress(track_tqdm=True))

    if character == "tsway":
        ref_audio_path = "/workspace/brain_rot_db/tsway/tsway.mp3"
        ref_audio_transcript_path = "/workspace/brain_rot_db/tsway/transcript-tsway.txt"
        with open(ref_audio_transcript_path, 'r') as f: ref_audio_transcript = f.read()

        cli_command_voice_gen = f"""f5-tts_infer-cli --model F5TTS_v1_Base --ref_audio "{ref_audio_path}" --ref_text "{ref_audio_transcript}" --gen_text "{text_to_audio}" --output_file /workspace/brain_rot_audio_gen/output.wav"""
        subprocess.run(cli_command_voice_gen, shell=True)
        audio_path = "/workspace/brain_rot_audio_gen/output.wav"
        input_video_path = "/workspace/brain_rot_db/tsway/tsway.mp4"
        processed_input_video_path = check_video(input_video_path)
        output_video_path, _ = inference(audio_path, processed_input_video_path, bbox_shift=0, extra_margin=10, parsing_mode="jaw", 
                left_cheek_width=30, right_cheek_width=30, progress=gr.Progress(track_tqdm=True))

    return output_video_path


# if __name__ == "__main__":
#     # text_to_audio = "Matrices are pretty straightforward babe, dont worry too much. Just look into my eyes and let me help you understand it. It is just a way of representing data that has both spatial and inherent meaning. The number 1, \"alone\"  has only an inherent value of 1 but when you write it inside the square bracket is now having a position information of 0 and inherent information of the value of 1. Just keep this analogy in mind and you will be able to apply this anywhere."
#     text_to_audio = "Hello my biggest fan, I know you have come to my doorstep. So i thought I'd say hi. Thanks for all the support Kriti, it means the world to me. My mind turns your life into folklore."
#     gen_video_with_audio("tsway", text_to_audio)
