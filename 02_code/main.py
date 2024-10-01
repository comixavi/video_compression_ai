import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2


def get_frame_from_video(video_path, frame_number):
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return None

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_number >= total_frames or frame_number < 0:
        print(f"Error: Frame number {frame_number} is out of range. Video has {total_frames} frames.")
        return None

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, ret_frame = video_capture.read()

    if not ret:
        print(f"Error: Could not retrieve frame {frame_number}.")
        return None

    video_capture.release()

    return ret_frame


video_path = r'C:\master_an1\video_compression\video_compression_ai\03_clips\video1.mp4'
frame_number = 100

frame = get_frame_from_video(video_path, frame_number)

if frame is not None:
    tFrame = torch.from_numpy(frame)

    print(type(tFrame))
    print(tFrame.shape)

    cv2.imshow(f'Frame {frame_number}', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

