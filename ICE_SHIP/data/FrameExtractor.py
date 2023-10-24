"""This file extracts frames from a video
and saves them to the images folder. Based on:https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/"""

import cv2
import os
import random

# Function to extract frames 
def FrameCapture(video_dir,vid_name,save_dir, frame): 
    vidObj = cv2.VideoCapture(video_dir+vid_name)  
    success, image = vidObj.read()
    raw_vid_name=os.path.splitext(vid_name)[0]
    cv2.imwrite(save_dir+raw_vid_name+f"_frame{frame}.jpg",image)

def NumFrames(video_path):
    vidObj = cv2.VideoCapture(video_path) 
    number_of_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return number_of_frames

if __name__ == '__main__': 
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    videos_dir=ROOT_DIR+"\\data\\NRC_data_v2\\videos\\"
    train_dir=ROOT_DIR+"\\data\\NRC_data_v2\\train\\"
    val_dir=ROOT_DIR+"\\data\\NRC_data_v2\\val\\"
    
    videos=os.listdir(videos_dir)
    vids_frames={}
    
    for vid in videos:
        vids_frames[f"{vid}"]=NumFrames(videos_dir+vid)
    
    num_train_images=28
    num_val_images=12
    
    randomTrainVids=[random.choice(videos) for _ in range(0,num_train_images)]
    randomValVids=[random.choice(videos) for _ in range(0,num_val_images)]
       
    for train_vid in randomTrainVids:
        FrameCapture(videos_dir,train_vid,train_dir,random.randint(0,vids_frames[train_vid]-1))
    
    for val_vid in randomValVids:
        FrameCapture(videos_dir,val_vid,val_dir,random.randint(0,vids_frames[val_vid]-1))   