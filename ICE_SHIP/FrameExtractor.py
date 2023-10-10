"""This file extracts frames from a video
and saves them to the images folder. Based on:https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/"""

import cv2
import os
import random

# Function to extract frames 
def FrameCapture(video_path,images_dir): 
  
    # Path to video file 
    vidObj = cv2.VideoCapture(video_path) 
    
    # Used as counter variable 
    count = 0
    # checks whether frames were extracted 
    success = 1
    
    total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    random_numbers = [random.randint(0, total_frames - 1) for _ in range(25)] #pick 25 random numbers
    
    for i in random_numbers:
        
        vidObj.set(1,i)
        success, image = vidObj.read()

        cv2.imwrite(images_dir+f"100m_frame_%d_{i}.jpg" % count, image) 
        count += 1

if __name__ == '__main__': 
    videos_dir="ICE_SHIP\\data\\videos\\"
    images_dir="ICE_SHIP\\data\\images\\"
    
    videos=os.listdir(videos_dir)
    for i in videos:
        FrameCapture(videos_dir+i,images_dir)
        