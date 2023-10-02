#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:35:37 2021

@author: ali
"""

import cv2
import os
import shutil
import glob

'''
path = "/home/ali/datasets/train_video/NewYork_train/NewYork_train7.mp4"
vidcap = cv2.VideoCapture(path)
skip_frame = 15

txt_dir = "/home/ali/YOLOV5/runs/detect/NewYork_train7/labels"
class_path = "/home/ali/datasets/train_video/classes.txt"

yolo_infer_txt = True
'''


def Analysis_path(path):
    file = path.split(os.sep)[-1]
    file_name = file.split(".")[0]
    file_dir = os.path.dirname(path)
    return file,file_name,file_dir

#c_file,c_file_name,c_file_dir = Analysis_path(class_path)

def video_extract_frame(video_path,args):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 1
    file,filename,file_dir = Analysis_path(video_path)
    print(file," ",filename," ",file_dir)
    save_folder_name =  filename + "_imgs"
    save_dir = os.path.join(file_dir,save_folder_name)
    os.makedirs(save_dir,exist_ok=True)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    # Copy class.txt to save_dir
    # shutil.copy(args.class_txt,save_dir)
    
    while success:
        if count% (args.skip_f)==0:
            
            #====extract video frame====
            filename_ = filename + "_" + str(count) + ".jpg"
            img_path = os.path.join(save_dir,filename_)
            
            cv2.imwrite(img_path,image)
            if args.yolo_infer:
                #=====Copy .txt file=======
                filename_txt_ = filename + "_" + str(count) + ".txt"
                txt_path = os.path.join(args.yolo_txt,filename_txt_)
                if os.path.exists(txt_path):
                    shutil.copy(txt_path,save_dir)
                    cv2.imwrite(img_path,image)
            #cv2.imwrite("/home/ali/datasets-old/TL4/frame%d.jpg" % count, image)     # save frame as JPEG file    
            print('save frame ',count)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def video_extract_frames(args):
    video_path_list = glob.glob(os.path.join(args.video_dir,"*.mp4"))
    for i in range(len(video_path_list)):
        video_extract_frame(video_path_list[i],args)

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-videodir','--video-dir',help="input video path",default=r"C:\\datasets\\Kaohsiung_Night_Lane_Marking")
    parser.add_argument('-videopath','--video-path',help="input video path",default="/home/ali/datasets/train_video/LosAngeles_train/LosAngeles_train1.mp4")
    parser.add_argument('-skipf','--skip-f',type=int,help="number of skp frame",default=3)
    parser.add_argument('-yoloinfer','--yolo-infer',action='store_true',help="have yolo infer txt")
    parser.add_argument('-yolotxt','--yolo-txt',help="yolo infer label txt dir",default="/home/ali/YOLOV4/inference/LosAngeles_train1-infer/labels")
    parser.add_argument('-classtxt','--class-txt',help="class.txt path",default="/home/ali/datasets/train_video/classes.txt")
    
    return parser.parse_args()
    
if __name__=="__main__":
    
    args=get_args()
    video_path = args.video_path
    skip_frame = args.skip_f
    yolo_txt_dir = args.yolo_txt
    class_path = args.class_txt
    yolo_infer = args.yolo_infer
    
    print("video_path =",video_path)
    print("skip_frame = ",skip_frame)
    print("yolo_txt_dir = ",yolo_txt_dir)
    print("class_path = ",class_path)
    print("yolo_infer = ",yolo_infer)
    
    # video_extract_frame(video_path,skip_frame,yolo_txt_dir,class_path,True)
    video_extract_frames(args)
    