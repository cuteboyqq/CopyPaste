import os
import shutil
import glob
import numpy as np
from task.stopsign import StopSignDataset

def get_args_StopSign():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/train")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train")
    parser.add_argument('-dridir','--dri-dir',help='drivable label dir', \
                        default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    # For StopSign parameter
    parser.add_argument('-roidir','--roi-dir',help='roi dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v87/roi")
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/stopsign_images")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save stopsign fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save stopsign yolo.txt')
    parser.add_argument('-showimg','--show-img',type=bool,default=False,help='show images result')
    parser.add_argument('-numimg','--num-img',type=int,default=8000,help='number of generate fake landmark images')
    parser.add_argument('-roimaxwidth','--roi-maxwidth',type=int,default=400,help='max width of stop sign ROI')
    parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='enable(True)/disable(False) mask method to generate landmark or not')
    parser.add_argument('-roimaskdirstopsign','--roi-maskdirstopsign',help='roi mask dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v87/mask")
    parser.add_argument('-opencv','--opencv',type=bool,default=False,help='enable(True)/disable(False) opencv method to generate landmark or not')
    return parser.parse_args() 


if __name__=="__main__":
    args_stopsign = get_args_StopSign()
    stopsign = StopSignDataset(args_stopsign)
    stopsign.CopyPaste()

