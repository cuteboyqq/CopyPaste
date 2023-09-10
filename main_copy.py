import os
import shutil
import glob
import numpy as np
from task.stopsign import StopSignDataset
from task.lanemarking import LaneMarkingDataset

def get_args_StopSign():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/train")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train")
    parser.add_argument('-dridir','--dri-dir',help='drivable label dir', \
                        default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    # For StopSign parameter
    parser.add_argument('-roidir','--roi-dir',help='roi dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v8787/roi")
    
    parser.add_argument('-maskdir','--mask-dir',help='mask dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v8787/mask")
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="/home/ali/Projects/GitHub_Code/ali/CopyPaste/generate_images")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save stopsign fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save stopsign yolo.txt')
    parser.add_argument('-showimg','--show-img',type=bool,default=False,help='show images result')
    parser.add_argument('-showroi','--show-roi',type=bool,default=False,help='show roi result')
    parser.add_argument('-numimg','--num-img',type=int,default=1000,help='number of generate fake landmark images')
    parser.add_argument('-roimaxwidth','--roi-maxwidth',type=int,default=400,help='max width of stop sign ROI')
    parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='enable(True)/disable(False) mask method to generate landmark or not')
    parser.add_argument('-roimaskdirstopsign','--roi-maskdirstopsign',help='roi mask dir',\
                        default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v87/mask")
    parser.add_argument('-opencv','--opencv',type=bool,default=False,help='enable(True)/disable(False) opencv method to generate landmark or not')
    parser.add_argument('-roilabel','--roi-label',type=int,default=10,help='stop sign label = 10')
    parser.add_argument('-method','--method',type=str,default="mask",help='use mask/opencv/both method to generate landmark')
    parser.add_argument('-carhoodratio','--carhood-ratio',type=float,default=0.75,help='carhood ratio')
    return parser.parse_args() 


def get_args_LaneMarking():
    import argparse
    
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/GitHub_Code/ali/CopyPaste/generate_images/images")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/GitHub_Code/ali/CopyPaste/generate_images/labels")
    parser.add_argument('-dridir','--dri-dir',help='drivable label dir', \
                        default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    parser.add_argument('-roidir','--roi-dir',help='roi dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/roi")

    parser.add_argument('-maskdir','--mask-dir',help='roi mask dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/mask")
    
    parser.add_argument('-drilabeldirtrain','--dri-labeldirtrain',help='drivable label dir fo train',default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/masks/train")
    parser.add_argument('-linelabeldir','--line-labeldir',help='line label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/lane/masks/train")
    
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="/home/ali/Projects/GitHub_Code/ali/CopyPaste/lanemark_fake_images")


    parser.add_argument('-saveimg','--save-img',action='store_true',help='save landmark fake images')
    #parser.add_argument('-savecolormap','--save-colormap',action='store_true',help='save generate semantic segment colormaps')
    #parser.add_argument('-savemask','--save-mask',action='store_true',help='save generate semantic segment train masks')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save lanemarking yolo.txt')
    parser.add_argument('-numimg','--num-img',type=int,default=900,help='number of generate fake landmark images')
    parser.add_argument('-useopencvratio','--use-opencvratio',type=float,default=0.50,help='ratio of using opencv method to generate landmark images')
    parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='use mask method to generate landmark or not')
    parser.add_argument('-show','--show',action='store_true',help='show images result')
    parser.add_argument('-usestopsigndataset','--use-stopsigndataset',type=bool,default=True,help='use stop sign label.txt dataset')

    parser.add_argument('-roilabel','--roi-label',type=int,default=11,help='lanemark label = 11')
    parser.add_argument('-method','--method',type=str,default="both",help='use mask/opencv/both method to generate landmark')
    parser.add_argument('-showroi','--show-roi',type=bool,default=False,help='show roi result')
    parser.add_argument('-showimg','--show-img',type=bool,default=False,help='show images result')

    parser.add_argument('-carhoodratio','--carhood-ratio',type=float,default=0.75,help='carhood ratio')
    return parser.parse_args()    


if __name__=="__main__":
    STOPSIGN=True
    LANEMARKING=True
    SEQUENTIAL_COPYPASTE=True
    if SEQUENTIAL_COPYPASTE:
        if STOPSIGN:
            args_stopsign = get_args_StopSign()
            stopsign = StopSignDataset(args_stopsign)
            stopsign.CopyPaste()
        if LANEMARKING:
            args_lanemarking = get_args_LaneMarking()
            lanemarking = LaneMarkingDataset(args_lanemarking)
            lanemarking.CopyPaste()
