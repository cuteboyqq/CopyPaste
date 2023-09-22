import os
import shutil
import glob
import numpy as np

## Setting here is more convienent than setting on the default of arguments.....

## Generate Number of images
StopSign_num = 41000
LnaeMarking_num = 39999
Pedestrian_num = 10

## label settings

# 0: pedestrian
# 1: rider
# 2: car
# 3: truck
# 4: bus
# 5: train
# 6: motorcycle
# 7: bicycle
# 8: traffic light
# 9: traffic sign

## Keep the traffic sign
KEEP_TRAFFIC_SIGN=True
TRAIN=True

if KEEP_TRAFFIC_SIGN:
    StopSign_label = 10
    LaneMarking_label = 11 
    Pedestrian_label = 0
    if TRAIN:
        IMAGE_DIR = "/home/ali/Projects/datasets/BDD100K-ori/images/100k/train"
        LABEL_DIR = "/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train"
        DRI_DIR = "/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train"
    else:
        IMAGE_DIR = "/home/ali/Projects/datasets/BDD100K-ori/images/100k/val"
        LABEL_DIR = "/home/ali/Projects/datasets/BDD100K-ori/labels/detection/val"
        DRI_DIR = "/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/val"

else:
    StopSign_label = 9
    LaneMarking_label = 10 
    Pedestrian_label = 0
    IMAGE_DIR = "/home/ali/Projects/datasets/bdd100k_data/images/100k/train"
    LABEL_DIR = "/home/ali/Projects/datasets/bdd100k_data/labels/detection/train"

if TRAIN:
    STOPSIGN_ROI_DIR = "./datasets/ROI/stopsign/train/roi"
    LANEMARKING_ROI_DIR = "./datasets/ROI/lanemarking/train/roi"
    PEDESTRIAN_ROI_DIR = "./datasets/ROI/pedestrain/train/roi"

    STOPSIGN_MASK_DIR = "./datasets/ROI/stopsign/train/mask"
    LANEMARKING_MASK_DIR = "./datasets/ROI/lanemarking/train/mask"
    PEDESTRIAN_MASK_DIR = "./datasets/ROI/pedestrain/train/mask"
else:
    STOPSIGN_ROI_DIR = "./datasets/ROI/stopsign/val/roi"
    LANEMARKING_ROI_DIR = "./datasets/ROI/lanemarking/val/roi"
    PEDESTRIAN_ROI_DIR = "./datasets/ROI/pedestrain/val/roi"

    STOPSIGN_MASK_DIR = "./datasets/ROI/stopsign/val/mask"
    LANEMARKING_MASK_DIR = "./datasets/ROI/lanemarking/val/mask"
    PEDESTRIAN_MASK_DIR = "./datasets/ROI/pedestrain/val/mask"


def get_args_StopSign():
    import argparse
    parser = argparse.ArgumentParser()
    ##   BDD100k datasets directory
    parser.add_argument('-imgdir','--img-dir',help='image dir',default=IMAGE_DIR)
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default=LABEL_DIR)
    parser.add_argument('-dridir','--dri-dir',help='drivable label dir', \
                        default=DRI_DIR)
    ##   Stop sign ROI/Mask directory
    parser.add_argument('-roidir','--roi-dir',help='roi dir',\
                        default=STOPSIGN_ROI_DIR)
    
    parser.add_argument('-maskdir','--mask-dir',help='mask dir',\
                        default=STOPSIGN_MASK_DIR)
    ## Save parameters
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="./runs/generate_images")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save stopsign fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save stopsign yolo.txt')
    parser.add_argument('-showimg','--show-img',type=bool,default=False,help='show images result')
    parser.add_argument('-showroi','--show-roi',type=bool,default=False,help='show roi result')


    ## Others setting
    parser.add_argument('-numimg','--num-img',type=int,default=StopSign_num,help='number of generate fake landmark images')
    parser.add_argument('-roith','--roi-th',type=int,default=100,help='roi w/h threshold')
    parser.add_argument('-overlapth','--overlap-th',type=float,default=0.01,help='roi IOU threshold')
    #parser.add_argument('-roimaxwidth','--roi-maxwidth',type=int,default=12000,help='max width of stop sign ROI')
    #parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='enable(True)/disable(False) mask method to generate landmark or not')
    #parser.add_argument('-roimaskdirstopsign','--roi-maskdirstopsign',help='roi mask dir',\
    #                    default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/datasets/stop_sign_new_v87/mask")
    #parser.add_argument('-opencv','--opencv',type=bool,default=False,help='enable(True)/disable(False) opencv method to generate landmark or not')

    ## stop sign label
    parser.add_argument('-roilabel','--roi-label',type=int,default=StopSign_label,help='stop sign label = 9')
    parser.add_argument('-method','--method',type=str,default="mask",help='use mask/opencv/both method to generate landmark')
    parser.add_argument('-carhoodratio','--carhood-ratio',type=float,default=0.90,help='carhood ratio')
    parser.add_argument('-numroi','--num-roi',type=int,default=4,help='number of pedestrain roi in image')
    return parser.parse_args() 


def get_args_LaneMarking():
    import argparse
    
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    ##   BDD100k datasets directory

    # image is from previous generate stop sign images
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="./runs/generate_images/images")#"./runs/generate_images/images"
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="./runs/generate_images/labels")#"./runs/generate_images/labels"
    parser.add_argument('-dridir','--dri-dir',help='drivable label dir', \
                        default=DRI_DIR)
    
    ##   lanemarking  ROI/Mask directory
    parser.add_argument('-roidir','--roi-dir',help='roi dir',default=LANEMARKING_ROI_DIR)

    parser.add_argument('-maskdir','--mask-dir',help='roi mask dir',default=LANEMARKING_MASK_DIR)


    
    parser.add_argument('-drilabeldirtrain','--dri-labeldirtrain',help='drivable label dir fo train',default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/masks/train")
    parser.add_argument('-linelabeldir','--line-labeldir',help='line label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/lane/masks/train")
    

    ## Save parameters
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="./runs/lanemark_fake_images")


    parser.add_argument('-saveimg','--save-img',action='store_true',help='save landmark fake images')
    #parser.add_argument('-savecolormap','--save-colormap',action='store_true',help='save generate semantic segment colormaps')
    #parser.add_argument('-savemask','--save-mask',action='store_true',help='save generate semantic segment train masks')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save lanemarking yolo.txt')
    parser.add_argument('-numimg','--num-img',type=int,default=LnaeMarking_num,help='number of generate fake landmark images')
    parser.add_argument('-roith','--roi-th',type=int,default=100,help='roi w/h threshold')
    parser.add_argument('-overlapth','--overlap-th',type=float,default=0.01,help='roi IOU threshold')


    parser.add_argument('-useopencvratio','--use-opencvratio',type=float,default=0.50,help='ratio of using opencv method to generate landmark images')
    parser.add_argument('-usemask','--use-mask',type=bool,default=True,help='use mask method to generate landmark or not')
    parser.add_argument('-show','--show',action='store_true',help='show images result')
    parser.add_argument('-usestopsigndataset','--use-stopsigndataset',type=bool,default=True,help='use stop sign label.txt dataset')
    ## lane marking label
    parser.add_argument('-roilabel','--roi-label',type=int,default=LaneMarking_label,help='lanemark label = 11')

    parser.add_argument('-method','--method',type=str,default="both",help='use mask/opencv/both method to generate landmark')
    parser.add_argument('-showroi','--show-roi',type=bool,default=False,help='show roi result')
    parser.add_argument('-showimg','--show-img',type=bool,default=False,help='show images result')

    parser.add_argument('-carhoodratio','--carhood-ratio',type=float,default=0.90,help='carhood ratio')

    parser.add_argument('-numroi','--num-roi',type=int,default=1,help='number of pedestrain roi in image')
    return parser.parse_args()


def get_args_Pedestrain():
    import argparse
    parser = argparse.ArgumentParser()
    ##   BDD100k datasets directory
    parser.add_argument('-imgdir','--img-dir',help='image dir',default="./runs/lanemark_fake_images/images")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="./runs/lanemark_fake_images/labels")
    parser.add_argument('-dridir','--dri-dir',help='drivable label dir', \
                        default=DRI_DIR)
    ##   Pedestrian ROI/Mask directory
    parser.add_argument('-roidir','--roi-dir',help='roi dir',\
                        default=PEDESTRIAN_ROI_DIR)
    
    parser.add_argument('-maskdir','--mask-dir',help='mask dir',\
                        default=PEDESTRIAN_MASK_DIR)
    ## Save parameters
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="./runs/generate_pedestrain_images")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save pedestrain fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save pedestrain yolo.txt')
    parser.add_argument('-showimg','--show-img',type=bool,default=False,help='show images result')
    parser.add_argument('-showroi','--show-roi',type=bool,default=False,help='show roi result')


    ## Others setting
    parser.add_argument('-numimg','--num-img',type=int,default=Pedestrian_num,help='number of generate fake pedestrain images')
    parser.add_argument('-roith','--roi-th',type=int,default=100,help='roi w/h threshold')
    parser.add_argument('-overlapth','--overlap-th',type=float,default=0.01,help='roi IOU threshold')

    parser.add_argument('-roilabel','--roi-label',type=int,default=Pedestrian_label,help='pedestrain label = 0')
    parser.add_argument('-method','--method',type=str,default="mask",help='use mask/opencv/both method to generate pedestrain')
    parser.add_argument('-carhoodratio','--carhood-ratio',type=float,default=0.90,help='carhood ratio')
    parser.add_argument('-numroi','--num-roi',type=int,default=12,help='number of pedestrain roi in image')
    return parser.parse_args()



def get_args_label():
    import argparse
    parser = argparse.ArgumentParser()
    ##   BDD100k datasets directory
    # parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/train")
    # parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train")
    # parser.add_argument('-drivabledir','--drivable-dir',help='drivable label dir', \
    #                     default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    
    # parser.add_argument('-lanedir','--lane-dir',help='lane dir', \
    #                     default="/home/ali/Projects/datasets/BDD100K-ori/labels/lane/colormaps/train")

    # ## Save parameters
    # parser.add_argument('-savedir','--save-dir',help='save img dir',default="../tools/re-label/train")
    # parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save pedestrain fake images')
    # parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save pedestrain yolo.txt')
    
    # parser.add_argument('--removelabellist','-remove-labellist',type=list,nargs='+',default="9",help='remove label list')

    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/nuimages/nuimages-dataset/images/train")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/nuimages/nuimages-dataset/labels/detection/train")
    parser.add_argument('-drivabledir','--drivable-dir',help='drivable label dir', \
                        default="/home/ali/Projects/datasets/nuimages/nuimages-v1.0-all-samples/labels/drivable/train")
    
    parser.add_argument('-lanedir','--lane-dir',help='lane dir', \
                        default="/home/ali/Projects/datasets/nuimages/nuimages-v1.0-all-samples/labels/lane/train")

    ## Save parameters
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="../tools/split_bdd100k_train_images")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save pedestrain fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save pedestrain yolo.txt')
    
    parser.add_argument('--removelabellist','-remove-labellist',type=list,nargs='+',default="9",help='remove label list')
    return parser.parse_args()

def get_args_roi():
    import argparse
    parser = argparse.ArgumentParser()
    ##   BDD100k datasets directory
    parser.add_argument('-datadir','--data-dir',help='data dir',default="/home/ali/Projects/datasets/nuimages/roi_train")
    parser.add_argument('-savedir','--save-dir',help='save dir',default="./nuimages_roi_2023-09-21")
    parser.add_argument('-roidir','--roi-dir',help='roi dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/runs/predict/0_filtered/train/roi")
    parser.add_argument('-maskdir','--mask-dir',help='mask dir',default="/home/ali/Projects/GitHub_Code/ali/landmark_issue/runs/predict/0/mask")
    return parser.parse_args()


if __name__=="__main__":
    #get_args_StopSign()

    #get_args_LaneMarking()

    get_args_Pedestrain()