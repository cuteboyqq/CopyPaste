import os
import numpy as np
import shutil
import random
import glob
import cv2

class BaseDatasets:
    def __init__(self,args):
        
        self.args = args

        # bdd100k datasets
        self.img_dir = args.img_dir
        self.dri_dir = args.dri_dir
        self.img_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        
        self.data_info = []

        for im_path in self.img_path_list:
            self.Parse_path()
            dri_file = self.im_name + ".png"
            dri_path = os.path.join(self.dri_dir,dri_file)
            im = cv2.imread(im_path)
            dri = cv2.imread(dri_path)

            vanish_y = 0
            get_vanish_y = False
            for i in range(dri.shape[0]):
                for j in range(dri.shape[1]):
                    if dri[i][j]!=0 and get_vanish_y==False:
                        vanish_y = i
                        get_vanish_y = True

            self.data_info.append([im_path,     dri_path,   im,     dri,    vanish_y])
        
        # roi datasets
        self.roi_label = args.roi_label
        self.roi_dir = args.roi_dir
        self.mask_dir = args.mask_dir

    def Parse_path(self):
        self.im  = self.im_path.split("//")[-1]
        self.im_name = self.im.split("//")[0] 

    def Get_Possible_ROI_Position_Area(self):
        #This are different to roi labels
        return NotImplemented

    def Get_ROI_Position_In_Image(self):
        x = 0
        y = 0
        return (x,y)
        return NotImplementedError

    def Get_ROI_Size_In_Image(self):
        w = 50
        h = 50
        return (w,h)
        return NotImplementedError

    def Get_ROI_Label(self):
        label = 9
        return label

    def Get_ROI_info(self):
        position  = self.Get_ROI_Position_In_Image()
        size = self.Get_ROI_Size_In_Image()
        label = self.Get_ROI_Label()
        return (label,position,size)

    def CopyPaste(self):
        return NotImplementedError
        

        return NotImplementedError

    
