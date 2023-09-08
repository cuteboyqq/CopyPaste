import os
import numpy as np
import shutil
import random
import cv2
import glob
from engine.datasets import BaseDatasets

class StopSignDataset(BaseDatasets):
     
    
    def Get_Possible_ROI_Position_Area(self):
        ## binary mask of enable/disable position mask
        mask = np.zeros(self.im.shape, dtype=np.uint8)
        mask[self.dri>0]=255
        mask[self.dri==0]=0
        return mask


    def Get_ROI_Position_In_Image(self):
        mask = self.Get_Possible_ROI_Position_Area()
        
        ## initial random (x,y)
        x = random(0,self.im.shape[1]-1)
        y = random(0,self.im.shape[0]-1)

        ## Stop should put at non-drivable area
        while(mask[y][x]!=0): # Exist while when (x,y) is in non-drivable area
            x = random(0,self.im.shape[1]-1)
            y = random(0,self.im.shape[0]-1)

        self.roi_x  = x
        self.roi_y  = y
        return (x,y)
        return NotImplementedError

    def Get_ROI_Size_In_Image(self,roi):
        ## small stop sign at top of image
        if self.y < self.vanish_y:
            roi_w = random.randint(15,50)
            roi_h = int(self.roi.shape[0]*(roi_w/self.roi.shape[1]))
        else: ## small~big stop sign at bottom of image
            roi_w = int(self.roi.shape[1]*float(random.randint(5,20)*0.1))
            roi_h = int(self.roi.shape[0]*(roi_w/self.roi.shape[1]))

        return (roi_w,roi_h)
        return NotImplementedError

    def Get_ROI_Label(self):
        label = 9
        return label

    def Get_ROI_info(self,roi):
        position  = self.Get_ROI_Position_In_Image()
        size = self.Get_ROI_Size_In_Image(roi)
        label = self.Get_ROI_Label()
        return (label,position,size)

    def Get_Random_ROI(self):
        roi_path_list = glob.glob(os.path.join(self.roi_dir,"*.jpg"))
        roi_index = random.randint(0,len(roi_path_list)-1)
        self.roi_path = roi_path_list[roi_index]
        roi = cv2.imread(self.roi_path)
        return roi

    def CopyPaste(self):
        for i in range(len(self.data_info)):
            self.im_path, self.dri_path, self.im, self.dri , self.vanish_y = self.data_info[i]
            roi = self.Get_Random_ROI()
            label,position,size = self.Get_ROI_info(roi)

            if self.args.save_txt:
                return NotImplemented

            if self.args.save_img:
                return NotImplemented

            if self.args.show_img:
                return NotImplemented
            
            if self.args.show_roi:
                return NotImplemented
                
            
            #mask = self.Get_Possible_ROI_Position_Area()
            
        return NotImplementedError
        

        return NotImplementedError

    
