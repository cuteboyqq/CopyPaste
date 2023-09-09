import os
import numpy as np
import shutil
import random
import cv2
import glob
from engine.datasets import BaseDatasets

class StopSignDataset(BaseDatasets):
    # def __init__(self):
    #     self.label = 10 #Stop sign label
    #     self.method = "mask"
    

    def Get_ROI_Label(self):
    
        return self.label

    def Get_Possible_ROI_Position_Area(self):
        ## binary mask of enable/disable position mask
        mask = np.zeros(self.im.shape, dtype=np.uint8)
        mask[self.dri>0]=255
        mask[self.dri==0]=0
        return mask


    def Get_ROI_XY_In_Image(self):
        mask = self.Get_Possible_ROI_Position_Area()
        
        ## initial random (x,y)
        x = random.randint(0,self.im.shape[1]-1)
        y = random.randint(0,self.im.shape[0]-1)

        ## Stop should put at non-drivable area
        while(mask[y][x][0]!=0): # Exist while when (x,y) is in non-drivable area
            x = random.randint(0,self.im.shape[1]-1)
            y = random.randint(0,self.im.shape[0]-1)

        self.roi_x  = x
        self.roi_y  = y
        return (x,y)
        return NotImplementedError

    def Get_ROI_WH_In_Image(self,roi,roi_mask):
        ## small stop sign at top of image
        if self.roi_y < self.vanish_y:
            self.roi_w = random.randint(20,50)
            self.roi_h = int(roi.shape[0]*(self.roi_w/roi.shape[1]))
        else: ## small~big stop sign at bottom of image
            self.roi_w = int(roi.shape[1]*float(random.randint(5,20)*0.1))
            self.roi_h = int(roi.shape[0]*(self.roi_w/roi.shape[1]))
        
        ## filter too large size of stop sign
        if self.roi_w > 350:
            roi_w_pre = self.roi_w
            self.roi_w = 350
            self.roi_h = int(self.roi_h * float(350/roi_w_pre))
        elif self.roi_h > 350:
            roi_h_pre = self.roi_h
            self.roi_h = 350
            self.roi_w = int(self.roi_w * float(350/roi_h_pre))

        self.roi_resized = cv2.resize(roi,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)
        self.roi_mask = cv2.resize(roi_mask,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)

        print(self.roi_resized.shape)
        print(self.roi_mask.shape )
        return (self.roi_w,self.roi_h,self.roi_resized,self.roi_mask)
        return NotImplementedError

    def Get_ROI_Label(self):
    
        return self.label



    
                
            
            
            
        



    
