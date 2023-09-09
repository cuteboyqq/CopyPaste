import os
import numpy as np
import shutil
import random
import cv2
import glob
from engine.datasets import BaseDatasets

class LaneMarkingDataset(BaseDatasets):
    def __init__(self):
        self.label = 11 #Stop sign label
        self.method = "mask"
    

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
        x = random.randint(int(self.im.shape[1]*float(2/7)) ,int(self.im.shape[1]*float(5/7)))
        y = random.randint(self.vanish_y+60,self.im.shape[0]-1)

        ## Stop should put at non-drivable area
        while(mask[y][x][0]==0): # Exist while when (x,y) is in drivable area
            x = random.randint(int(self.im.shape[1]*float(2/7)) ,int(self.im.shape[1]*float(5/7)))
            y = random.randint(self.vanish_y+60,self.im.shape[0]-1)

        self.roi_x  = x
        self.roi_y  = y
        return (x,y)
        return NotImplementedError

    def Get_ROI_WH_In_Image(self,roi,roi_mask, dri_path):

        dri_map = cv2.imread(dri_path)
        ## find left line point x
        left_x = self.roi_x
        y = self.roi_y
        while(dri_map[y][left_x][0]!=0):
            left_x-=1

        print("left_x:{}".format(left_x))

        ## find right line point x
        right_x = self.roi_x
        while(dri_map[y][right_x][0]!=0):
            right_x+=1

        print("right_x:{}".format(right_x))


        ## lane width = abs(right - left)
        ## roi_w = (lane_width)*0.50
        lane_width = abs(right_x - left_x)
        self.roi_w = int(lane_width * 0.50)

        ratio = float(self.roi_w/roi.shape[1])
        self.roi_h = int(roi.shape[0]*ratio)
        
        print("self.roi_w:{}".format(self.roi_w))
        print("self.roi_h:{}".format(self.roi_h))

        self.roi_resized = cv2.resize(roi,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)
        self.roi_mask = cv2.resize(roi_mask,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)

        print(self.roi_resized.shape)
        print(self.roi_mask.shape )
        return (self.roi_w,self.roi_h,self.roi_resized,self.roi_mask)
        return NotImplementedError

    def Get_ROI_Label(self):
    
        return self.label



    
                
            
            
            
        



    
