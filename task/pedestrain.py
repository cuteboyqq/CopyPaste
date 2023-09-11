import os
import numpy as np
import shutil
import random
import cv2
import glob
from engine.datasets import BaseDatasets

class PedestrainDataset(BaseDatasets):
    # def __init__(self):
    #     self.label = 11 #Stop sign label
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
        y = random.randint(self.vanish_y,self.im.shape[0]-1)

        ## Stop should put at non-drivable area
        cnt = 1
        while(mask[y][x][0]==0): # Exist while when (x,y) is in drivable area
            x = random.randint(0,self.im.shape[1]-1)
            y = random.randint(self.vanish_y - 50,self.im.shape[0]-1)
            cnt+=1
            if cnt==100:
                break

        self.roi_x  = x
        self.roi_y  = y
        return (x,y)
        return NotImplementedError

    def Get_ROI_WH_In_Image(self,roi,roi_mask,dri_path):
        ## find left line

        ## find right line

        ## lane width = abs(right - left)
        ## roi_w = (lane_width)*0.50
        # if self.roi_y < self.vanish_y:
        #     self.roi_w = random.randint(60,120)
        #     self.roi_h = int(roi.shape[0]*(self.roi_w/roi.shape[1]))
        # else: ## small~big stop sign at bottom of image
        #     self.roi_w = int(roi.shape[1]*float(random.randint(10,20)*0.1))
        #     self.roi_h = int(roi.shape[0]*(self.roi_w/roi.shape[1]))
        self.roi_w = roi.shape[1]
        self.roi_h = roi.shape[0]
        
        ## filter too large size of stop sign
        if self.roi_w > 300:
            roi_w_pre = self.roi_w
            self.roi_w = 300
            self.roi_h = int(self.roi_h * float(300/roi_w_pre))
        elif self.roi_h > 300:
            roi_h_pre = self.roi_h
            self.roi_h = 300
            self.roi_w = int(self.roi_w * float(300/roi_h_pre))

        self.roi_resized = cv2.resize(roi,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)
        if roi_mask is not None:
            self.roi_mask = cv2.resize(roi_mask,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)
        else:
            self.roi_mask = None

        print(self.roi_resized.shape)
        #print(self.roi_mask.shape )
        return (self.roi_w,self.roi_h,self.roi_resized,self.roi_mask)
        return NotImplementedError

    def Get_ROI_Label(self):
    
        return self.label
    
    def Check_And_Update_ROI_XY_In_Image_Boundary(self,x_c,y_c,final_roi_w,final_roi_h,x_add,y_add):
        ## keep the road sign in the images
        print("final_roi_h = {}".format(final_roi_h))
        print("final_roi_w = {}".format(final_roi_w))
        print("x_c = {}".format(x_c))
        print("y_c = {}".format(y_c))


        if final_roi_h < 60:
            y_c = random.randint(self.vanish_y,self.vanish_y + 100)

        ## left boundary
        if x_c-int(final_roi_w/2.0)<=0:
            x_c = int(final_roi_w/2.0) + 1
        
        ## right boundary
        if x_c+int(final_roi_w/2.0)+x_add >= self.im.shape[1]:
            x_c = x_c - (int(final_roi_w/2.0)+x_add+1)

        ## ceiling(Top) boundary
        if y_c-int(final_roi_h/2.0)<=0:
            y_c =  int(final_roi_h/2.0) + 1 + y_add

        ## floor(Down) boundary
        if y_c+int(final_roi_h/2.0)+y_add>=self.im.shape[0]:
            y_c = self.im.shape[0]-(int(final_roi_h/2.0)+y_add+1)
            
        print("after update x_c,y_c")
        print("x_c = {}".format(x_c))
        print("y_c = {}".format(y_c))
        return x_c, y_c



    
                
            
            
            
        



    
