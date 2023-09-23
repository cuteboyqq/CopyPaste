import os
import numpy as np
import shutil
import random
import cv2
import glob
from engine.datasets import BaseDatasets

class LaneMarkingDataset(BaseDatasets):
    # def __init__(self,args):
    #     super.__init__(self)
    #     self.label = 11 #Stop sign label
    #     self.method = args.method
    

    def Get_ROI_Label(self):
    
        return self.label

    # def Get_Update_Vanish_Y(self):
    #     im_h = self.im.shape[0]
    #     vanish_y = 0
    #     x = int(self.im.shape[1]/2.0)
    #     while(self.dri[vanish_y][x][0]==0):
    #         if vanish_y+1< (im_h-1):
    #             vanish_y+=1
    #         else:
    #             break

    #     if vanish_y>= int(im_h * self.carhood_ratio):
    #         vanish_y = int(im_h * self.carhood_ratio) - 100
            
      
    #     self.vanish_y = vanish_y

    #     return vanish_y


    def Get_Possible_ROI_Position_Area(self):
        ## binary mask of enable/disable position mask
        mask = np.zeros(self.im.shape, dtype=np.uint8)
        mask[self.dri>0]=255
        mask[self.dri==0]=0
        return mask


    def Get_ROI_XY_In_Image(self,vanish_y,carhood_ratio):
        #self.vanish_y = self.Get_Update_Vanish_Y()
        print("self.vanish_y:{}".format(self.vanish_y))
        #input()
        mask = self.Get_Possible_ROI_Position_Area()
        
        ## initial random (x,y)
        x = random.randint(int(self.im.shape[1]*float(2/7)) ,int(self.im.shape[1]*float(5/7)))
        range = int(self.im.shape[0]*carhood_ratio) - vanish_y
        y = random.randint(vanish_y,int(self.im.shape[0]*carhood_ratio)-int(range/2.0))
        # if self.vanish_y + 100 < self.im.shape[0] - 1:
        #     y = random.randint(self.vanish_y,int(self.vanish_y+100))
        # else:
        #     y = self.vanish_y

        ## Stop should put at non-drivable area
        cnt = 1
        while(mask[y][x][0]==0): # Exist while when (x,y) is in drivable area
            print("mask[y][x][0]==0 re-assign XY~~~~")
            ra = random.randint(3,8)
            x = random.randint(int(self.im.shape[1]*float(2/7)) ,int(self.im.shape[1]*float(5/7)))
            y = random.randint(vanish_y,int(self.im.shape[0]*carhood_ratio)-int(range*2/ra))
            # if self.vanish_y+100 < self.im.shape[0] - 1:
            #     y = random.randint(self.vanish_y,int(self.vanish_y+100))
            # else:
            #     y = self.vanish_y
            cnt+=1
            if cnt==100:
                break
            
        self.roi_x  = x
        self.roi_y  = y
        print("self.roi_x :{}".format(self.roi_x))
        print("self.roi_y :{}".format(self.roi_y))
        return (x,y)
        return NotImplementedError

    def Get_ROI_WH_In_Image(self,roi,roi_mask, dri_path):

        dri_map = cv2.imread(dri_path)
        ## find left line point x
        left_x = self.roi_x
        y = self.roi_y
        print("left_x:{}".format(left_x))
        print("y:{}".format(y))
        while(dri_map[y][left_x][0]!=0):
            #print(left_x)
            if left_x-1 >=0:
                left_x-=1
            else:
                break

        print("left_x:{}".format(left_x))
        self.left_line_x = left_x
        ## find right line point x
        right_x = self.roi_x
        while(dri_map[y][right_x][0]!=0):
            if right_x+1<=self.im.shape[1]-1:
                right_x+=1
                #print(right_x)
            else:
                break

        print("right_x:{}".format(right_x))
        self.right_line_x = right_x

        ## lane width = abs(right - left)
        ## roi_w = (lane_width)*0.50
        

        if abs(right_x - left_x)==0:
            lane_width = 100
        else:
            lane_width = abs(right_x - left_x)

        ## initial roi_w, roi_h
        self.roi_w = int(lane_width * 0.40)
        ratio = float(self.roi_w/roi.shape[1])
        self.roi_h = int(roi.shape[0]*ratio)
        
        # ## check ratio
        # if ratio < 0.33:
        #     self.roi_w = int(lane_width * 0.20)
        
        # ratio = float(self.roi_w/roi.shape[1])
        # self.roi_h = int(roi.shape[0]*ratio)

        ##Check roi_w, roi_h
        if self.roi_h >= int(self.im.shape[0] * 0.30):
            self.roi_h = int(self.im.shape[0] * 0.30)
            ratio = float(self.roi_h/roi.shape[0])
            self.roi_w = int(roi.shape[1]*ratio)
        elif self.roi_h<=0:
            self.roi_h = 20
            self.roi_w = 80
            

        print("self.roi_w:{}".format(self.roi_w))
        print("self.roi_h:{}".format(self.roi_h))

        self.roi_resized = cv2.resize(roi,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)

        if roi_mask is not None:
            self.roi_mask = cv2.resize(roi_mask,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)
        else:
            self.roi_mask = None

        print(self.roi_resized.shape)
        print(self.roi_mask.shape )
        return (self.roi_w,self.roi_h,self.roi_resized,self.roi_mask)
        return NotImplementedError


    def Check_And_Update_ROI_XY_In_Image_Boundary(self,x_c,y_c,final_roi_w,final_roi_h,x_add,y_add):
        ## keep the lane-marking in the middle of drive area lane

        x_c = int((self.left_line_x+self.right_line_x)/2.0)

        print("final_roi_h = {}".format(final_roi_h))
        print("final_roi_w = {}".format(final_roi_w))
        print("x_c = {}".format(x_c))
        print("y_c = {}".format(y_c))
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

    def Get_ROI_Label(self):
    
        return self.label



    
                
            
            
            
        



    
