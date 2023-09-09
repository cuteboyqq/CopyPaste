import os
import numpy as np
import shutil
import random
import cv2
import glob
from engine.datasets import BaseDatasets

class StopSignDataset(BaseDatasets):
    def __init__(self):
        self.label = 9 #Stop sign label
        self.method = "mask"
    
    def Get_Possible_ROI_Position_Area(self):
        ## binary mask of enable/disable position mask
        mask = np.zeros(self.im.shape, dtype=np.uint8)
        mask[self.dri>0]=255
        mask[self.dri==0]=0
        return mask


    def Get_ROI_XY_In_Image(self):
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

    def Get_ROI_WH_In_Image(self,roi,roi_mask):
        ## small stop sign at top of image
        if self.y < self.vanish_y:
            self.roi_w = random.randint(15,50)
            self.roi_h = int(self.roi.shape[0]*(self.roi_w/self.roi.shape[1]))
        else: ## small~big stop sign at bottom of image
            self.roi_w = int(self.roi.shape[1]*float(random.randint(5,20)*0.1))
            self.roi_h = int(self.roi.shape[0]*(self.roi_w/self.roi.shape[1]))

        self.roi_resized = cv2.resize(roi,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)
        self.roi_mask = cv2.resize(roi_mask,(self.roi_w,self.roi_h),interpolation=cv2.INTER_NEAREST)
        return (self.roi_w,self.roi_h)
        return NotImplementedError

    def Get_ROI_Label(self):
    
        return self.label

    def Get_ROI_lxywh_In_Image(self,roi,roi_mask):
        roi_x,roi_y  = self.Get_ROI_XY_In_Image()
        roi_w,roi_h = self.Get_ROI_WH_In_Image(roi,roi_mask)
        roi_label = self.Get_ROI_Label()
        return (roi_label,roi_x,roi_y,roi_w,roi_h)

    def Get_Random_ROI_And_ROIMask(self):
        ## Get random stop sign ROI
        roi_path_list = glob.glob(os.path.join(self.roi_dir,"*.jpg"))
        roi_index = random.randint(0,len(roi_path_list)-1)
        self.roi_path = roi_path_list[roi_index]

        ## Get corresponding roi mask
        roi_file = self.roi_path.split("/")[-1]
        self.roi_mask_path = os.path.join(self.mask_dir,roi_file)
        roi_mask = cv2.imread(self.roi_mask_path)


        roi = cv2.imread(self.roi_path)
        return roi, roi_mask

    def Get_ROI_X2Y2_Padding(w,h):
        ##Pre-process the coordinate 
        h_r = int(h)
        print("h_r = {}".format(h_r))
        y_add = 0
        if h_r%2!=0:
            y_add = 1

        w_r = int(w)
        print("w_r = {}".format(w_r))
        x_add = 0
        if w_r%2!=0:
            x_add = 1
        return y_add,x_add

    def Check_And_Update_ROI_XY_In_Image_Boundary(self,x_c,y_c,final_roi_w,final_roi_h,x_add,y_add):
        ## keep the road sign in the images
        if x_c-int(final_roi_w/2.0)<=0:
            x_c = int(final_roi_w/2.0) + 1
        
        if x_c+int(final_roi_w/2.0)+x_add >= self.im.shape[1]:
            x_c = x_c - (int(final_roi_w/2.0)+x_add+1)

        if y_c-int(final_roi_h/2.0)<=0:
            y_c =  int(final_roi_h/2.0) + 1
        
        if y_c+int(final_roi_h/2.0)+y_add>=self.im.shape[0]:
            y_c = y_c-(int(final_roi_h/2.0)+y_add+1)

        return x_c, y_c

    def CopyPaste(self):
        for i in range(len(self.data_info)):
            ## Get image information
            self.im_path, self.dri_path, self.im, self.dri , self.vanish_y, self.label_path = self.data_info[i]

            ## Get stop sign ROI by random
            roi,roi_mask = self.Get_Random_ROI_And_ROIMask()

            ## Get the coordinate (x,y) and width, height , label of ROI that we want to copy-paset into image 
            l,x,y,w,h = self.Get_ROI_lxywh_In_Image(roi,roi_mask)

            ## Save corresponding yolo label.txt
            if self.args.save_txt:
                line = str(l) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
                with open(self.label_path,"a") as f:
                    f.write(line)
                    f.write("\n")
                f.close()
                return NotImplemented

            ## Save coptpasted image
            if self.args.save_img:
                if self.method == "opencv":
                    center = (x,y)
                    output = cv2.seamlessClone(roi, self.im, roi_mask, center, cv2.MIXED_CLONE)   #MIXED_CLONE
                    return NotImplemented
                elif self.method == "mask":
                    y_add,x_add = self.Get_ROI_X2Y2_Padding(w,h)

                    x_c,y_c = self.Check_And_Update_ROI_XY_In_Image_Boundary(x,y,w,y,x_add,y_add)

                    ## ROI Bounding Box (x1,y1): left-top point, (x2,y2): down-right point 
                    y1 = y_c - int(h/2.0)
                    y2 = y_c + int(h/2.0) + y_add
                    x1 = x_c - int(w/2.0) 
                    x2 = x_c + int(w/2.0) + x_add

                    img_roi = self.im[y1:y2,x1:x2]

                    roi_tmp = np.zeros(roi.shape, dtype=np.uint8)

                    ## processing stop sign ROI
                    roi_tmp[self.roi_mask>20] = roi[self.roi_mask>20] ## Fill stop sign forground
                    roi_tmp[self.roi_mask==0] = img_roi[self.roi_mask==0] ## Fill ROI background with image

                    ## "Copypaste" processed ROI into image
                    self.im[y1:y2,x1:x2] = roi_tmp
                    
                ## Save image
                os.makedirs(self.save_imdir,exist_ok=True)


                return NotImplemented

                return NotImplemented

            if self.args.show_img:
                return NotImplemented
            
            if self.args.show_roi:
                return NotImplemented
                
            
            #mask = self.Get_Possible_ROI_Position_Area()
            
        return NotImplementedError
        

        return NotImplementedError

    
