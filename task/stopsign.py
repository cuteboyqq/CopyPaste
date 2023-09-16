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

        ## not overlapped with other bounding box
        if os.path.exists(self.label_path):
            with open(self.label_path,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_list = line.split(" ")
                    label = line_list[0]
                    x = int(float(line_list[1])*self.im.shape[1])
                    y = int(float(line_list[2])*self.im.shape[0])
                    w = int(float(line_list[3])*self.im.shape[1])
                    h = int(float(line_list[4])*self.im.shape[0])
                    
                    mask[y-int(h/2.0):y+int(h/2.0),x-int(w/2.0):x+int(w/2.0)] = (255,255,255)

        return mask


    def Get_ROI_XY_In_Image(self):
        mask = self.Get_Possible_ROI_Position_Area()
        
        ## initial random (x,y)
        x = random.randint(0,self.im.shape[1]-1)
        y = random.randint(self.vanish_y - 50,int(self.im.shape[0]*self.carhood_ratio))

        ## Stop should put at non-drivable area
        cnt = 1
        while(mask[y][x][0]!=0): # Exist while when (x,y) is in non-drivable area
            x = random.randint(0,self.im.shape[1]-1)
            y = random.randint(self.vanish_y - 50,int(self.im.shape[0]*self.carhood_ratio))
            cnt+=1
            if cnt==100:
                break
        self.roi_x  = x
        self.roi_y  = y
        return (x,y)
        return NotImplementedError

    

    def Get_ROI_WH_In_Image(self,roi,roi_mask,dri_path):
        OVERLAPPED=True
        cnt = 1
        while(OVERLAPPED and cnt<=100):
            ## small stop sign at top of image
            if self.roi_y < self.vanish_y:
                self.roi_w = random.randint(20,50)
                self.roi_h = int(roi.shape[0]*(self.roi_w/roi.shape[1]))
            else: ## small~big stop sign at bottom of image
                self.roi_w = int(roi.shape[1]*float(random.randint(2,20)*0.1))
                self.roi_h = int(roi.shape[0]*(self.roi_w/roi.shape[1]))
            
            ## check if roi w is too large
            if self.roi_w > self.roi_th:
                roi_w_pre = self.roi_w
                self.roi_w = self.roi_th
                self.roi_h = int(self.roi_h * float(self.roi_th/roi_w_pre))
                print("case 1 : self.roi_w > {}".format(self.roi_th))
                print("after correct")
                print("self.roi_w:{}".format(self.roi_w))
                print("self.roi_h:{}".format(self.roi_h))
            
            ## check if roi h is too large 
            if self.roi_h > self.roi_th:
                roi_h_pre = self.roi_h
                self.roi_h = self.roi_th
                self.roi_w = int(self.roi_w * float(self.roi_th/roi_h_pre))
                print("case 2 : self.roi_h > {}".format(self.roi_th))
                print("after correct")
                print("self.roi_w:{}".format(self.roi_w))
                print("self.roi_h:{}".format(self.roi_h))

            # if cnt>50:
            #     roi_h_pre = self.roi_h
            #     self.roi_h = 60
            #     self.roi_w = int(self.roi_w * float(60/roi_h_pre))
            #     print("case 3: cnt>100")
            #     print("after correct")
            #     print("self.roi_w:{}".format(self.roi_w))
            #     print("self.roi_h:{}".format(self.roi_h))

            xywh_stop_sign = [self.roi_x,self.roi_y,self.roi_w,self.roi_h]
            print("xywh_stop_sign:{}".format(xywh_stop_sign))
            xyxy_stop_sign = self.xywh_to_xyxy(xywh_stop_sign)

            
            ## not overlapped with other bounding box
            im,im_name = self.Parse_path_2(self.label_path)
            label_txt = im_name + ".txt"
            save_txt_path = os.path.join(self.save_dir,"labels",label_txt)

            if os.path.exists(save_txt_path):
                open_path = save_txt_path
                #print("open_path = {}".format(open_path))
                #input()
            else:
                open_path = self.label_path
           
            
            #print(open_path)
            #input()
            if os.path.exists(open_path):
                with open(open_path,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line_list = line.split(" ")
                        label = line_list[0]
                        x = int(float(line_list[1])*self.im.shape[1])
                        y = int(float(line_list[2])*self.im.shape[0])
                        w = int(float(line_list[3])*self.im.shape[1])
                        h = int(float(line_list[4])*self.im.shape[0])
                        
                        bb_bdd_xywh = [x,y,w,h]
                        bb_bdd_xyxy = self.xywh_to_xyxy(bb_bdd_xywh)    
                        
                        bb_stopsign = {"x1":xyxy_stop_sign[0],
                                        "x2":xyxy_stop_sign[2],
                                        "y1":xyxy_stop_sign[1],
                                        "y2":xyxy_stop_sign[3]}
                        
                        bb_bdd100k = {"x1":bb_bdd_xyxy[0],
                                        "x2":bb_bdd_xyxy[2],
                                        "y1":bb_bdd_xyxy[1],
                                        "y2":bb_bdd_xyxy[3]}
                        
                        print("bb_stopsign:{}".format(bb_stopsign))
                        print("bb_bdd100k:{}".format(bb_bdd100k))

                        iou = self.get_iou(bb_stopsign,bb_bdd100k)
                        print("iou={}".format(iou))
                        #input()
                        if iou > self.overlap_th:
                            OVERLAPPED=True
                            IS_FAILED=True
                            print("case 1 : iou={} > {}".format(iou,self.overlap_th))
                            
                            
                            cnt+=1
                            print("cnt={}".format(cnt))
                            #input()
                        else:
                            OVERLAPPED=False
                            print("case 2 : iou={}".format(iou))
                            print("cnt={}".format(cnt))
                            #input()
            else:
                print("label is not exists !!")
                OVERLAPPED=False
                cnt = 9999
                print("cnt = {}".format(cnt))        

            cnt+=1
            print("cnt = {}".format(cnt))    

        if cnt>100:
                roi_h_pre = self.roi_h
                self.roi_h = 30
                self.roi_w = int(self.roi_w * float(60/roi_h_pre))
                print("case 3: cnt>100")
                print("after correct")
                print("self.roi_w:{}".format(self.roi_w))
                print("self.roi_h:{}".format(self.roi_h))

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



    
                
            
            
            
        



    
