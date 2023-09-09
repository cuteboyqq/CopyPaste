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
        self.label_dir = args.label_dir
        self.img_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        


        self.data_info = []

        for im_path in self.img_path_list:
            self.Parse_path()
            dri_file = self.im_name + ".png"
            dri_path = os.path.join(self.dri_dir,dri_file)
            im = cv2.imread(im_path)
            dri = cv2.imread(dri_path)
            
            label_file = self.im_name +  ".txt"
            label_path = os.path.join(self.label_dir,label_file)

            vanish_y = 0
            get_vanish_y = False
            for i in range(dri.shape[0]):
                for j in range(dri.shape[1]):
                    if dri[i][j]!=0 and get_vanish_y==False:
                        vanish_y = i
                        get_vanish_y = True

            self.data_info.append([im_path,     dri_path,   im,     dri,    vanish_y,   label_path])
        
        # roi datasets
        self.roi_label = args.roi_label
        self.roi_dir = args.roi_dir
        self.mask_dir = args.mask_dir
        


        # Save
        self.save_imdir = args.save_imdir
        self.save_labeldir = args.save_labeldir

    def Parse_path(self):
        self.im  = self.im_path.split("//")[-1]
        self.im_name = self.im.split("//")[0] 

    def Get_Possible_ROI_Position_Area(self):
        #This are different to roi labels
        return NotImplemented

    def Get_ROI_XY_In_Image(self):
        # x = 0
        # y = 0
        # return (x,y)
        return NotImplementedError

    def Get_ROI_WH_In_Image(self):
        # w = 50
        # h = 50
        # return (w,h)
        return NotImplementedError

    def Get_ROI_Label(self):
    
        return NotImplementedError

    def Get_ROI_lxywh_In_Image(self,roi,roi_mask):
        x,y  = self.Get_ROI_XY_In_Image()
        w,h = self.Get_ROI_WH_In_Image(roi,roi_mask)
        label = self.Get_ROI_Label()
        return (label,x,y,w,h)


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

        ## left boundary
        if x_c-int(final_roi_w/2.0)<=0:
            x_c = int(final_roi_w/2.0) + 1
        
        ## right boundary
        if x_c+int(final_roi_w/2.0)+x_add >= self.im.shape[1]:
            x_c = x_c - (int(final_roi_w/2.0)+x_add+1)

        ## ceiling(Top) boundary
        if y_c-int(final_roi_h/2.0)<=0:
            y_c =  int(final_roi_h/2.0) + 1

        ## floor(Down) boundary
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
                ## normalize xywh
                x = str( int(float( (x) / self.im.shape[1] )*1000000)/1000000 ) 
                y = str( int(float( (y) / self.im.shape[0] )*1000000)/1000000 )
                w = str( int((roi.shape[1]/self.im.shape[1])*1000000)/1000000)
                h = str( int((roi.shape[0]/self.im.shape[0])*1000000)/1000000)
                line = str(l) + " " + x + " " + y + " " + w + " " +h

                add_line = str(l) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)

                ## Get corresponding label path
                img_file = self.im_path.split("/")[-1]
                img_filename = img_file.split(".")[0]

                label_file = img_filename+".txt"
                save_label_path = os.path.join(self.save_labeldir,label_file)

                ## open save label.txt
                f_new=open(save_label_path,"a")

                ## Copy original label.txt into save label.txt
                with open(self.label_path,"r") as f:
                    lines=f.readlines()
                    for line in lines:
                        f_new.write(line)
                    
                f.close()
                
                ## Add new stop sign label lxxywh into save label.txt
                f_new.write(add_line)
                f_new.write("\n")
                f_new.close()
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
                img_file = self.im_path.split("/")[-1]
                save_img_path = os.path.join(self.save_imdir,img_file)

                if self.method == "opencv":
                    cv2.imwrite(save_img_path,output)
                else:
                    cv2.imwrite(save_img_path,self.im)

                return NotImplemented

                return NotImplemented

            if self.args.show_img:
                return NotImplemented
            
            if self.args.show_roi:
                return NotImplemented

    
