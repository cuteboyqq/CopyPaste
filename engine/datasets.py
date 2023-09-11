import os
import numpy as np
import shutil
import random
import glob
import cv2

class BaseDatasets:
    def __init__(self,args):
        
        self.args = args

        self.label = args.roi_label #Stop sign label
        self.method = args.method
        
        # bdd100k datasets
        self.img_dir = args.img_dir
        self.dri_dir = args.dri_dir
        self.label_dir = args.label_dir
        #self.laneline_dir = args.laneline_dir
        self.img_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        #self.vanish_y = 300 ##  default vanish y

        ## other setting
        self.carhood_ratio = args.carhood_ratio
        self.num_img = args.num_img
        self.data_info = []
        cnt = 1
        for i in range(self.num_img+1):
            self.im_path = self.img_path_list[i]
            #self.im_path = im_path
            self.Parse_path()
            dri_file = self.im_name + ".png"
            dri_path = os.path.join(self.dri_dir,dri_file)
                
            label_file = self.im_name +  ".txt"
            label_path = os.path.join(self.label_dir,label_file)

            self.im = cv2.imread(self.im_path)
            self.dri = cv2.imread(dri_path)
            im_h = self.im.shape[0]
            vanish_y = 0
            x = int(self.im.shape[1]/2.0)
            while(self.dri[vanish_y][x][0]==0):
                if vanish_y+1< (im_h-1):
                    vanish_y+=1
                else:
                    break

            if vanish_y>= int(im_h * self.carhood_ratio):
                vanish_y = int(im_h * self.carhood_ratio) - 100
                
        
            self.vanish_y = vanish_y



            print("i = {}".format(i))
            self.data_info.append([self.im_path,     dri_path,    self.vanish_y,   label_path])
            cnt+=1
            
        # roi datasets
        self.roi_label = args.roi_label
        self.roi_dir = args.roi_dir
        self.mask_dir = args.mask_dir
        self.num_roi = args.num_roi
        #self.use_mask = args.use_mask

        # Save
        self.save_dir = args.save_dir
        self.save_img = args.save_img
        self.save_txt = args.save_txt
        os.makedirs(self.save_dir,exist_ok=True)
        self.show_roi = args.show_roi
        self.show_img = args.show_img

    def Parse_path(self):
        self.im  = self.im_path.split("/")[-1]
        self.im_name = self.im.split(".")[0]
        #print(self.im)
        #print(self.im_name)

    def Get_Possible_ROI_Position_Area(self):
        #This are different to roi labels
        return NotImplemented

    def Get_ROI_XY_In_Image(self):
        # x = 0
        # y = 0
        # return (x,y)
        return NotImplementedError

    def Get_ROI_WH_In_Image(self,roi,roi_mask,dri_path):
        # w = 50
        # h = 50
        # return (w,h)
        return NotImplementedError

    def Get_ROI_Label(self):
    
        return NotImplementedError

    def Get_ROI_lxywh_In_Image(self,roi,roi_mask, dri_path):
        x,y  = self.Get_ROI_XY_In_Image()
        w,h,roi,mask = self.Get_ROI_WH_In_Image(roi,roi_mask,dri_path)
        label = self.Get_ROI_Label()
        return (label,x,y,w,h,roi,mask)


    def Get_Random_ROI_And_ROIMask(self):
        ## Get random stop sign ROI
        roi_path_list = glob.glob(os.path.join(self.roi_dir,"*.jpg"))
        #print(roi_path_list)
        roi_index = random.randint(0,len(roi_path_list)-1)
        self.roi_path = roi_path_list[roi_index]

        roi_mask = None
        ## Get corresponding roi mask
        roi_file = self.roi_path.split("/")[-1]
        self.roi_mask_path = os.path.join(self.mask_dir,roi_file)

        # check if roi mask exists
        if os.path.exists(self.roi_mask_path):
            roi_mask = cv2.imread(self.roi_mask_path)
        else:
            roi_mask = None

        roi = cv2.imread(self.roi_path)
        # cv2.imshow("msak",roi_mask)
        # cv2.imshow("roi",roi)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        return roi, roi_mask

    def Get_ROI_X2Y2_Padding(self,w,h):
        ##Pre-process the coordinate 
        h_r = int(h)
        #print("h_r = {}".format(h_r))
        y_add = 0
        if h_r%2!=0:
            y_add = 1

        w_r = int(w)
        #print("w_r = {}".format(w_r))
        x_add = 0
        if w_r%2!=0:
            x_add = 1
        return y_add,x_add


    def Check_And_Update_ROI_XY_In_Image_Boundary(self,x_c,y_c,final_roi_w,final_roi_h,x_add,y_add):
        ## keep the road sign in the images
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

    def CopyPaste(self):
        #print(self.data_info)
        print(len(self.data_info))
        for i in range(len(self.data_info)):
            ## Get image information
            self.im_path, self.dri_path, self.vanish_y, self.label_path = self.data_info[i]


            self.im = cv2.imread(self.im_path)
            self.dri = cv2.imread(self.dri_path)

            ## Get stop sign ROI by random
            roi,roi_mask = self.Get_Random_ROI_And_ROIMask()
            

            ## Get the coordinate (x,y) and width, height , label of ROI that we want to copy-paset into image
            #  
            l,x,y,w,h,roi,roi_mask = self.Get_ROI_lxywh_In_Image(roi,roi_mask, self.dri_path)
            print("{},{},{},{},{}".format(l,x,y,w,h))

            y_add,x_add = self.Get_ROI_X2Y2_Padding(w,h)
            x_c,y_c = self.Check_And_Update_ROI_XY_In_Image_Boundary(x,y,w,h,x_add,y_add)
            x = x_c
            y = y_c
            if os.path.exists(self.label_path):
                

                ## Save coptpasted image
                if True:
                    IS_FAILED=False
                    if self.method=="both":
                        choose = random.randint(1,2)
                        if choose==1:
                            self.method = "opencv"
                        else:
                            self.method = "mask"

                    print("save image")
                    if self.method == "opencv":
                        print("opencv method~~")
                        #input()
                        center = (x,y)
                        mask = 255 * np.ones(roi.shape, roi.dtype)
                        print("roi.shape:{}".format(roi.shape))
                        print("roi.dtype:{}".format(roi.dtype))
                        try:
                            output = cv2.seamlessClone(roi, self.im, mask, center, cv2.MIXED_CLONE)   #MIXED_CLONE
                        except:
                            IS_FAILED=True
                            pass
                        #return NotImplemented
                    elif self.method == "mask":
                        print("mask method~~")
                        #input()
                        ## ROI Bounding Box (x1,y1): left-top point, (x2,y2): down-right point 
                        y1 = y - int(h/2.0)
                        y2 = y + int(h/2.0) + y_add
                        x1 = x - int(w/2.0) 
                        x2 = x + int(w/2.0) + x_add
                        print("y:{},x:{}".format(y,x))
                        print("h:{} w:{}".format(h,w))
                        img_roi = self.im[y1:y2,x1:x2]
                        print("img_roi:")
                        print(img_roi.shape)
                        print("roi_mask:")
                        print(roi_mask.shape)
                        roi_tmp = np.zeros(roi.shape, dtype=np.uint8)

                        
                        ## processing stop sign ROI
                        roi_tmp[roi_mask>30] = roi[roi_mask>30] ## Fill stop sign forground
                        roi_tmp[roi_mask<=30] = img_roi[roi_mask<=30] ## Fill ROI background with image

                        ## "Copypaste" processed ROI into image
                        self.im[y1:y2,x1:x2] = roi_tmp
                    
                    ## Save image
                    save_im_dir = os.path.join(self.save_dir,"images")
                    os.makedirs(save_im_dir,exist_ok=True)
                    img_file = self.im_path.split("/")[-1]
                    save_img_path = os.path.join(save_im_dir,img_file)
                    if IS_FAILED==False:
                        if self.method == "opencv":
                            print("save image")
                            cv2.imwrite(save_img_path,output)
                        else:
                            print("mask method save image")
                            cv2.imwrite(save_img_path,self.im)

                    #return NotImplemented

                    #return NotImplemented
                ## Save corresponding yolo label.txt
                if self.save_txt and IS_FAILED==False:
                    print("save txt")
                    ## normalize xywh
                    x_s = str( int(float( (x) / self.im.shape[1] )*1000000)/1000000 ) 
                    y_s = str( int(float( (y) / self.im.shape[0] )*1000000)/1000000 )
                    w_s = str( int((roi.shape[1]/self.im.shape[1])*1000000)/1000000)
                    h_s = str( int((roi.shape[0]/self.im.shape[0])*1000000)/1000000)
                    add_line = str(l) + " " + x_s + " " + y_s + " " + w_s + " " + h_s

                    ## Get corresponding label path
                    img_file = self.im_path.split("/")[-1]
                    img_filename = img_file.split(".")[0]

                    label_file = img_filename+".txt"
                    save_label_dir = os.path.join(self.save_dir,"labels")
                    os.makedirs(save_label_dir,exist_ok=True)
                    save_label_path = os.path.join(save_label_dir,label_file)

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
                    #return NotImplemented
                if self.show_img:
                    continue
                if self.show_roi:
                    continue
    
    def CopyPasteSimple(self):
        for i in range(len(self.data_info)):
            ## Get image information
            self.im_path, self.dri_path, self.vanish_y, self.label_path = self.data_info[i]

            self.im = cv2.imread(self.im_path)
            self.dri = cv2.imread(self.dri_path)

            ## Get stop sign ROI by random
            if self.num_roi == 1:
                roi,roi_mask = self.Get_Random_ROI_And_ROIMask()
            else:
                roi_roimask = []
                for i in range(self.num_roi):
                    roi,roi_mask = self.Get_Random_ROI_And_ROIMask()
                    roi_roimask.append([roi, roi_mask])

            ## Get the coordinate (x,y) and width, height , label of ROI that we want to copy-paset into image
            #
            COPY_ORI_TXT_DONE = False
            for i in range(len(roi_roimask)):

                roi = roi_roimask[i][0]
                roi_mask = roi_roimask[i][1]

                l,x,y,w,h,roi,roi_mask = self.Get_ROI_lxywh_In_Image(roi,roi_mask, self.dri_path)
                print("{},{},{},{},{}".format(l,x,y,w,h))

                y_add,x_add = self.Get_ROI_X2Y2_Padding(w,h)
                x_c,y_c = self.Check_And_Update_ROI_XY_In_Image_Boundary(x,y,w,h,x_add,y_add)
                x = x_c
                y = y_c
                IS_FAILED = False
                if os.path.exists(self.label_path):
                    ## ROI Bounding Box (x1,y1): left-top point, (x2,y2): down-right point 
                    y1 = y - int(h/2.0)
                    y2 = y + int(h/2.0) + y_add
                    x1 = x - int(w/2.0) 
                    x2 = x + int(w/2.0) + x_add
                    print("y:{},x:{}".format(y,x))
                    print("h:{} w:{}".format(h,w))

                    try:
                        ## "Copypaste" processed ROI into image
                        self.im[y1:y2,x1:x2] = roi
                    except:
                        IS_FAILED=True
                        pass
                    
                    ## Save image
                    if IS_FAILED==False:
                        save_im_dir = os.path.join(self.save_dir,"images")
                        os.makedirs(save_im_dir,exist_ok=True)
                        img_file = self.im_path.split("/")[-1]
                        save_img_path = os.path.join(save_im_dir,img_file)
                    
                        print("save image")
                        cv2.imwrite(save_img_path,self.im)
                    

                    
                    ## Save corresponding yolo label.txt
                    if self.save_txt and IS_FAILED==False:
                        print("save txt")
                        ## normalize xywh
                        x_s = str( int(float( (x) / self.im.shape[1] )*1000000)/1000000 ) 
                        y_s = str( int(float( (y) / self.im.shape[0] )*1000000)/1000000 )
                        w_s = str( int((roi.shape[1]/self.im.shape[1])*1000000)/1000000)
                        h_s = str( int((roi.shape[0]/self.im.shape[0])*1000000)/1000000)
                        add_line = str(l) + " " + x_s + " " + y_s + " " + w_s + " " + h_s

                        ## Get corresponding label path
                        img_file = self.im_path.split("/")[-1]
                        img_filename = img_file.split(".")[0]

                        label_file = img_filename+".txt"
                        save_label_dir = os.path.join(self.save_dir,"labels")
                        os.makedirs(save_label_dir,exist_ok=True)
                        save_label_path = os.path.join(save_label_dir,label_file)

                        ## open save label.txt
                        f_new=open(save_label_path,"a")

                        ## Copy original label.txt into save label.txt
                        if COPY_ORI_TXT_DONE==False:
                            with open(self.label_path,"r") as f:
                                lines=f.readlines()
                                for line in lines:
                                    f_new.write(line)
                            COPY_ORI_TXT_DONE=True
                            f.close()
                        
                        ## Add new stop sign label lxxywh into save label.txt
                        f_new.write(add_line)
                        f_new.write("\n")
                        f_new.close()
                        #return NotImplemented
                    if self.show_img:
                        continue
                    if self.show_roi:
                        continue