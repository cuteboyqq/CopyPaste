import sys
sys.path.append('..')

import glob
import os
import shutil
import cv2
import numpy as np

## how the argsparse list
#https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse

class LabelDatasets:
    def __init__(self,args):

        ## BDD100K Dataset directory
        self.img_dir = args.img_dir
        self.label_dir = args.label_dir
        self.drivable_dir = args.drivable_dir
        self.lane_dir = args.lane_dir

        ## Merge labels
        self.datasetA_dir = args.datasetA_dir
        self.datasetB_dir = args.datasetB_dir
        ## task parameters
        self.remove_label_list = ['9']#args.remove_labellist

        self.label_path_list = glob.glob(os.path.join(self.label_dir,"*.txt"))

        ## Save directory
        self.save_dir = args.save_dir
        self.save_img_dir = os.path.join(self.save_dir,"images")
        self.save_txt_dir = os.path.join(self.save_dir,"labels")

        os.makedirs(self.save_dir,exist_ok=True)
        os.makedirs(self.save_img_dir,exist_ok=True)
        os.makedirs(self.save_txt_dir,exist_ok=True)


        self.dilation_kernel = 7
        self.process_type = args.process_type

        # lane_mapping: {0: 2,            1: lane_bg,     2: 1,           3: 3,           4: 5, 
        #         5: lane_bg,     6: 2,           7: 3,           8: lane_bg,     9: lane_bg,
        #         10: lane_bg,    11: lane_bg,    12: lane_bg,    13: lane_bg,    14: lane_bg,
        #         15: lane_bg,    16: 2,          17: lane_bg,    18: 1,          19: 3,
        #         20: 5,          21: lane_bg,    22: 2,          23: 3,          24: lane_bg,
        #         25: lane_bg,    26: lane_bg,    27: lane_bg,    28: lane_bg,    29: lane_bg,
        #         30: lane_bg,    31: lane_bg,    32: 4,          33: lane_bg,    34: lane_bg,  
        #         35: lane_bg,    36: 5,          37: lane_bg,    38: 4,          39: lane_bg,
        #         40: lane_bg,    41: lane_bg,    42: lane_bg,    43: lane_bg,    44: lane_bg,
        #         45: lane_bg,    46: lane_bg,    47: lane_bg,    48: 4,          49: lane_bg,
        #         50: lane_bg,    51: lane_bg,    52: 5,          53: lane_bg,    54: 4,
        #         55: lane_bg,
        #         255: lane_bg}

        self.lane_mapping = {0:255,  1:2,  2:0,  3:3,   4:32,   5:4}


    def parse_path(self,path):
        file = path.split("/")[-1]
        file_name = file.split(".")[0]
        return file,file_name
        

    def RemoveLabelsInLabelTXT(self):
        print(self.label_path_list)
        for label_path in self.label_path_list:
            #print(label_path)
            label_file,_ = self.parse_path(label_path)
            save_txt_path = os.path.join(self.save_txt_dir,label_file)
            print(save_txt_path)
            f_new = open(save_txt_path,'a')
            with open(label_path,"r") as f:
                lines = f.readlines()
                for line in lines:
                    if not str(line.split(" ")[0]) in self.remove_label_list:
                        f_new.write(line)

            f.close()
            f_new.close()

    def SplitAllImagesToSpitFolders(self):
        '''
        func:SplitAllImagesToSpitFolders
        input:
            self.img_dir
            self.save_dir
        output:
            image in split folders
        '''
        img_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        #print(img_path_list)
        num_images = len(img_path_list)
        count = 1
        folder_label = 1

        for i in range(len(img_path_list)):
            save_folder = "train_" + str(int((i+1)/2500)+1)
            img_path = img_path_list[i]
            file,file_name = self.parse_path(img_path)
            save_dir = os.path.join(self.save_img_dir,save_folder)
            os.makedirs(save_dir,exist_ok=True)
            shutil.copy(img_path,save_dir)
            print("{} : copy successful".format(i))
            i+=1
    
    def parse_path_2(self,path):
        local_path = path.split(os.path.dirname(self.img_dir))[1]
        local_path = "./images" + local_path
        return local_path

    def Get_datasets_img_path_txt(self):

        train_img_path_list = glob.glob(os.path.join(self.img_dir,"train","*.jpg"))
        #print(train_img_path_list)
        if len(train_img_path_list)==0:
            train_img_path_list = glob.glob(os.path.join(self.img_dir,"train","*.png"))
        val_img_path_list = glob.glob(os.path.join(self.img_dir,"val","*.jpg"))
        if len(val_img_path_list)==0:
            val_img_path_list = glob.glob(os.path.join(self.img_dir,"val","*.png"))

        train_f = open("train_new_2023-10-19.txt","a")
        for i in range(len(train_img_path_list)):
            local_path = self.parse_path_2(train_img_path_list[i])
            print("train {}:{}".format(i,local_path))
            train_f.write(local_path)
            train_f.write("\n")

        train_f.close()


        val_f = open("val_new_2023-10-19.txt","a")
        for i in range(len(val_img_path_list)):
            local_path = self.parse_path_2(val_img_path_list[i])
            print("val {}:{}".format(i,local_path))
            val_f.write(local_path)
            val_f.write("\n")

        val_f.close()
    
    ## func: process_lane_map
    ## Erode the maps
    def process_lane_map(self):
        lane_map_path_list = glob.glob(os.path.join(self.lane_dir,self.process_type,"*.png"))
        for i in range(len(lane_map_path_list)):
            print(lane_map_path_list[i])
            print(len(lane_map_path_list))
            mask = cv2.imread(lane_map_path_list[i])
            #cv2.imshow("mask",mask)
            #cv2.waitKey(2000)
            #cv2.destroyAllWindows()
            eroded_mask = self.split_map_and_erode(mask)
            file,file_name = self.parse_path(lane_map_path_list[i])
            new_folder_name = "eroded_" + self.process_type 
            save_dir = os.path.join(self.save_dir,"labels","lane",new_folder_name,"train")
            os.makedirs(save_dir,exist_ok=True)
            #print("save_path :{}".format(save_path))
            cv2.imwrite(save_dir+"/"+file_name+".png",eroded_mask)
            print("{}:save new mask succesful".format(i))

    ## Do Erosion
    def split_map_and_erode(self,mask):
        #print("mask:{}".format(mask.shape)) #900,600,3
        """Split mask to n-channels map where n is equal to the number of foreground classes that needs to be dilated."""
        kernel = np.ones((self.dilation_kernel, self.dilation_kernel), dtype=np.uint8)
        cls = np.unique(mask)
        #print(cls)
        #print(len(cls))
        if len(cls) >= 2:   # if there is foreground
            #print("len(cls) > 2 ~~~")
            cls = cls[1:]   # ignore background
            if isinstance(cls, int):
                cls = [cls]
            
            layers = np.zeros((len(cls), mask.shape[0], mask.shape[1]), dtype=np.uint8)
            #print("layers:{}".format(layers.shape)) # 7,1600,900
            for i, lb in enumerate(cls):
                    #print("lb:{}".format(lb))
                    layers[i][mask[:,:,0] == lb] = 255     # trick to dilate
                    layers[i] = cv2.erode(layers[i], kernel, iterations=2)
                    layers[i][np.where(layers[i] == 255)] = lb
                    
            
            outputs = np.zeros_like(mask, dtype=np.uint8)
            # for j in self.lane_order:   # set it by priority
            for j in cls:
                outputs[np.where(layers == j)[1:]] = j
            return outputs
        return mask
    

    def Convert_image_format(self,type=None):
        
        # if type == "images" or "image":
        #     print(self.img_dir)
        #     img_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        #elif type == "drivable":
        print(self.drivable_dir)
        img_path_list = glob.glob(os.path.join(self.drivable_dir,"*.jpg"))
        # elif type == "lane":
        #     print(self.lane_dir)
        #     img_path_list = glob.glob(os.path.join(self.lane_dir,"*.jpg"))

        for i in range(len(img_path_list)):
            print(img_path_list[i])
            im = cv2.imread(img_path_list[i])
            file,filename = self.parse_path(img_path_list[i])
            save_im = filename + ".png"
            save_im_path = os.path.join(self.save_dir,save_im)
            os.makedirs(self.save_dir,exist_ok=True)
            cv2.imwrite(save_im_path,im)
            print("{}:save to .png successful".format(i))

    def Convert_labels(self):
        print(self.lane_dir)
        img_path_list = glob.glob(os.path.join(self.lane_dir,"*.png"))
        
        for i in range(len(img_path_list)):
            print(img_path_list[i])
            mask = cv2.imread(img_path_list[i])
            out = self.map_segment_labels(mask)
            file,filename = self.parse_path(img_path_list[i])
            save_im = filename + ".png"
            save_im_path = os.path.join(self.save_dir,save_im)
            os.makedirs(self.save_dir,exist_ok=True)
            cv2.imwrite(save_im_path,out)
            print("{}:Convert_labels successful".format(i))
        return NotImplemented
    
    def map_segment_labels(self,mask):
        """Update the segmentation mask labels by the mapping variable."""
        out = np.empty((mask.shape), dtype=mask.dtype)
        for k, v in self.lane_mapping.items():
            #if isinstance(v, str):
            #    out[mask == k] = bg_lb
            #else:
            out[mask == k] = v
        return out

    

    def CorrectNumberOfImagesAndLablesInDatasets(self):
        img_path_list = glob.glob(os.path.join(self.img_dir,"**","*.jpg"))
        #label_path_list = os.path.join(self.label_dir,"*.txt")
        #drivable_path_list = os.path.join(self.drivable_dir,"*.png")

        ## Create save folder
        save_img_dir = os.path.join(self.save_dir,"images","train")
        os.makedirs(save_img_dir,exist_ok=True)

        save_label_dir = os.path.join(self.save_dir,"labels","detection","train")
        os.makedirs(save_label_dir,exist_ok=True)

        save_lane_mask_dir = os.path.join(self.save_dir,"labels","lane","masks","train")
        os.makedirs(save_lane_mask_dir,exist_ok=True)

        save_lane_colormap_dir = os.path.join(self.save_dir,"labels","lane","colormaps","train")
        os.makedirs(save_lane_colormap_dir,exist_ok=True)

        save_drivable_mask_dir = os.path.join(self.save_dir,"labels","drivable","masks","train")
        os.makedirs(save_drivable_mask_dir,exist_ok=True)

        save_drivable_colormap_dir = os.path.join(self.save_dir,"labels","drivable","colormaps","train")
        os.makedirs(save_drivable_colormap_dir,exist_ok=True)
        c = 1
        for img_path in  img_path_list:
            file,file_name = self.parse_path(img_path)
            print("{}:{}".format(c,file))
            c+=1
            label = file_name + ".txt"
            label_path = os.path.join(self.label_dir,"train",label)

            ## check label.txt exist or not
            if not os.path.exists(label_path):
                print("label_path not exists :{}".format(label_path))
                continue
            drivable_label = file_name + ".jpg"
            drivable_colormap_path = os.path.join(self.drivable_dir,"colormaps","train",drivable_label)

            drivable_mask_path = os.path.join(self.drivable_dir,"masks","train",drivable_label)
            ## check drivable map exist or not
            if not os.path.exists(drivable_colormap_path):
                print("drivable_colormap_path not exists :{}".format(drivable_colormap_path))
                #co = cv2.imread(drivable_colormap_path)
                #cv2.imshow("co",co)
                #cv2.waitKey(2000)
                continue
            
            lane_label_jpg = file_name + ".jpg"
            lane_colormap_path = os.path.join(self.lane_dir,"colormaps","train",lane_label_jpg)
            lane_label_png = file_name + ".jpg"
            lane_mask_path = os.path.join(self.lane_dir,"masks","train",lane_label_png)
            ## check lane map exist or not
            if not os.path.exists(lane_colormap_path):
                print("lane_colormap_path not exists:{}".format(lane_colormap_path))
                continue
            
            try:
                ## Copy images/labels
                #img
                shutil.copy(img_path,save_img_dir)
                #label.txt
                shutil.copy(label_path,save_label_dir)

                #drivable_colormap.png
                shutil.copy(drivable_colormap_path,save_drivable_colormap_dir)
                #drivable_mask.png
                shutil.copy(drivable_mask_path,save_drivable_mask_dir)

                #lane_colormap.png
                shutil.copy(lane_colormap_path,save_lane_colormap_dir)
                #lane_mask.png
                shutil.copy(lane_mask_path,save_lane_mask_dir)
                print("copy successful !")
            except:
                print("copy error !")
                pass
    
    def Parse_Path(self,path):
        file = path.split(os.sep)[-1]
        return file
    
    def Get_Wanted_Label_From_DatasetA_And_Put_Into_DatasetB(self):
        """
        2023-10-19
        input:
            self.datasetA_dir, for example: "/home/ali/Projects/GitHub_Code/ali/CopyPaste/tools/bdd100k_data_pedestrain_2023-10-19/infer_result"
            self.datasetB_dir, for example: "/home/ali/Projects/GitHub_Code/ali/CopyPaste/tools/nuImage_data_include_pedestrain/labels/detection/train"
            self.wanted_label_list
            self.save_dir
        output:
            merged datasetC
        """
        
          #V0.2.10 YOLO-ADAS
            # 0: perdestrian
            # 1: small vehicle
            # 2: big vehicle
            # 3: train
            # 4: traffic light
            # 5: traffic sign
            # 6: stop sign
            # 7: lane marking
           # BDD100K Object Detection Labels
            # 0: pedestrian
            # 1: rider
            # 2: car
            # 3: truck
            # 4: bus
            # 5: train
            # 6: motorcycle
            # 7: bicycle
            # 8: traffic light
            # 9: traffic sign
            # 10: stop sign
            # 11: lane marking
        det_mapping = {4:8, 5:9, 6:10, 7:11}
        datasetA_label_path_list = glob.glob(os.path.join(self.datasetA_dir,"***","**","*.txt"))
        #print(datasetA_label_path_list)
        for i in range(len(datasetA_label_path_list)):
            #print(datasetA_label_path_list[i])
            label = self.Parse_Path(datasetA_label_path_list[i])
            print(label)

            datasetB_label_path = os.path.join(self.datasetB_dir,label)
            if os.path.exists(datasetB_label_path):
                print("{} exist !!".format(datasetB_label_path))
                ## open save label_path
                save_label_dir = os.path.join(self.save_dir,"labels","detection","train")
                os.makedirs(save_label_dir,exist_ok=True)
                save_label_path = os.path.join(save_label_dir,label)
                f_save = open(save_label_path,"a")

                ## Copy original labels from datasetB
                with open(datasetB_label_path,"r") as f_b:
                    lines = f_b.readlines()
                    for line in lines:
                        f_save.write(line)
                print("{} : Copy original labels from datasetB complete !!".format(i))
                ## Copy wanted label from datasetA
                with open(datasetA_label_path_list[i],"r") as f_a:
                    lines = f_a.readlines()
                    for line in lines:
                        #print(line)
                        la_str = line.split(" ")[0]
                        if la_str=="4" or la_str=="5" or la_str=="6" or la_str=="7":
                            BDD100K_label = det_mapping[int(la_str)]
                            new_line = str(BDD100K_label) + ' ' \
                                        + line.split(" ")[1] + ' '\
                                        + line.split(" ")[2] + ' '\
                                        + line.split(" ")[3] + ' '\
                                        + line.split(" ")[4]
                            #print(new_line)
                            f_save.write(new_line)
                            #print("find wnated label, write to label.txt succesful")
                            #input()
                print(" {} : Copy wanted label from datasetA complete !!".format(i))
                f_save.close()

        return NotImplemented


    def Get_Label_Path(self,im_path):
        '''
        the dataset format is  images/100k/train
                                images/100k/val
                                images/100k/test
        input : image path
        output : label path
        
        '''
        USE_BDD100K_DATA = True

        im_dir = os.path.dirname(im_path)
        label_dir = "/labels/detection".join(im_dir.split("/images"))
        if USE_BDD100K_DATA:
            label_dir = "".join(label_dir.split("/100k"))
        im = os.path.basename(im_path)
        label = im.split(".")[0] + ".txt"
        label_path = os.path.join(label_dir,label)


        dri_mask_dir = "/labels/drivable/masks".join(im_dir.split("/images"))
        if USE_BDD100K_DATA:
            dri_mask_dir = "".join(dri_mask_dir.split("/100k"))
        dri_mask = im.split(".")[0] + ".png"
        dri_mask_path = os.path.join(dri_mask_dir,dri_mask)


        dri_colormap_dir = "/labels/drivable/colormaps".join(im_dir.split("/images"))
        if USE_BDD100K_DATA:
            dri_colormap_dir = "".join(dri_colormap_dir.split("/100k"))
        dri_colormap = im.split(".")[0] + ".png"
        dri_colormap_path = os.path.join(dri_colormap_dir,dri_colormap)


        lane_mask_dir = "/labels/lane/masks".join(im_dir.split("/images"))
        if USE_BDD100K_DATA:
            lane_mask_dir = "".join(lane_mask_dir.split("/100k"))
        lane_mask = im.split(".")[0] + ".png"
        lane_mask_path = os.path.join(lane_mask_dir,lane_mask)


        lane_colormap_dir = "/labels/lane/colormaps".join(im_dir.split("/images"))
        if USE_BDD100K_DATA:
            lane_colormap_dir = "".join(lane_colormap_dir.split("/100k"))
        lane_colormap = im.split(".")[0] + ".png"
        lane_colormap_path = os.path.join(lane_colormap_dir,lane_colormap)

        return label_path,dri_mask_path,dri_colormap_path,lane_mask_path,lane_colormap_path,label,dri_mask,lane_mask


    def Get_Pedestrain_Dataset(self):
        '''
        func: Get_Pedestrain_Dataset
            input: 
                self.img_dir (image directory), For example : "/home/ali/Projects/datasets/BDD100K-ori/images/100k/train"
                self.save_dir (save new datasets directory), For example : "../tools/bdd100k_data_pedestrain_2023-10-18"

            return: 
                BDD100K datasets include pedestrain labels

            parameters:
                INCLUDE_PEDESTRIAN_LABEL_METHOD
                FILTER_OUT_ONLY_VEHICLE_LABEL_METHOD
        '''
        im_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        #print(im_path_list)
        c = 1
        f_p = 1
        for i in range(len(im_path_list)):
            # print("{}:{}".format(i+1,im_path_list[i]))
            label_exist = False
            label_path,dri_mask_path,dri_colormap_path,lane_mask_path,lane_colormap_path,label,dri_mask,lane_mask = self.Get_Label_Path(im_path_list[i])
            # print("label_path {}:{}".format(c,label_path))
            # print("dri_mask_path {}:{}".format(c,dri_mask_path))
            # print("lane_mask_path {}:{}".format(c,lane_mask_path))
            find_wanted_label = False
          

            # BDD100K Object Detection Labels
            # 0: pedestrian
            # 1: rider
            # 2: car
            # 3: truck
            # 4: bus
            # 5: train
            # 6: motorcycle
            # 7: bicycle
            # 8: traffic light
            # 9: traffic sign
            # 10: stop sign
            # 11: lane marking
            INCLUDE_PEDESTRIAN_LABEL_METHOD=False
            FILTER_OUT_ONLY_VEHICLE_LABEL_METHOD=True
           
            if os.path.exists(label_path):
                # print("label_path {} exist".format(label_path))
                with open(label_path,"r") as f:
                    lines = f.readlines()
                    for line in lines:
                        # print(line)
                        la_str = line.split(" ")[0]
                        if INCLUDE_PEDESTRIAN_LABEL_METHOD:
                            if la_str=="0" or la_str=="1":
                                find_wanted_label = True
                        elif FILTER_OUT_ONLY_VEHICLE_LABEL_METHOD:
                            if la_str=="0" or la_str=="1" or la_str=="6" or la_str=="7" or la_str=="10" or la_str=="11":
                                find_wanted_label = True
            else:
                print("{} does not exist!!".format(label_path))
                
            if find_wanted_label:
                save_im_dir = os.path.join(self.save_dir,"images","train")
                os.makedirs(save_im_dir,exist_ok=True)
                shutil.copy(im_path_list[i],save_im_dir)

                save_label_dir = os.path.join(self.save_dir,"labels","detection","train")
                os.makedirs(save_label_dir,exist_ok=True)
                if not os.path.exists(os.path.join(save_label_dir,label)):
                    shutil.copy(label_path,save_label_dir)
                else:
                    label_exist=True
                save_dri_mask_dir = os.path.join(self.save_dir,"labels","drivable","masks","train")
                os.makedirs(save_dri_mask_dir,exist_ok=True)
                if not os.path.exists(os.path.join(save_dri_mask_dir,dri_mask)):
                    shutil.copy(dri_mask_path,save_dri_mask_dir)
                else:
                    label_exist=True
                save_dri_colormap_dir = os.path.join(self.save_dir,"labels","drivable","colormaps","train")
                os.makedirs(save_dri_colormap_dir,exist_ok=True)
                if not os.path.exists(os.path.join(save_dri_colormap_dir,dri_mask)):
                    shutil.copy(dri_colormap_path,save_dri_colormap_dir)
                else:
                    label_exist=True

                save_lane_mask_dir = os.path.join(self.save_dir,"labels","lane","masks","train")
                os.makedirs(save_lane_mask_dir,exist_ok=True)
                if not os.path.exists(os.path.join(save_lane_mask_dir,lane_mask)):
                    shutil.copy(lane_mask_path,save_lane_mask_dir)
                else:
                    label_exist=True
                save_lane_colormaps_dir = os.path.join(self.save_dir,"labels","lane","colormaps","train")
                os.makedirs(save_lane_colormaps_dir,exist_ok=True)
                if not os.path.exists(os.path.join(save_lane_colormaps_dir,lane_mask)):
                    shutil.copy(lane_colormap_path,save_lane_colormaps_dir)
                else:
                    label_exist=True
                print("{}:saved image including pedestrain successfully".format(f_p))

                if label_exist:
                    print("label_exist = {},pass~~~~~~".format(label_exist))
                f_p+=1
            c+=1

def get_args_label():
    import argparse
    parser = argparse.ArgumentParser()
    ##   BDD100k datasets directory
    # parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/BDD100K-ori/images/100k/train")
    # parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/BDD100K-ori/labels/detection/train")
    # parser.add_argument('-drivabledir','--drivable-dir',help='drivable label dir', \
    #                     default="/home/ali/Projects/datasets/BDD100K-ori/labels/drivable/colormaps/train")
    
    # parser.add_argument('-lanedir','--lane-dir',help='lane dir', \
    #                     default="/home/ali/Projects/datasets/BDD100K-ori/labels/lane/colormaps/train")

    # ## Save parameters
    # parser.add_argument('-savedir','--save-dir',help='save img dir',default="../tools/re-label/train")
    # parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save pedestrain fake images')
    # parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save pedestrain yolo.txt')
    
    # parser.add_argument('--removelabellist','-remove-labellist',type=list,nargs='+',default="9",help='remove label list')

    parser.add_argument('-imgdir','--img-dir',help='image dir',default="/home/ali/Projects/datasets/bdd100k_data_pedestrain_2023-10-19/images/100k")
    parser.add_argument('-labeldir','--label-dir',help='yolo label dir',default="/home/ali/Projects/datasets/nuimages/nuimage_data/labels/detection/train")
    parser.add_argument('-drivabledir','--drivable-dir',help='drivable label dir', \
                        default="/home/ali/Projects/datasets/nuimages/nuimages-v1.0-all-samples/labels/drivable/train")
    
    parser.add_argument('-lanedir','--lane-dir',help='lane dir', \
                        default="/home/ali/Projects/datasets/nuimages/nuimage_data/labels/lane/masks/train")

    ## Save parameters
    parser.add_argument('-savedir','--save-dir',help='save img dir',default="../tools/bdd100k_data_pedestrain_2023-10-19-MergeLabel")
    parser.add_argument('-saveimg','--save-img',type=bool,default=True,help='save pedestrain fake images')
    parser.add_argument('-savetxt','--save-txt',type=bool,default=True,help='save pedestrain yolo.txt')
    
    parser.add_argument('--removelabellist','-remove-labellist',type=list,nargs='+',default="9",help='remove label list')


    parser.add_argument('-processtype','--process-type',help='proxess type',default="colormap")


    ##==============Func: Get_Wanted_Label_From_DatasetA_And_Put_Into_DatasetB====================================
    parser.add_argument('-datasetAdir','--datasetA-dir',help='image dir',default=\
                        "/home/ali/Projects/GitHub_Code/ali/CopyPaste/tools/bdd100k_data_pedestrain_2023-10-19/infer_result")
    parser.add_argument('-datasetBdir','--datasetB-dir',help='image dir',default=\
                        "/home/ali/Projects/GitHub_Code/ali/CopyPaste/tools/nuImage_data_include_pedestrain/labels/detection/train")

    return parser.parse_args()

if __name__=="__main__":

    label_parameters = get_args_label()
    label = LabelDatasets(label_parameters)
    #label.RemoveLabelsInLabelTXT()
    #label.CorrectNumberOfImagesAndLablesInDatasets()
    #label.SplitAllImagesToSpitFolders()
    #label.Get_Wanted_Label_From_DatasetA_And_Put_Into_DatasetB()
    #label.Get_datasets_img_path_txt()
    label.Get_datasets_img_path_txt()
    #label.Convert_labels()
    #label.Get_Pedestrain_Dataset()
    #label.Get_datasets_img_path_txt()
    #label.process_lane_map()
                        

