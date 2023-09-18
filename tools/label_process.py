import sys
sys.path.append('..')
from utils.config import get_args_label
import glob
import os
import shutil
import cv2

## how the argsparse list
#https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse

class LabelDatasets:
    def __init__(self,args):

        ## BDD100K Dataset directory
        self.img_dir = args.img_dir
        self.label_dir = args.label_dir
        self.drivable_dir = args.drivable_dir
        self.lane_dir = args.lane_dir

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

    
    def CorrectNumberOfImagesAndLablesInDatasets(self):
        img_path_list = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        #label_path_list = os.path.join(self.label_dir,"*.txt")
        #drivable_path_list = os.path.join(self.drivable_dir,"*.png")

        ## Create save folder
        save_img_dir = os.path.join(self.save_dir,"images","train")
        os.makedirs(save_img_dir,exist_ok=True)

        save_label_dir = os.path.join(self.save_dir,"labels","detection","train")
        os.makedirs(save_label_dir,exist_ok=True)

        save_lane_mask_dir = os.path.join(self.save_dir,"labels","lane","train","masks")
        os.makedirs(save_lane_mask_dir,exist_ok=True)

        save_lane_colormap_dir = os.path.join(self.save_dir,"labels","lane","train","colormaps")
        os.makedirs(save_lane_colormap_dir,exist_ok=True)

        save_drivable_mask_dir = os.path.join(self.save_dir,"labels","drivable","train","masks")
        os.makedirs(save_drivable_mask_dir,exist_ok=True)

        save_drivable_colormap_dir = os.path.join(self.save_dir,"labels","drivable","train","colormaps")
        os.makedirs(save_drivable_colormap_dir,exist_ok=True)
        c = 1
        for img_path in  img_path_list:
            file,file_name = self.parse_path(img_path)
            print("{}:{}".format(c,file))
            c+=1
            label = file_name + ".txt"
            label_path = os.path.join(self.label_dir,label)
            if not os.path.exists(label_path):
                print("label_path not exists :{}".format(label_path))
                continue
            drivable_label = file_name + ".png"
            drivable_colormap_path = os.path.join(self.drivable_dir,"colormaps",drivable_label)

            drivable_mask_path = os.path.join(self.drivable_dir,"masks",drivable_label)

            if not os.path.exists(drivable_colormap_path):
                print("drivable_colormap_path not exists")
                continue

            lane_label_jpg = file_name + ".jpg"
            lane_colormap_path = os.path.join(self.lane_dir,"colormaps",lane_label_jpg)
            lane_label_png = file_name + ".png"
            lane_mask_path = os.path.join(self.lane_dir,"masks",lane_label_png)

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


if __name__=="__main__":

    label_parameters = get_args_label()
    label = LabelDatasets(label_parameters)
    #label.RemoveLabelsInLabelTXT()
    label.CorrectNumberOfImagesAndLablesInDatasets()
                        

