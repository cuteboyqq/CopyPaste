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
        return file
        

    def RemoveLabelsInLabelTXT(self):
        print(self.label_path_list)
        for label_path in self.label_path_list:
            #print(label_path)
            label_file = self.parse_path(label_path)
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

if __name__=="__main__":

    label_parameters = get_args_label()
    label = LabelDatasets(label_parameters)
    label.RemoveLabelsInLabelTXT()
                        

