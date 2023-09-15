import sys
sys.path.append('..')
import glob
import shutil
import os
import cv2
from utils.config import get_args_roi

class ROIProcess:
    def __init__(self,args):
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir

        self.nuimage_wanted_label = ["human.pedestrian.adult",
                            "human.pedestrian.child",
                            "human.pedestrian.construction_worker",
                            "human.pedestrian.personal_mobility",
                            "human.pedestrian.police_officer",
                            "human.pedestrian.stroller",
                            "human.pedestrian.wheelchair",
                            ]
        self.nuimage_wanted_label_detail = ["cycle.with_rider"]

    def parse_path(self,path):
        file = path.split("/")[-1]
        file_dir = os.path.dirname(path)
        file_dir_dir = os.path.dirname(file_dir)
        file_dir_dir_dir = os.path.dirname(file_dir_dir)
        return file,file_dir,file_dir_dir,file_dir_dir_dir

    def parse_nuimage_foldername(foldername):
        label = foldername.split(".")[0]
        label_detail = foldername.split(".")[1]
        return label, label_detail
    
    def ParseNuImageROIDataset(self):
        c=1
        roi_path_list = glob.glob(os.path.join(self.data_dir,"***","**","roi","*.jpg"))
        for roi_path in roi_path_list:
            print("{}:{}".format(c,roi_path))
            save_roi_dir = os.path.join(self.save_dir,"roi")
            os.makedirs(save_roi_dir,exist_ok=True)
            
            save_mask_dir = os.path.join(self.save_dir,"mask")
            os.makedirs(save_mask_dir,exist_ok=True)
            c+=1
            file,file_dir,attribute_name,label_name = self.parse_path(roi_path)
            #label, label_detail = self.parse_nuimage_foldername(label_name)
            mask_path = os.path.join(attribute_name,"mask",file)

            if os.path.exists(mask_path) and (label_name in self.nuimage_wanted_label or attribute_name in self.nuimage_wanted_label_detail):
                print("{}/{}".format(label_name,attribute_name))
                shutil.copy(mask_path,save_mask_dir)
                shutil.copy(roi_path,save_roi_dir)
                print("{}:{} copy successful".format(c,file))


        # mask_path_list = glob.glob(os.path.join(self.data_dir,"***","**","mask","*.jpg"))
        # for mask_path in mask_path_list:
        #     print(mask_path)

if __name__=="__main__":
    args = get_args_roi()
    roi = ROIProcess(args)
    roi.ParseNuImageROIDataset()
