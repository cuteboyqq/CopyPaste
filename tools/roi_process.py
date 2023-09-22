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

        ## roi directory
        self.roi_dir = args.roi_dir
        self.mask_dir = args.mask_dir

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
            #print("{}:{}".format(c,roi_path))
            save_roi_dir = os.path.join(self.save_dir,"roi")
            os.makedirs(save_roi_dir,exist_ok=True)
            
            save_mask_dir = os.path.join(self.save_dir,"mask")
            os.makedirs(save_mask_dir,exist_ok=True)
            c+=1
            file,file_dir,attribute_dir,label_dir = self.parse_path(roi_path)
            #print(label_dir)
            #print(attribute_dir)
            #label, label_detail = self.parse_nuimage_foldername(label_name)
            attribute_name = os.path.basename(attribute_dir)
            label_name = os.path.basename(label_dir)
            #print(label_name)
            #print(attribute_name)
            mask_path = os.path.join(attribute_dir,"mask",file)

            if os.path.exists(mask_path) and (label_name in self.nuimage_wanted_label or attribute_name in self.nuimage_wanted_label_detail):
                roi = cv2.imread(roi_path)
                h,w=roi.shape[0],roi.shape[1]
                if h*w>=0:
                    print("{}/{}".format(label_name,attribute_name))
                    shutil.copy(mask_path,save_mask_dir)
                    shutil.copy(roi_path,save_roi_dir)
                    print("{}:{} copy successful".format(c,file))

    

        # mask_path_list = glob.glob(os.path.join(self.data_dir,"***","**","mask","*.jpg"))
        # for mask_path in mask_path_list:
        #     print(mask_path)
    def parse_path_2(self,path):
        file=path.split("/")[-1]
        file_name = file.split(".")[0]
        path_dir = os.path.dirname(path)
        path_dir_dir = os.path.dirname(path_dir)
        return file, file_name,path_dir,path_dir_dir
    
    def GetCorrespondingMasks(self):
        roi_path_list = glob.glob(os.path.join(self.roi_dir,"*.jpg"))
        print(roi_path_list)
        c = 1
        for roi_path in roi_path_list:
            file, file_name, path_dir,path_dir_dir = self.parse_path_2(roi_path)
            print(path_dir_dir)
            mask = file
            mask_path = os.path.join(self.mask_dir,mask)
            print(mask_path)
            save_mask_dir = os.path.join(path_dir_dir,"mask")
            if os.path.exists(mask_path):
                shutil.copy(mask_path,save_mask_dir)
                print("{}: {} copy succesful!".format(c,file))
                c+=1

if __name__=="__main__":
    args = get_args_roi()
    roi = ROIProcess(args)
    roi.ParseNuImageROIDataset()
    #roi.GetCorrespondingMasks()

