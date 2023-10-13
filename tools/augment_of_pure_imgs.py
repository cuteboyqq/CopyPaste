# -*- coding: utf-8 -*-
"""
Created on Sun May  8 12:59:13 2022

@author: User
"""
import os
import shutil
import cv2
import glob
import tqdm
import numpy as np
import random

def Analysis_Img_Path(img_path):
    img_dir = os.path.dirname(img_path)
    img_dir_name = os.path.basename(img_dir)
    img = img_path.split(os.sep)[-1]
    img_name = img.split(".jpg")[0]
    
    print(img_name)
    return img_dir_name,img,img_name

def Analysis_Img_Path_2(img_path):
    img_dir = os.path.dirname(img_path)
    img_dir_name = os.path.basename(img_dir)
    img = img_path.split(os.sep)[-1]
    img_name = img.split(".jpg")[0]
    
    print(img_name)
    return img_dir_name,img,img_name,img_dir




def augment_hsv(img_path, save_img_dir, hgain=0.5, sgain=0.5, vgain=0.5, do_he=False, num=10):
    print("aug_hsv not implemented")
    r = np.random.uniform(-1,1,3) * [hgain, sgain, vgain] + 1 #random gains
    
    img_cv2 = cv2.imread(img_path)
    img = np.array(img_cv2)
    
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype # uint8
    
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    #lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    hsv_images = []
    hsv_he_images = []
    '''====================================================================='''
    ori_lable_name, img_file, img_name = Analysis_Img_Path(img_path)
    new_folder_name = ori_lable_name
    save_img_dir = save_img_dir + '_' +  'hsv' 
    
    save_dir = os.path.join(save_img_dir,new_folder_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    '''======================================================================'''
    num = int(num)
    for i in range(num*2):
        ra = (i) *  (float)(1/num)
        lut_v = np.clip(x * ra, 0, 255).astype(dtype)
    
        img_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v))).astype(dtype)
    #img_hsv = cv2.merge((h, s, cv2.LUT(v, lut_v))).astype(dtype)
        result_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        hsv_images.append(result_img)
        result_he_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR) #no return needed
        
        
        '''===================================save image====================================='''
        
        save_img_file = img_name + '_' + str(i) + '.jpg'
        save_img_file_path = os.path.join(save_dir,save_img_file)
        
        img_dir_name,img,img_name,img_dir = Analysis_Img_Path_2(img_path)
        if not img_dir_name=="mask":
            if (i) > (num/5) and i!=(num): 
                cv2.imwrite(save_img_file_path,result_img)
        else: # Save original Mask 
            if (i) > (num/5) and i!=(num): 
                cv2.imwrite(save_img_file_path,img_cv2)
        '''===================================save image====================================='''
        
    #Histogram equalization
        if do_he==True:
            if random.random() < 0.2:
                for i in range(3):
                    result_he_img[:,:,i] = cv2.equalizeHist(result_img[:,:,i])
            
            else:
                result_he_img = result_img
            
            hsv_he_images.append(result_he_img)
    
    return hsv_images,hsv_he_images


def aug_blur(img_path,save_img_dir,blur_type,blur_size):
    print("aug_blur not implemented")
    
    im = cv2.imread(img_path)
    img_mean  = cv2.blur(im,(blur_size,blur_size))
    img_Gaussian = cv2.GaussianBlur(im,(blur_size,blur_size),0)
    img_median = cv2.medianBlur(im,blur_size)
    img_bilater = cv2.bilateralFilter(im,9,75,75)
    
    titles = ['srcImg', 'mean', 'Gaussian', 'median', 'bilateral']
    imgs =   [im, img_mean, img_Gaussian, img_median , img_bilater]
    
    ori_lable_name, img, img_name = Analysis_Img_Path(img_path)
  
    new_folder_name = ori_lable_name
    save_img_dir = save_img_dir + '_' +  titles[blur_type] + '_' + str(blur_size)
    
    save_dir = os.path.join(save_img_dir,new_folder_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_img_file = img_name + '_' + titles[blur_type] + '_' + str(blur_size) + '.jpg'
    save_img_file_path = os.path.join(save_dir,save_img_file)
    
    cv2.imwrite(save_img_file_path,imgs[blur_type])
    
    return imgs,titles
    
def aug_flip(img_path,save_img_dir,flip_type):
    print(" auf_flip not implemented")
    img = cv2.imread(img_path) #BGR
    img = np.array(img)
    
    img_flip_lrud = cv2.flip(img,-1)
    img_flip_lr = cv2.flip(img,1)
    img_flip_ud = cv2.flip(img,0)
    
    titles = ['flip_lrud', 'flip_lr', 'flip_ud']
    imgs =   [img_flip_lrud, img_flip_lr, img_flip_ud]
    
    ori_lable_name, img, img_name = Analysis_Img_Path(img_path)
    
    new_folder_name = ori_lable_name
    
    save_img_dir = save_img_dir + '_' +  titles[flip_type] 
    
    save_dir = os.path.join(save_img_dir,new_folder_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_img_file = img_name + '_' + titles[flip_type] + '.jpg'
    save_img_file_path = os.path.join(save_dir,save_img_file)
    
    
    cv2.imwrite(save_img_file_path,imgs[flip_type])

def augment_resize(img_path, save_img_dir, resize_width_ratio, resize_height_ratio):
    img = cv2.imread(img_path) #BGR
    #img = np.array(img)
    img_h, img_w = img.shape[0], img.shape[1]
    
    ori_label_name, img_file, img_name, img_dir = Analysis_Img_Path_2(img_path)
    print("img_dir:{}".format(img_dir))
    print("ori_lable_name:{}".format(ori_label_name))
    print("img_file:{}".format(img_file))
    print("img_name:{}".format(img_name))
    

    img_dir_dir = os.path.dirname(img_dir)
    print("img_dir_dir:{}".format(img_dir_dir))
    
    mask_path = os.path.join(img_dir_dir,"mask",img_file)
    print("mask_path:{}".format(mask_path))

    mask = cv2.imread(mask_path)
    save_resize_img = False
    #cv2.imshow("mask",mask)
    #cv2.waitKey(0)
    h_min = int(resize_height_ratio*100) - 20 if int(resize_height_ratio*100) - 20 > 0 else 0
    w_min = int(resize_width_ratio*100) - 20 if int(resize_height_ratio*100) - 20 > 0 else 0
    random_ratio_h = random.randint(h_min,int(resize_height_ratio*100)) / 100.0
    random_ratio_w = random.randint(w_min,int(resize_width_ratio*100)) / 100.0

    SIZE_W_TH = 100
    SIZE_H_TH = 100
    if img_h>SIZE_H_TH and img_w>SIZE_W_TH and ori_label_name=="roi": #filter too small images
        
        random_num = random.randint(1,4)
        if random_num==1:
            img_resize = cv2.resize(img,(int(img_w*(1-random_ratio_w)),int(img_h*(1-(random_ratio_h/2.0)))),interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask,(int(img_w*(1-random_ratio_w)),int(img_h*(1-(random_ratio_h/2.0)))),interpolation=cv2.INTER_NEAREST)
        elif random_num==2:
            img_resize = cv2.resize(img,(int(img_w*(1-random_ratio_w)),int(img_h*(1+random_ratio_h))),interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask,(int(img_w*(1-random_ratio_w)),int(img_h*(1+random_ratio_h))),interpolation=cv2.INTER_NEAREST)
        elif random_num==3:
            img_resize = cv2.resize(img,(int(img_w*(1+random_ratio_w)),int(img_h*(1-(random_ratio_h/2.0)))),interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask,(int(img_w*(1+random_ratio_w)),int(img_h*(1-(random_ratio_h/2.0)))),interpolation=cv2.INTER_NEAREST)
        elif random_num==4:
            img_resize = cv2.resize(img,(int(img_w*(1+random_ratio_w)),int(img_h*(1+random_ratio_h))),interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask,(int(img_w*(1+random_ratio_w)),int(img_h*(1+random_ratio_h))),interpolation=cv2.INTER_NEAREST)

        save_resize_img = True
        

    elif img_h<SIZE_H_TH and img_w>SIZE_W_TH and ori_label_name=="roi":

        random_num = random.randint(1,2)
        if random_num==1:
            img_resize = cv2.resize(img,(int(img_w*(1-random_ratio_w)),int(img_h*(1+(random_ratio_h/1.0)))),interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask,(int(img_w*(1-random_ratio_w)),int(img_h*(1+(random_ratio_h/1.0)))),interpolation=cv2.INTER_NEAREST)
        elif random_num==2:
            img_resize = cv2.resize(img,(int(img_w*(1+random_ratio_w)),int(img_h*(1+random_ratio_h))),interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask,(int(img_w*(1+random_ratio_w)),int(img_h*(1+random_ratio_h))),interpolation=cv2.INTER_NEAREST)
     
        save_resize_img=True

    elif img_h>SIZE_H_TH and img_w<SIZE_W_TH and ori_label_name=="roi":
        # ---------
        # |       |
        # |       |
        # |       |
        # |       |
        # |       |
        # |       |
        # --------- 
        random_num = random.randint(1,2)
        if random_num==1:
            img_resize = cv2.resize(img,(int(img_w*(1+random_ratio_w)),int(img_h*(1+(random_ratio_h/1.0)))),interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask,(int(img_w*(1+random_ratio_w)),int(img_h*(1+(random_ratio_h/1.0)))),interpolation=cv2.INTER_NEAREST)
        elif random_num==2:
            img_resize = cv2.resize(img,(int(img_w*(1+random_ratio_w)),int(img_h*(1-random_ratio_h/2.0))),interpolation=cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask,(int(img_w*(1+random_ratio_w)),int(img_h*(1-random_ratio_h/2.0))),interpolation=cv2.INTER_NEAREST)
     
        save_resize_img = True

    elif img_h<SIZE_H_TH and img_w<SIZE_W_TH and ori_label_name=="roi":
        img_resize = cv2.resize(img,(int(img_w*(1+random_ratio_w)),int(img_h*(1+(random_ratio_h/1.0)))),interpolation=cv2.INTER_NEAREST)
        mask_resize = cv2.resize(mask,(int(img_w*(1+random_ratio_w)),int(img_h*(1+(random_ratio_h/1.0)))),interpolation=cv2.INTER_NEAREST)
        save_resize_img = True
    else:
        print("img w or h is smaller than 50 pixels or not roi image, so do not resize ~~~")
        save_resize_img = False
    
    if save_resize_img:
        new_folder_name = ori_label_name
        save_img_dir_ori = save_img_dir + '_resize'

        save_img_dir = os.path.join(save_img_dir_ori,new_folder_name)
        os.makedirs(save_img_dir,exist_ok=True)

        save_img_file = img_name + '_resize.jpg' 
        save_img_file_path = os.path.join(save_img_dir,save_img_file)
        
        cv2.imwrite(save_img_file_path,img_resize)

        new_folder_mask_name = "mask"
        save_mask_dir = os.path.join(save_img_dir_ori,new_folder_mask_name)
        #print("save_mask_dir:{}".format(save_mask_dir))
        os.makedirs(save_mask_dir,exist_ok=True)

        save_mask_file = img_name + '_resize.jpg'
        save_mask_file_path = os.path.join(save_mask_dir,save_mask_file)
        #print("save_mask_file_path:{}".format(save_mask_file_path))
        #input()
        cv2.imwrite(save_mask_file_path,mask_resize)

def pure_img_augmentation(do_blur,blur_type,blur_size,
                          do_flip,flip_type,
                          do_hsv,
                          numv,
                          do_resize,
                          resize_width_ratio,
                          resize_height_ratio,
                          img_dir,
                          save_img_dir):
    #print("not implemented")
    hsv_images,hsv_he_images = [],[]
    img_path_list = glob.iglob(os.path.join(img_dir,'**/*.jpg'))
    for img_path in img_path_list:
        print(img_path)
        if do_blur:
            imgs, titles = aug_blur(img_path,save_img_dir,blur_type,blur_size)
            
            #print("not implemented")
            
        if do_flip:
            aug_flip(img_path,save_img_dir,flip_type)
            #print("not implemented")
        
        if do_hsv:
            hsv_images,hsv_he_images = augment_hsv(img_path, save_img_dir, hgain=0.0, sgain=0.0, vgain=0.9, do_he=False, num=numv)
        
        if do_resize:
            augment_resize(img_path, save_img_dir, resize_width_ratio, resize_height_ratio)

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    '''============================input img/output img parameters setting================================='''
    parser.add_argument('-imgdir','--img-dir',help='image dir',default='../datasets/ROI/lanemarking/train')
    parser.add_argument('-savedir','--save-dir',help='save aug-img dir',default='../datasets/ROI/lanemarking/roi_aug')
    
    '''===================blur parameter settings=========================================================='''
    parser.add_argument('-blur','--blur',help='enable blur augment',action='store_true')
    parser.add_argument('-blurtype','--blur-type',help='blur type : 0:original; 1:mean; 2:Gaussian; 3:median; 4:bilateral',default=2)
    parser.add_argument('-blursize','--blur-size',help='blur size',default=7)
    '''===================flip parameter settings=========================================================='''
    parser.add_argument('-flip','--flip',help='enable flip augment',action='store_true')
    parser.add_argument('-fliptype','--flip-type',help='flip type: 0:lrud, 1:lr, 2:ud' ,default=1)
    '''===================hsv parameter settings=========================================================='''
    parser.add_argument('-hsv','--hsv',help='enable hsv augment',action='store_true')
    parser.add_argument('-numv','--numv',help='num of v' ,default=3)
    '''===================resize parameter settings========================================================='''
    parser.add_argument('-resize','--resize',help='enable resize augment',action='store_true')
    parser.add_argument('-resize_w','--resize_w',help='random shorter/larger num of pixel in width ratio' ,default=0.40)
    parser.add_argument('-resize_h','--resize_h',help='random shorter/larger num of pixel in height ratio' ,default=0.90)
    
    return parser.parse_args()

if __name__=="__main__":
    
    
    args = get_args()
    print("===========IO settings================")
    img_dir = args.img_dir
    save_img_dir = args.save_dir
    print('img_dir=',img_dir)
    print('save_img_dir=',save_img_dir)
    print("=====blur parameter settings=====")
    do_blur = args.blur
    blur_type = args.blur_type
    blur_size = args.blur_size
    print('do_blur =',do_blur)
    print('blur_type=',blur_type)
    print('blur_size=',blur_size)
    print("=====flip parameter settings=====")
    do_flip = args.flip
    flip_type = args.flip_type
    print('flip_type=',flip_type)
    print("=====hsv parameter settings=====")
    do_hsv = args.hsv
    numv = args.numv
    print("resize parameter settings=======")
    do_resize = True # args.resize
    resize_width_ratio  = args.resize_w
    resize_height_ratio = args.resize_h

    #do_blur = True
    #do_flip = True
    #blur_type = 2
    #blur_size = 7
    #flip_type = 1
    #img_dir = "C:/TLR/datasets/roi-original"
    #save_img_dir = "C:/TLR/datasets"
    pure_img_augmentation(False,#do blur
                          blur_type,
                          blur_size,
                          True, # do flip
                          flip_type,
                          False, # do hsv
                          numv,
                          True, # do_resize,
                          resize_width_ratio,
                          resize_height_ratio,
                              img_dir,
                              save_img_dir
                             
                              )
    
        