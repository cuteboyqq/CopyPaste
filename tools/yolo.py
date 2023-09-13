from ultralytics import YOLO
import glob
# Load a model
#model = YOLO("yolov8.yaml")  # build a new model from scratch
#model = YOLO("yolov8m-cls.yaml")  # load a pretrained model (recommended for training)
#model = YOLO("/home/ali/Projects/GitHub_Code/WNC/wnc-adas/runs/detect/train17/weights/best.pt")
# Use the model
#model.train(data="Argoverse.yaml", epochs=30)  # train the model
#model.train(data="VisDrone.yaml", epochs=10)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model("https://ultralytics.com/images/bus.jpg",save=True)  # predict on an image
#results = model("/home/ali/Projects/GitHub_Code/WNC/datasets/VisDrone/VisDrone2019-DET-val/images/",save=True)  # predict on an image
#results = model("https://www.youtube.com/watch?v=JZm7T1JezNM",save=True)  # predict on an image
#path = model.export(format="onnx",opset=12, imgsz=288)  # export the model to ONNX format
choose = "INSTANCE_SEGMENT"

if choose=="INSTANCE_SEGMENT":
    INSTANCE_SEGMENT=True
    DETECTION=False
    CLASSIFY=False
    TRACKING=False
elif choose=="DETECTION":
    INSTANCE_SEGMENT=False
    DETECTION=True
    CLASSIFY=False
    TRACKING=False
elif choose=="CLASSIFY":
    INSTANCE_SEGMENT=False
    DETECTION=False
    CLASSIFY=True
    TRACKING=False
elif choose=="TRACKING":
    INSTANCE_SEGMENT=False
    DETECTION=False
    CLASSIFY=False
    TRACKING=True


import torch
import cv2
#===============Instance Segmentation====================================================
#INSTANCE_SEGMENT=False
import os
if INSTANCE_SEGMENT:
#https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9
    model = YOLO("yolov8l-seg.pt")  # load a pretrained model (recommended for training)
    img_dir = "./images/one_image"
    save_dir = "./one_image_result"
    os.makedirs(save_dir, exist_ok=True)
    im_path_list = glob.glob(os.path.join(img_dir,"*.jpg"))
    c=1
    for im_path in im_path_list:
        print(im_path)
        #results = model("/home/ali/Projects/datasets/screen_records/20230906_102718.mp4",save=True,save_crop=True,show=False)  # predict on an image
        results = model(im_path,save=True,save_crop=False,show=False)  # predict on an image
        im = cv2.imread(im_path)
        h = im.shape[0]
        w = im.shape[1]
        print("w:{},h:{}".format(w,h))
        result = results[0]
        masks = result.masks
        boxes = result.boxes
        #print(len(boxes))
        #print(len(masks))
        if masks is not None:
            for i in range(len(masks)):
                mask1 = masks[i]
                mask = mask1.data[0].cpu().numpy()
                polygon = mask1.xy[0]
                #from PIL import Image
                #mask_img = Image.fromarray(mask,"I")
                #mask_img = mask_img.resize((1920,1080))
                mask_img = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
                #print("===============================================================")
                #print(i)
                #print(boxes[i].xyxy)
                show_img = False
                if int(boxes[i].cls.cpu().numpy())==0:
                    #mask_img.show()
                    if show_img:
                        cv2.imshow("mask_img",mask_img)
                    xyxy = boxes[i].xyxy.cpu().numpy().squeeze()
                    #print("xyxy")
                    #print(xyxy)
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])
                    mask_roi = mask_img[y1:y2,x1:x2]*255
                    roi      = im[y1:y2,x1:x2]
                    
                    roi_dir = os.path.join(save_dir,"roi")
                    os.makedirs(roi_dir,exist_ok=True)
                    roi_path = roi_dir + "/" + str(c) + ".jpg"
                    if show_img:
                        cv2.imshow("roi",roi)
                    cv2.imwrite(roi_path,roi)

                    mask_dir = os.path.join(save_dir,"mask")
                    os.makedirs(mask_dir,exist_ok=True)
                    mask_path = mask_dir + "/" + str(c) + ".jpg"
                    if show_img:
                        cv2.imshow("mask_roi",mask_roi)
                    cv2.imwrite(mask_path,mask_roi)

                    
                    if show_img:
                        cv2.waitKey(100)
                        cv2.destroyAllWindows()

                    c+=1
                    print("c={}".format(c))
                    print("=================================================")
            #mask_roi = mask_img[]
    # for result in results:
    #     # get array results
    #     masks = result.masks.masks
    #     boxes = result.boxes.boxes
    #     # extract classes
    #     clss = boxes[:, 5]
        

    #     # get indices of results where class is 0 (people in COCO)
    #     people_indices = torch.where(clss == 0)
    #     # use these indices to extract the relevant masks
    #     people_masks = masks[people_indices]
       
    #     # scale for visualizing results
    #     people_mask = torch.any(people_masks, dim=0).int() * 255
    #     print(people_mask)
    #     print("w:{},h:{}".format(w,h))
    #     people_mask = people_masks.cpu().numpy()
    #     print(people_mask.shape)
    #     #print(people_mask[0].shape)
    #     #people_mask[0] = cv2.resize(people_mask[0],(1920,1080),interpolation=cv2.INTER_AREA)
    #     c=1
    #     for box in boxes:
    #         x1 = int(box[0]  * float(640/1920))
    #         y1 = int(box[1]  * float(384/1080))
    #         x2 = int(box[2]  * float(640/1920))
    #         y2 = int(box[3]  * float(384/1080))
    #         print("x1:{},y1:{},x2:{},y2:{}".format(x1,y1,x2,y2))
    #         print("people_mask:{}".format(people_mask.shape))
    #         roi_mask = people_mask[y1:y2,x1:x2]
    #         print("roi_mask:{}".format(roi_mask.shape))
    #         #roi_mask = cv2.imread(roi_mask)
    #         #cv2.imshow("mask",roi_mask)
    #         #cv2.waitKey(1000)
    #         #cv2.destroyAllWindows()
    #         file = str(c) + ".jpg"
    #         path = os.path.join(model.predictor.save_dir,file)
    #         print(path)
    #         #cv2.imwrite(path, roi_mask)
    #         c+=1

       

    #     # save to file
    #     cv2.imwrite(str(model.predictor.save_dir / 'merged_segs.jpg'), people_mask.cpu().numpy())

#===============Detection=========================================================
#DETECTION=True
if DETECTION:
    model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)
    results = model("/home/ali/Projects/datasets/screen_records/20230906_103138.mp4",save=True,save_crop=True,show=False)  # predict on an image


#===============Classify==========================================================
#CLASSIFY=False
if CLASSIFY:
    # Load a model
    model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
    #model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
    #model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='/home/ali/Projects/datasets/landmark', epochs=40, imgsz=64)


#==============Tracking=================================================================
#TRACKING=False
if TRACKING:
    # Load an official or custom model
    model = YOLO('yolov8s.pt')  # Load an official Detect model
    #model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
    #model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
    #model = YOLO('path/to/best.pt')  # Load a custom trained model

    # Perform tracking with the model
    results = model.track(source="https://www.youtube.com/watch?v=JZm7T1JezNM", show=True, tracker="bytetrack.yaml")  # Tracking with default tracker
    #results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker