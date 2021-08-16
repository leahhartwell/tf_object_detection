import argparse, cv2, os
import numpy as np
from PIL import Image
from urllib.request import urlopen
import load_objdetect as lod

# commandline argument parser
parser = argparse.ArgumentParser(description = "load_model.py Usage")
parser.add_argument("-m", "--model", help = "Name or Path to Trained Input Model, Format: sting", required = True, type = str)
parser.add_argument("-l", "--label_map", help = "Name or Path to Label Map, Format: string, Default: label_map.pbtxt", required = False, default = "label_map.pbtxt", type = str)
parser.add_argument("-i", "--input", help = "Path to Input, Format: sting, Default: test", required = False, default = "test", type = str)
parser.add_argument("-o", "--output", help = "Path to Output Folder, Format: sting, Default: detect", required = False, default = "detect", type = str)
parser.add_argument("-b", "--boxes", help = "Max Number of Boxes to Detect, Format: int, Default: 20", required = False, default = 20, type = int)
parser.add_argument("-t", "--threshold", help = "Detection Threshold, Format: float, Default: 0.5", required = False, default = 0.5, type = float)
parser.add_argument("-it", "--input_type", help = "Input File Format, Format: i/c/v/s", required = True, type = str)

args = parser.parse_args()

detect_fn, category_index = lod.load_model(args.model,args.label_map)
output_dir = lod.output_dir(args.output,args.model)

if args.input_type == "i":
    IMAGE_PATHS, FILENAMES = lod.load_images(args.input)
    for image in range(len(IMAGE_PATHS)):
        print('Running inference for {}... '.format(IMAGE_PATHS[image]), end='')
        image_np = np.array(Image.open(IMAGE_PATHS[image]))
        image_tensor = lod.process_image_model(image_np)
        detections = detect_fn(image_tensor)
        lod.vis_detect(image_np,detections,category_index,args.boxes,args.threshold)
        lod.save_detect(image_np,output_dir,FILENAMES[image])
elif args.input_type == "c":
    cap = cv2.VideoCapture(0)
    output_file = cv2.VideoWriter(os.path.join(output_dir,os.path.basename(output_dir)+".mp4"), -1, 25.0, (640,480))
    while True:
        ret, image_np = cap.read()
        image_tensor = lod.process_image_model(image_np)
        detections = detect_fn(image_tensor)
        lod.vis_detect(image_np,detections,category_index,args.boxes,args.threshold)
        cv2.imshow('object detection', cv2.resize(image_np,(640,480)))
        output_file.write(image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    output_file.release()
    cv2.destroyAllWindows()
elif args.input_type == "v":
    VIDEO_PATHS, FILENAMES = lod.load_videos(args.input)
    for video in range(len(VIDEO_PATHS)):
        cap = cv2.VideoCapture(VIDEO_PATHS[video])
        output_file = cv2.VideoWriter(os.path.join(output_dir,"detect_"+FILENAMES[video]), -1, 25.0, (640,480))
        while True:
            ret, image_np = cap.read()
            if ret:
                image_tensor = lod.process_image_model(image_np)
                detections = detect_fn(image_tensor)
                lod.vis_detect(image_np,detections,category_index,args.boxes,args.threshold)
                cv2.imshow('object detection', cv2.resize(image_np,(640,480)))
                output_file.write(image_np)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        output_file.release()
        cv2.destroyAllWindows()
elif args.input_type == "s":
    IP = args.input
    stream = urlopen("http://" + IP + ":5000/video_feed")
    bytes = b''
    img_arr = []
    detect_img_arr = []
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            image_np = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            detect_img_np = image_np.copy()
            img_arr.append(image_np)
            image_tensor = lod.process_image_model(detect_img_np)
            detections = detect_fn(image_tensor)
            lod.vis_detect(detect_img_np,detections,category_index,args.boxes,args.threshold)
            cv2.imshow('object detection', cv2.resize(detect_img_np,(640,480)))
            height, width, layers = image_np.shape
            size = (width,height)
            detect_img_arr.append(detect_img_np)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break              
    orig_file = cv2.VideoWriter(os.path.join(output_dir,"orig_" + IP + ".mp4"), -1, 15, size)
    detect_file = cv2.VideoWriter(os.path.join(output_dir,"detect_" + IP + ".mp4"), -1, 15, size)
    print("Starting Video Processing...")
    for i in range(len(img_arr)):
        orig_file.write(img_arr[i])
    print("Finished Processing Original Video")
    for i in range(len(detect_img_arr)):
        detect_file.write(detect_img_arr[i])
    print("Finished Processing Detection Video")
    orig_file.release()
    detect_file.release()
    cv2.destroyAllWindows()      
    