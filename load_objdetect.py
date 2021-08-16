import os
import datetime
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
        
def load_model(MODEL,LABELS):
    print("Loading in new saved_model...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.get_logger().setLevel('ERROR') 
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    if os.path.isabs(MODEL) == False:
        MODELS_DIR = os.path.join(os.getcwd(),'exported-models',MODEL)
    else:
        MODELS_DIR = MODEL
    if os.path.isabs(LABELS) == False:
        FILE = [x for x in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR ,x))]
        if FILE[0] == os.path.basename(LABELS):
            PATH_TO_LABELS = os.path.join(MODELS_DIR,FILE[0])
        else:
            PATH_TO_LABELS = os.path.join(os.getcwd(),"annotations",LABELS)
    else:
        PATH_TO_LABELS = LABELS
    
    PATH_TO_SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
    print("Model: " + MODEL + " has finished loading...")
    return detect_fn, category_index

def load_checkpoint(MODEL,LABELS):
    print("Loading in new checkpoint_model...")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.get_logger().setLevel('ERROR') 
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    if os.path.isabs(MODEL) == False:
        MODELS_DIR = os.path.join(os.getcwd(),'models',MODEL) 
    else:
        MODELS_DIR = MODEL
    if "ckpt" in MODEL:
        PATH_TO_CKPT = MODEL
    else: 
        CKPT = [x for x in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR ,x))]
        if CKPT[0] == "checkpoint":
            PATH_TO_CKPT = os.path.join(MODELS_DIR,CKPT[0])
        else:
            PATH_TO_CKPT = MODELS_DIR
        PATH_TO_CKPT = tf.train.latest_checkpoint(PATH_TO_CKPT)
        
    if os.path.isabs(LABELS) == False:
        FILE = [x for x in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR ,x))]
        print(FILE)
        if FILE[0] == os.path.basename(LABELS):
            PATH_TO_LABELS = os.path.join(MODELS_DIR,FILE[0])
        else:
            PATH_TO_LABELS = os.path.join(os.getcwd(),"annotations",LABELS)
    else:
        PATH_TO_LABELS = LABELS
    
    PATH_TO_CFG = os.path.join(MODELS_DIR, 'pipeline.config')
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    print(PATH_TO_CKPT)
    ckpt.restore(PATH_TO_CKPT).expect_partial()
    
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)
    
    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
    print("Model: " + MODEL + " has finished loading...")
    return detect_fn, category_index
    
def load_images(input):
    print("Loading in new images...")
    IMAGE_PATHS = []
    FILENAMES = []
    if os.path.isfile(input):
        IMAGE_PATHS.append(str(input))
        FILENAMES.append(str(os.path.basename(input)))
    else:
        if input == "test":
            IMG_DIR = os.path.join(os.getcwd(),input)
        else:
            IMG_DIR = input

        with os.scandir(IMG_DIR) as input_files:
            for file in input_files:
                if file.name.endswith(".jpg") or file.name.endswith(".png"):
                    FILENAMES.append(file.name)
                    image_path = os.path.join(IMG_DIR,file.name)
                    IMAGE_PATHS.append(str(image_path))
    print('Done')
    return IMAGE_PATHS, FILENAMES

def load_videos(input):
    print("Loading in new videos...")
    VIDEO_PATHS = []
    FILENAMES = []
    if os.path.isfile(input):
        VIDEO_PATHS.append(str(input))
        FILENAMES.append(str(os.path.basename(input)))
    else:
        if input == "test":
            VID_DIR = os.path.join(os.getcwd(),input)
        else:
            VID_DIR = input

        with os.scandir(VID_DIR) as input_files:
            for file in input_files:
                if file.name.endswith(".mp4"):
                    FILENAMES.append(file.name)
                    image_path = os.path.join(VID_DIR,file.name)
                    VIDEO_PATHS.append(str(image_path))
    print('Done')
    return VIDEO_PATHS, FILENAMES

def output_dir(output,model):
    now = str(datetime.datetime.now())[:19]
    now = now.replace(":","-")
    now = now.replace(" ","_")
    if output == "detect":
        OUTPUT_DIR = os.path.join(os.getcwd(),output)
    else:
        OUTPUT_DIR = output
    model = os.path.basename(model) + "_"
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, model + now)
    os.mkdir(OUTPUT_DIR)
    return OUTPUT_DIR

def process_image_model(image_np):
    image_tensor = tf.convert_to_tensor(image_np)
    image_tensor = image_tensor[tf.newaxis, ...]
    return image_tensor

def process_image_checkpoint(image_np):
    image_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    return image_tensor

def vis_detect(image_np,detect,cat_index,max_box,min_thresh):
    return viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np,
          detect['detection_boxes'][0].numpy(),
          (detect['detection_classes'][0].numpy()).astype(int),
          detect['detection_scores'][0].numpy(),
          cat_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=max_box,
          min_score_thresh=min_thresh,
          agnostic_mode=False)

def save_detect(image_np,output_dir,filename):
    plt.figure()
    plt.imshow(image_np)
    detect_path = os.path.join(output_dir,'detect_' + filename)
    plt.savefig(detect_path)
    print('Done')