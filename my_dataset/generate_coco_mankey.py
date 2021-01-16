import json
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import transformations as transformations
import shutil
import yaml
import argparse
import time
from pycocotools import mask
from skimage import measure


# Generate COCO

## Basic info needed in COOC.json format
cur_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

info = {'description': 'This is developing 0.1 version of the door knob dataset.',
         'url': '',
         'version': '0.1',
         'year': 2019,
         'contributor': 'Jerry Wang',
         'date_created': cur_time}
licenses = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
          'id': 1,
          'name': 'Attribution-NonCommercial-ShareAlike License'},
         {'url': 'http://creativecommons.org/licenses/by-nc/2.0/',
          'id': 2,
          'name': 'Attribution-NonCommercial License'},
         {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/',
          'id': 3,
          'name': 'Attribution-NonCommercial-NoDerivs License'},
         {'url': 'http://creativecommons.org/licenses/by/2.0/',
          'id': 4,
          'name': 'Attribution License'},
         {'url': 'http://creativecommons.org/licenses/by-sa/2.0/',
          'id': 5,
          'name': 'Attribution-ShareAlike License'},
         {'url': 'http://creativecommons.org/licenses/by-nd/2.0/',
          'id': 6,
          'name': 'Attribution-NoDerivs License'},
         {'url': 'http://flickr.com/commons/usage/',
          'id': 7,
          'name': 'No known copyright restrictions'},
         {'url': 'http://www.usa.gov/copyright.shtml',
          'id': 8,
          'name': 'United States Government Work'}]
categories_dict = [
    {'supercategory': 'person', 'id': 1, 'name': 'person'},
    {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
    {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
    {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'},
    {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'},
    {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'},
    {'supercategory': 'vehicle', 'id': 7, 'name': 'train'},
    {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'},
    {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'},
    {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'},
    {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'},
    {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'},
    {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'},
    {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'},
    {'supercategory': 'animal', 'id': 16, 'name': 'bird'},
    {'supercategory': 'animal', 'id': 17, 'name': 'cat'},
    {'supercategory': 'animal', 'id': 18, 'name': 'dog'},
    {'supercategory': 'animal', 'id': 19, 'name': 'horse'},
    {'supercategory': 'animal', 'id': 20, 'name': 'sheep'},
    {'supercategory': 'animal', 'id': 21, 'name': 'cow'},
    {'supercategory': 'animal', 'id': 22, 'name': 'elephant'},
    {'supercategory': 'animal', 'id': 23, 'name': 'bear'},
    {'supercategory': 'animal', 'id': 24, 'name': 'zebra'},
    {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'},
    {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'},
    {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'},
    {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'},
    {'supercategory': 'accessory', 'id': 32, 'name': 'tie'},
    {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'},
    {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'},
    {'supercategory': 'sports', 'id': 35, 'name': 'skis'},
    {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'},
    {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'},
    {'supercategory': 'sports', 'id': 38, 'name': 'kite'},
    {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'},
    {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'},
    {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'},
    {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'},
    {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'},
    {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'},
    {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'},
    {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'},
    {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'},
    {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'},
    {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'},
    {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'},
    {'supercategory': 'food', 'id': 52, 'name': 'banana'},
    {'supercategory': 'food', 'id': 53, 'name': 'apple'},
    {'supercategory': 'food', 'id': 54, 'name': 'sandwich'},
    {'supercategory': 'food', 'id': 55, 'name': 'orange'},
    {'supercategory': 'food', 'id': 56, 'name': 'broccoli'},
    {'supercategory': 'food', 'id': 57, 'name': 'carrot'},
    {'supercategory': 'food', 'id': 58, 'name': 'hot dog'},
    {'supercategory': 'food', 'id': 59, 'name': 'pizza'},
    {'supercategory': 'food', 'id': 60, 'name': 'donut'},
    {'supercategory': 'food', 'id': 61, 'name': 'cake'},
    {'supercategory': 'furniture', 'id': 62, 'name': 'chair'},
    {'supercategory': 'furniture', 'id': 63, 'name': 'couch'},
    {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'},
    {'supercategory': 'furniture', 'id': 65, 'name': 'bed'},
    {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'},
    {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'},
    {'supercategory': 'electronic', 'id': 72, 'name': 'tv'},
    {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'},
    {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'},
    {'supercategory': 'electronic', 'id': 75, 'name': 'remote'},
    {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'},
    {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'},
    {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'},
    {'supercategory': 'appliance', 'id': 79, 'name': 'oven'},
    {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'},
    {'supercategory': 'appliance', 'id': 81, 'name': 'sink'},
    {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'},
    {'supercategory': 'indoor', 'id': 84, 'name': 'book'},
    {'supercategory': 'indoor', 'id': 85, 'name': 'clock'},
    {'supercategory': 'indoor', 'id': 86, 'name': 'vase'},
    {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'},
    {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'},
    {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
    {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'},

    {'supercategory': 'indoor', 'id': 91, 'name': 'door'},
    {'supercategory': 'door', 'id': 92, 'name': 'lever'},
    {'supercategory': 'door', 'id': 93, 'name': 'round'},
    {'supercategory': 'door', 'id': 94, 'name': 'pull'},
]

def first_nonzero_idx(
        binary_array,
        reversed):  # type: (np.ndarray, bool) -> int
    """
    Get the index of the first element in an array that is not zero
    reversed means whether the binary_array should be reversed
    :param binary_array: A 1-D numpy array
    :param reversed:
    :return: The index to the first non-zero element
    """
    start = 0
    end = binary_array.size
    step = 1
    if reversed:
        start = binary_array.size - 1
        end = -1
        step = -1

    # The iteration
    for i in range(start, end, step):
        if binary_array[i] > 0:
            return i

    # Everything is zero
    return None

def mask2bbox(mask_img):  # type: (np.ndarray) -> (PixelCoord, PixelCoord)
    """
    Given an object binary mask, get the tight object bounding box
    as a tuple contains top_left and bottom_right pixel coord
    :param mask_img: (height, width, 3) mask image
    :return: A tuple contains top_left and bottom_right pixel coord
    """
    binary_mask = mask_img.max(axis=2)
    n_rows, n_cols = binary_mask.shape
    # Compute sum over the row and compute the left and right
    mask_rowsum = np.sum(binary_mask, axis=0, keepdims=False)
    assert mask_rowsum.size == n_cols
    left = first_nonzero_idx(mask_rowsum, False)
    right = first_nonzero_idx(mask_rowsum, True)

    # Compute sum over the col and compute the top and bottom
    mask_colsum = np.sum(binary_mask, axis=1)
    assert mask_colsum.size == n_rows
    top = first_nonzero_idx(mask_colsum, False)
    bottom = first_nonzero_idx(mask_colsum, True)

    # Ok
#     top_left = PixelCoord()
#     top_left.x = left
#     top_left.y = top
#     bottom_right = PixelCoord()
#     bottom_right.x = right
#     bottom_right.y = bottom
#     return top_left, bottom_right
    return [left, top, right,bottom]


def mask_to_yaml(log_path):
    mask_frames_path = os.path.join(log_path, "processed", "image_masks")
    assert os.path.exists(mask_frames_path)
    file_name_lists = os.listdir(mask_frames_path)

    if len(file_name_lists) % 2 == 0:
        num_images = int((len(file_name_lists)) / 2)
        yaml_path = os.path.join(log_path, "processed", "door_lever_3_keypoint.yaml")
    if os.path.exists(yaml_path):
        os.remove(yaml_path)

    for i in range(num_images):
        mask_image_filename = "%06i_mask.png" % i
        image_path = os.path.join(mask_frames_path, mask_image_filename)
        assert os.path.exists(image_path)
        im = cv2.imread(image_path)
        left_top_right_bottom = mask2bbox(im)

        yaml_dict = {i: {"bbox_bottom_right_xy": [left_top_right_bottom[2], left_top_right_bottom[3]],
                         "bbox_top_left_xy": [left_top_right_bottom[0], left_top_right_bottom[1]],
                         "rgb_image_filename": "%06i_rgb.png" % i}
                     }

        with open(yaml_path, "a") as f:
            yaml.dump(yaml_dict, f)

        ## Check
        vis = "%06i_visible_mask.png" % i
        vis_a = os.path.join(mask_frames_path, vis)
        save_path = os.path.join(log_path, "bbox_test")
        assert os.path.exists(vis_a)
        img = cv2.imread(vis_a)
        xmin = left_top_right_bottom[0]
        xmax = left_top_right_bottom[2]
        ymin = left_top_right_bottom[1]
        ymax = left_top_right_bottom[3]

        draw = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        save = "%06i_bbox_mask.png" % i
        save_file = os.path.join(save_path, save)
        cv2.imwrite(save_file, draw)







class make_my_coco(object):
    def __init__(self, coco_path):

        self.coco_path = coco_path
        self.coco_train_path = os.path.join(self.coco_path, 'images', 'train2014')
        self.coco_val_path = os.path.join(self.coco_path, 'images', 'val2014')
        self.coco_annot_path = os.path.join(self.coco_path, 'annotations')

        self.train_json_path = os.path.join(self.coco_annot_path, 'instances_train2014.json')
        self.val_json_path = os.path.join(self.coco_annot_path, 'instances_val2014.json')

        self.initial()

        self.train_dict = self.read_json(mode = 'train')
        self.val_dict = self.read_json(mode = 'val')

    def initial(self):
        if not os.path.exists(self.coco_path):
            os.makedirs(self.coco_path)
        if not os.path.exists(self.coco_train_path):
            os.makedirs(self.coco_train_path)
        if not os.path.exists(self.coco_val_path):
            os.makedirs(self.coco_val_path)
        if not os.path.exists(self.coco_annot_path):
            os.makedirs(self.coco_annot_path)

        if not os.path.exists(self.train_json_path):
            self.build_json(mode = 'train')
        if not os.path.exists(self.val_json_path):
            self.build_json(mode = 'val')

    def build_json(self, mode):
        if mode =='train':
            coco_json_path = self.train_json_path
        elif mode =='val':
            coco_json_path = self.val_json_path
        else:
            raise NotImplementedError

        coco_dict = {
            "info": '',
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories" :categories_dict
        }
        with open(coco_json_path ,"w") as f:
            json.dump(coco_dict ,f)

    def read_json(self, mode):

        if mode =='train':
            coco_json_path = self.train_json_path
        elif mode =='val':
            coco_json_path = self.val_json_path
        else:
            raise NotImplementedError

        with open(coco_json_path ,'r') as load_f:
            coco_dict = json.load(load_f)

        return coco_dict

    def get_dataset_number(self, mode):
        if mode =='train':
            coco_number = len(self.train_dict['images'])
        elif mode =='val':
            coco_number = len(self.val_dict['images'])
        else:
            raise NotImplementedError

        return coco_number

    def add_data_to_coco(self, mode, data_path, category_number):
        if mode =='train':
            coco_dict =  self.train_dict
            coco_images_path =  self.coco_train_path
            coco_json_path = self.train_json_path
        elif mode =='val':
            coco_dict =  self.val_dict
            coco_images_path = self.coco_val_path
            coco_json_path = self.val_json_path
        else:
            raise NotImplementedError

        images_path = os.path.join(data_path ,'processed', 'images')
        masks_path = os.path.join(data_path ,'processed', 'image_masks')
        # TODO
        yaml_path = os.path.join(data_path ,'processed', 'door_lever_3_keypoint.yaml')

        with open(yaml_path, 'r') as f:
            dataset_yaml_map = yaml.load(f.read())


        id_index = self.get_dataset_number(mode)

        train_mode = 'Door_ ' +mode

        for key in dataset_yaml_map.keys():
            # TODO
            origin_file_path = os.path.join(images_path, dataset_yaml_map[key]['rgb_image_filename'])
            target_file_name = train_mode + '_%06d.png ' %id_index
            target_file_path = os.path.join(coco_images_path, target_file_name)

            shutil.copyfile(origin_file_path ,target_file_path )

            img_dict ={'license': 3,
                        'file_name': target_file_name,
                        'coco_url': '',
                        'height': 480,
                        'width': 640,
                        'date_captured': '2013-11-14 11:18:45',
                        'flickr_url': '',
                        'id': id_index}

            x, y = dataset_yaml_map[key]['bbox_top_left_xy']
            x2, y2 = dataset_yaml_map[key]['bbox_bottom_right_xy']
            w = x2 - x
            h = y2 - y
            area = float(w * h)

            img_number = int(dataset_yaml_map[key]['rgb_image_filename'].split('_')[0])
            mask_file_path = os.path.join(masks_path, "%06d_mask.png" % img_number)

            ground_truth_binary_mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)

            # plt.imshow(ground_truth_binary_mask)
            # plt.colorbar()

            fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
            encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
            ground_truth_area = mask.area(encoded_ground_truth)
            ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
            contours = measure.find_contours(ground_truth_binary_mask, 0.5)

            annot_dict = {'segmentation': [],
                          'area': ground_truth_area.tolist(),
                          'iscrowd': 0,
                          'image_id': id_index,
                          'bbox': [x, y, w, h],
                          'category_id': category_number,
                          'id': id_index}

            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentation = contour.ravel().tolist()
                annot_dict["segmentation"].append(segmentation)

            coco_dict['images'].append(img_dict)
            coco_dict['annotations'].append(annot_dict)
            id_index += 1

        with open(coco_json_path, "w") as f:
            json.dump(coco_dict, f)


    def get_datapath_from_config(pdc_data_root, config_file_path):
        assert os.path.exists(pdc_data_root)
        assert os.path.exists(config_file_path)

        # Read the config file
        scene_root_list = []
        with open(config_file_path, 'r') as config_file:
            lines = config_file.read().split('\n')
            for line in lines:
                if len(line) == 0:
                    continue
                scene_root = os.path.join(pdc_data_root, line)
                scene_root_list.append(scene_root)
        return  scene_root_list


config_dict = {
    'train':
        [
            "D:/Student Research Training/mankey_to_coco/mankey_dataset/2019-11-19-21-00-00"
        ],
    'val':
        [
            "D:/Student Research Training/mankey_to_coco/mankey_dataset/2019-11-19-21-00-00"
        ]
}

main_path = "D:\Student Research Training\mankey_to_coco"

log_path = os.path.join(main_path, "mankey_dataset/2019-11-19-21-00-00")
save_path = os.path.join(log_path, "bbox_test")

if os.path.exists(save_path):
     shutil.rmtree(save_path)
if not os.path.exists(save_path):
     os.makedirs(save_path)

if not os.path.exists(log_path):
    os.makedirs(log_path)

mask_to_yaml(log_path)
coco_path = os.path.join(main_path, "coco_test")
if os.path.exists(coco_path):
    shutil.rmtree(coco_path)

Make_coco = make_my_coco(coco_path)
category_number = 92

for data_path in config_dict['train']:
    Make_coco.add_data_to_coco('train', data_path=data_path,
                               category_number=category_number)
print('Total number of train dataset is :', Make_coco.get_dataset_number('train'))

for data_path in config_dict['val']:
    Make_coco.add_data_to_coco('val', data_path=data_path,
                               category_number=category_number)
    print('Total number of val dataset is :', Make_coco.get_dataset_number('val'))




