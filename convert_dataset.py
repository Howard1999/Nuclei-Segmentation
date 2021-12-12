import os
import cv2
import numpy as np
import json
import image_handle
from pycocotools import coco as COCO


def convert2coco(from_dir, to_dir, unsharp=True, grayscale=False):
    val_size = 3  # val set

    train_dir = from_dir+'/train/'
    test_dir = from_dir+'/test/'

    train_dst_dir = to_dir + '/train/'
    val_dst_dir = to_dir + '/val/'
    test_dst_dir = to_dir + '/test/'

    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    if not os.path.exists(train_dst_dir):
        os.makedirs(train_dst_dir)

    if not os.path.exists(val_dst_dir):
        os.makedirs(val_dst_dir)

    if not os.path.exists(test_dst_dir):
        os.makedirs(test_dst_dir)

    # train/val set
    coco_train = {
        'images': [],
        "annotations": [],
        "categories": [{
            'id': 0,
            'name': 'background'
        }, {
            'id': 1,
            'name': 'nuclei'
        }]
    }
    coco_val = {
        'images': [],
        "annotations": [],
        "categories": [{
            'id': 0,
            'name': 'background'
        }, {
            'id': 1,
            'name': 'nuclei'
        }]
    }
    train_list = os.listdir(train_dir)
    for i, img_name in enumerate(train_list):
        img_path = train_dir + img_name + '/images/' + img_name + '.png'
        mask_dir = train_dir + img_name + '/masks/'

        img = cv2.imread(img_path)
        h, w, c = img.shape

        if unsharp:
            img = image_handle.unsharp(img)
        if grayscale:
            img = image_handle.grayscale(img)

        if i > 24-val_size-1:
            coco = coco_val
            cv2.imwrite(val_dst_dir + '/' + img_name + '.png', img)
        else:
            coco = coco_train
            cv2.imwrite(train_dst_dir + '/' + img_name + '.png', img)

        image = {"id": i, "width": w, "height": h, "file_name": img_name + '.png'}
        coco['images'].append(image)
        for mask_name in os.listdir(mask_dir):
            if mask_name[-4:] != '.png':
                continue
            mask = cv2.imread(mask_dir + mask_name)
            rle = COCO.maskUtils.encode(np.asfortranarray(mask[:, :, 0]))
            rle['counts'] = rle['counts'].decode("utf-8")
            annotation = {
                "id": len(coco['annotations']),
                "image_id": i,
                "category_id": 1,
                "segmentation": rle,
                "area": COCO.maskUtils.area(rle).item(),
                "bbox": COCO.maskUtils.toBbox(rle).tolist(),
                "iscrowd": 0,
            }
            coco['annotations'].append(annotation)

    coco_test = {
        "images": [],
        "categories": [{
            'id': 0,
            'name': 'background'
        }, {
            'id': 1,
            'name': 'nuclei'
        }],
        "annotations": []
    }
    # test set
    for img_name in os.listdir(test_dir):
        img = cv2.imread(test_dir + '/' + img_name)

        if unsharp:
            img = image_handle.unsharp(img)
        if grayscale:
            img = image_handle.grayscale(img)

        cv2.imwrite(test_dst_dir + '/' + img_name, img)

    # annotation files
    with open(to_dir + 'test.json', 'w') as fp:
        with open(from_dir + '/test_img_ids.json') as test_fp:
            coco_test['images'] = json.load(test_fp)
        fp.write(json.dumps(coco_test, sort_keys=True, indent=4))
    with open(to_dir + 'train.json', 'w') as fp:
        fp.write(json.dumps(coco_train, sort_keys=True, indent=4))
    with open(to_dir + 'val.json', 'w') as fp:
        fp.write(json.dumps(coco_val, sort_keys=True, indent=4))


if __name__ == '__main__':
    convert2coco('../dataset/', '../coco/normal/', unsharp=False, grayscale=False)
    # convert2coco('../dataset/', '../coco/unsharp/', unsharp=True, grayscale=False)
    # convert2coco('../dataset/', '../coco/unsharp_and_grayscale/', unsharp=True, grayscale=True)
