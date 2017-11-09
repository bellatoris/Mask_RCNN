import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import coco
import model as modellib
import visualize


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def main():
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

    img_path = sys.argv[1]
    # img_path = 'images/1045023827_4ec3e8ba5c_z.jpg'

    print(img_path)

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode='inference',
                              model_dir=MODEL_DIR,
                              config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    image = scipy.misc.imread(img_path)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'],
                                r['class_ids'], class_names,
                                r['scores'])

    rois = r['rois']
    rois = transformation(rois, image.shape[:2])

    class_ids = r['class_ids']
    _class_names = np.array(class_names)[class_ids, np.newaxis]

    result = np.hstack((_class_names, rois, r['scores'][:, np.newaxis]))
    print(result)

    plt.show()


def transformation(rois, img_shape):
    y_trans, x_trans = img_shape
    transformed_rois = [rois[:, 1, np.newaxis], y_trans - rois[:, 0, np.newaxis],
                        rois[:, 3, np.newaxis], y_trans - rois[:, 2, np.newaxis]]

    return np.hstack(transformed_rois)


if __name__ == '__main__':
    main()
