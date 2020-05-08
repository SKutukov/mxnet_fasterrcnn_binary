import cv2
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
from bounding_box import bounding_box as bb
import os

def vis_detection(im_orig, detections, class_names, file='', thresh=0.7):
    """visualize [cls, conf, x1, y1, x2, y2]"""
    cmap = ['black', 'navy', 'blue',  'silver', 'aqua', 'teal', 'olive',  'purple', 'green',
             'fuchsia', 'lime',  'red', 'yellow',
             'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray', 'silver']

    im_orig = cv2.cvtColor(im_orig, cv2.COLOR_BGR2RGB)

    for [cls, conf, x1, y1, x2, y2] in detections:
        cls = int(cls)
        if cls > 0 and conf > thresh:

            bb.add(im_orig, int(x1), int(y1), int(x2), int(y2),
                   '{:s} {:.3f}'.format(class_names[cls], conf),
                   cmap[cls])

    # plt.axis('off')
    # # plt.show()
    # plt.savefig('test.png')
    # cv2.imshow("d", im_orig)
    cv2.imwrite(os.path.join('/home/skutukov/Pictures/result_full_precision', file.strip() + '.jpg'), im_orig)
