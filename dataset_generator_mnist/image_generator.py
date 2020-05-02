import cv2
import random
from utils.bbox import BBox
import os
import numpy as np

class ImageGenerator:
    """

    """
    def __init__(self, folder_path: str):
        """

        :param folder_path: path to folder with images
        """

        self.folder_path = folder_path
        self.max_bbox = 1
        self.max_box_width = 64
        self.min_box_width = 32
        self.max_box_height = 64
        self.min_box_height = 32

        # init dict with folder names for different numbers
        self.number_images = {}
        for number in range(10):
            self.number_images[number] = os.listdir(os.path.join(self.folder_path, str(number)))

    def generate_bbox(self, image_width: int, image_height: int):
        """
        generate one bbox
        :param image_width: original image width
        :param image_height: original image height
        :return: bbox as class BBox
        """
        x = random.randint(0, image_width - self.max_box_width)
        y = random.randint(0, image_height - self.max_box_height)

        width = random.randint(self.min_box_width, self.max_box_width)
        height = random.randint(self.min_box_height, self.max_box_height)

        return BBox(x=x,
                    y=y,
                    width=width,
                    height=height)

    def generate_bboxes(self, image_width: int, image_height: int, count: int):
        """
        generate several bbox
        :param image_width: original image width
        :param image_height: original image height
        :param count: count of generated bbox
        :return: bboxes as list of class BBox
        """
        bboxes = []
        for i in range(count):
            bboxes.append(self.generate_bbox(image_width, image_height))
        return bboxes

    def generate_image_and_bboxes(self, image_width: int, image_height: int):
        """

        :param image_width: original image width
        :param image_height: original image height
        :return:
        """
        count = random.randint(1, self.max_bbox)
        bboxes = self.generate_bboxes(image_width, image_height, count)
        numbers = []
        zeros_image = np.zeros(shape=(image_height, image_width))
        for box in bboxes:
            number = random.randint(0, 9)
            image_id = random.randint(0, len(self.number_images[number]) - 1)
            number_image = cv2.imread(os.path.join(self.folder_path, str(number), self.number_images[number][image_id]),
                                      0)
            zeros_image[box.y:box.y + box.height, box.x:box.x + box.width] = \
                cv2.resize(number_image, (int(box.width), int(box.height)))
            numbers.append(number)

        return zeros_image, bboxes, numbers