import argparse
import cv2
import pandas as pd
import os
from dataset_generator_mnist.image_generator import ImageGenerator
from lxml import etree

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist_folder", type=str,
                        help="path to mnist dataset")
    parser.add_argument("--width", type=int,
                        help="width of generated images")
    parser.add_argument("--height", type=int,
                        help="height of generated images")
    parser.add_argument("--count", type=int,
                        help="count of generated images")
    parser.add_argument("--out_path", type=str,
                        help="count of generated images")

    return parser.parse_args()


if __name__ == "__main__":

    class_dict = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine"

    }
    # parse argument
    args = parse_argument()
    width = args.width
    height = args.height
    out_path = args.out_path

    generator = ImageGenerator(args.mnist_folder)

    # generate folder if not exist
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if not os.path.exists(os.path.join(out_path, "MNIST")):
        os.mkdir(os.path.join(out_path, "MNIST"))

    if not os.path.exists(os.path.join(out_path, "MNIST", "Annotations")):
        os.mkdir(os.path.join(out_path, "MNIST", "Annotations"))

    if not os.path.exists(os.path.join(out_path, "MNIST", "JPEGImages")):
        os.mkdir(os.path.join(out_path, "MNIST", "JPEGImages"))

    if not os.path.exists(os.path.join(out_path, "MNIST", "ImageSets")):
        os.mkdir(os.path.join(out_path, "MNIST", "ImageSets"))

    if not os.path.exists(os.path.join(out_path, "MNIST", "ImageSets", "Main")):
        os.mkdir(os.path.join(out_path, "MNIST", "ImageSets", "Main"))
    # bash scripts / run_test_faster_rcnn_qvgg_CI.sh 2;  bash scripts / run_test_faster_rcnn_qvgg_CI.sh  3;  bash  scripts / run_test_faster_rcnn_qvgg_CI.sh 4; bash scripts / run_test_faster_rcnn_qvgg_CI.sh   # 5;  bash
    # scripts / run_test_faster_rcnn_qvgg_CI.sh
    # 1;
    # bash
    # scripts / run_test_faster_rcnn_qvgg_CI.sh
    # 0

    file_ids = []
    # begin generate dataset
    for i in range(args.count):
        file_ids.append(i)
        image, bboxes, numbers = generator.generate_image_and_bboxes(image_width=width, image_height=height)
        file_name = str(i) + ".jpg"
        xml_name = str(i) + ".xml"

        full_file_name = os.path.join(out_path, "MNIST", "JPEGImages", file_name)
        full_xml_name = os.path.join(out_path, "MNIST", "Annotations", xml_name)
        cv2.imwrite(full_file_name, image)

        root = etree.Element('annotation')

        child = etree.Element('folder')
        child.text = "MNIST"
        root.append(child)

        child = etree.Element('filename')
        child.text = file_name
        root.append(child)

        child = etree.Element('size')
        child_width = etree.Element('width')
        child_width.text = str(width)
        child.append(child_width)
        child_height = etree.Element('height')
        child_height.text = str(height)
        child.append(child_height)
        child_depth = etree.Element('depth')
        child_depth.text = str(3)
        child.append(child_depth)

        root.append(child)

        child = etree.Element('segmented')
        child.text = "0"
        root.append(child)

        for box, number in zip(bboxes, numbers):
            child_box = etree.Element('object')
            child_name = etree.Element('name')
            child_name.text = class_dict[number]
            child_box.append(child_name)

            child_pose = etree.Element('pose')
            child_pose.text = "Unspecified"
            child_box.append(child_pose)

            child_truncated = etree.Element('truncated')
            child_truncated.text = "0"
            child_box.append(child_truncated)

            child_difficult = etree.Element('difficult')
            child_difficult.text = "0"
            child_box.append(child_difficult)

            # temp = [(file_name, , box.y, box.x + box.width, box.y + box.height, class_dict[number])]
            child_bndbox = etree.Element('bndbox')

            child_xmin = etree.Element('xmin')
            child_xmin.text = str(box.x)
            child_bndbox.append(child_xmin)

            child_ymin = etree.Element('ymin')
            child_ymin.text = str(box.y)
            child_bndbox.append(child_ymin)

            child_xmax = etree.Element('xmax')
            child_xmax.text = str(box.x + box.width)
            child_bndbox.append(child_xmax)

            child_ymax = etree.Element('ymax')
            child_ymax.text = str(box.y + box.height)
            child_bndbox.append(child_ymax)

            child_box.append(child_bndbox)

            root.append(child_box)

        with open(full_xml_name, "wb") as xml_file:
            s = etree.tostring(root, pretty_print=True)
            xml_file.write(s)
        xml_file.close()

    file_ids_train = file_ids[:int(0.25 * len(file_ids))]
    file_ids_val = file_ids[int(0.25 * len(file_ids)):]
    train_path = os.path.join(out_path, "MNIST", "ImageSets", "Main", "trainval.txt")
    val_path = os.path.join(out_path, "MNIST", "ImageSets", "Main", "test.txt")

    with open(train_path, "a") as train_file:
        for index in file_ids_train:
            train_file.write(str(index) + "\n")

    with open(val_path, "a") as val_file:
        for index in file_ids_val:
            val_file.write(str(index) + "\n")
