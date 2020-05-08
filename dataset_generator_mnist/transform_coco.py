import argparse
import cv2
import os
import json
from mnist.image_generator import ImageGenerator

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

def safe_makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

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
    safe_makedir(out_path)
    safe_makedir(os.path.join(out_path, "images"))
    safe_makedir(os.path.join(out_path, "annotations"))

    def generate_set(mode, class_dict, count):
        image_base_path = os.path.join(out_path, "images", mode)
        safe_makedir(image_base_path)

        json_result = {}
        json_result["type"] = "MNIST"
        images_coco = []
        bboxes_coco = []
        # begin generate dataset
        bb_id = 0
        for image_index in range(count):
            image, bboxes, numbers = generator.generate_image_and_bboxes(image_width=width, image_height=height)
            base_name = str(image_index) + ".jpg"
            file_name = os.path.join(image_base_path, base_name)
            cv2.imwrite(file_name, image)

            im_dict = {
                 "file_name": base_name,
                 "height": height,
                 "width": width,
                 "id": image_index
                }
            images_coco.append(im_dict)

            for box, number in zip(bboxes, numbers):
                box_dict = {
                    "id": bb_id,
                    "bbox": [
                        box.x,
                        box.y,
                        box.width,
                        box.height
                    ],
                    "image_id": image_index,
                    "segmentation": [],
                    "ignore": 0,
                    "area": box.width * box.height,
                    "iscrowd": 0,
                    "category_id": number
                }
                bb_id += 1
                bboxes_coco.append(box_dict)


        json_result["images"] = images_coco
        json_result["annotations"] = bboxes_coco

        categories = []
        for key, value in class_dict.items():
            class_dict = {
                "supercategory": "none",
                "name": value,
                "id": key
                }
            categories.append(class_dict)
        json_result["categories"] = categories

        name = "instances_" + mode + ".json"
        with open(os.path.join(os.path.join(out_path, "annotations"), name), 'w') as f:
            json.dump(json_result, f, sort_keys=True, indent=4)


    generate_set("train1001", class_dict, int(args.count * 0.8))
    generate_set("test1001", class_dict, int(args.count * 0.2))
