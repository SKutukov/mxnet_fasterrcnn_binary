import argparse
import cv2
import pandas as pd
import os
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

    if not os.path.exists(os.path.join(out_path, "img")):
        os.mkdir(os.path.join(out_path, "img"))

    df_result = pd.DataFrame(columns=('filename', 'x_from', 'y_from', 'width', 'height', 'class'))
    # begin generate dataset
    for i in range(args.count):
        image, bboxes, numbers = generator.generate_image_and_bboxes(image_width=width, image_height=height)
        file_name = str(i) + ".jpg"
        file_name = os.path.join(out_path, "img", file_name)
        cv2.imwrite(file_name, image)

        for box, number in zip(bboxes, numbers):
            temp = [(file_name, box.x, box.y, box.x + box.width, box.y + box.height, class_dict[number])]

            dfObj = pd.DataFrame(temp, columns=['filename', 'x_from', 'y_from',
                                                'width', 'height', 'class'])
            df_result = df_result.append(dfObj)

    # write csv to filesystem
    path_to_train_csv = "res_train.csv"
    path_to_val_csv = "res_val.csv"
    split_count = int(args.count * 0.8)
    df_result_train, df_result_val = df_result.iloc[:split_count, :], df_result.iloc[split_count:, :]

    full_path_to_csv = os.path.join(out_path, path_to_train_csv)
    df_result.to_csv(full_path_to_csv, index=False, header=False)
    full_path_to_csv = os.path.join(out_path, path_to_val_csv)
    df_result.to_csv(full_path_to_csv, index=False, header=False)


    df_result = pd.DataFrame(columns=('class_name', 'class_id'))
    for key, value in class_dict.items():
        temp = [(value, key)]
        dfObj = pd.DataFrame(temp, columns=['class_name', 'class_id'])
        df_result = df_result.append(dfObj)


    # write class map csv to filesystem
    path_to_csv = "classes.csv"
    full_path_to_csv = os.path.join(out_path, path_to_csv)
    df_result.to_csv(full_path_to_csv, index=False, header=False)