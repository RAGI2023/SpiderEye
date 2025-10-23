import os
import cv2
import numpy as np
import argparse

def main():
    arg = argparse.ArgumentParser(description="Convert dataset format")
    arg.add_argument("-i", "--input_path", type=str, required=True, help="Path to the input dataset")
    arg.add_argument("-o", "--output_path", type=str, required=True, help="Path to save the converted dataset")
    args = arg.parse_args()

    CANVAS_SIZE = (1024, 512)  # (W, H)

    os.makedirs(args.output_path, exist_ok=True)

    # 读取原图
    img1 = cv2.imread(f"{args.input_path}/1.JPG")  # [H, W, C]
    img2 = cv2.imread(f"{args.input_path}/2.JPG")

    if img1 is None or img2 is None:
        raise FileNotFoundError("Cannot find 1.JPG or 2.JPG under the given input path")

    # 拆分前后左右视图
    front = img1[:, :img1.shape[1] // 2, :]
    back = img1[:, img1.shape[1] // 2:, :]
    left = img2[:, img2.shape[1] // 2:, :]
    right = img2[:, :img2.shape[1] // 2, :]

    # resize 成统一大小 (512x512)
    img_front = cv2.resize(front, (512, 512))
    img_back  = cv2.resize(back,  (512, 512))
    img_left  = cv2.resize(left,  (512, 512))
    img_right = cv2.resize(right, (512, 512))

    outs = {}
    outs['front'] = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8) # H, W, C, uint8
    outs['front'][:, :CANVAS_SIZE[0]//2, :] = img_front

    outs['back']  = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    outs['back'][:, CANVAS_SIZE[0]//2:, :] = img_back

    outs['left']  = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    outs['left'][:, :CANVAS_SIZE[0]//4, :] = img_left[:,CANVAS_SIZE[0]//4:CANVAS_SIZE[0]//4 * 2, :]
    outs['left'][:, CANVAS_SIZE[0]//4 * 3:, :] = img_left[:, :CANVAS_SIZE[0]//4, :]

    outs['right'] = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)
    outs['right'][:, CANVAS_SIZE[0]//4:CANVAS_SIZE[0]//4 * 3, :] = img_right

    cv2.imwrite(f"{args.output_path}/front.jpg", outs['front'])
    cv2.imwrite(f"{args.output_path}/back.jpg", outs['back'])
    cv2.imwrite(f"{args.output_path}/left.jpg", outs['left'])
    cv2.imwrite(f"{args.output_path}/right.jpg", outs['right'])


if __name__ == "__main__":
    main()
