import os
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from typing import List
import time

proj_root_dir = f"{os.path.dirname(__file__)}/.."
os.chdir(proj_root_dir)
print(f"cwd: {os.getcwd()}")

FONT_PATH = "assets/chinese_font.ttf"
FONT_SIZE = 16
OCR_HIGH_CONF = 0.8
OCR_MEDIUM_CONF = 0.5
OCR_LOW_CONF = 0.3

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # need to run only once to download and load model into memory

# helper function to display ocr result
def visualizeOcrResults(img: np.ndarray, result: List) -> np.ndarray:
    # convert np array to pil image
    ocr_img = (
        cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 2
        else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )
    pil_img = Image.fromarray(ocr_img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # draw boxes and text for each ocr text
    for ele in result:
        bbox, txt_with_conf = ele
        x, y = [coor[0] for coor in bbox], [coor[1] for coor in bbox]
        tl, br = (int(min(x)), int(min(y))), (int(max(x)), int(max(y)))
        text, conf = txt_with_conf

        # choose colour according to confidence level
        if conf >= OCR_HIGH_CONF:
            colour = "green"
        elif conf >= OCR_MEDIUM_CONF:
            colour = "gold"
        elif conf >= OCR_LOW_CONF:
            colour = "orange"
        else:
            colour = "red"
        
        # draw rectangle and text
        draw.rectangle((tl, br), outline=colour, width=3)
        draw.text(
            (tl[0], tl[1] - FONT_SIZE),
            text,
            font=font,
            fill=colour,
        )

    # convert pil image back to np array
    ocr_img = np.array(pil_img)
    ocr_img = cv2.cvtColor(ocr_img, cv2.COLOR_RGB2BGR)
    return ocr_img

# ocr all pages in a directory
def test_ocr_results(dir_name: str):
    # loop through each image in the dir
    for i, file_name in enumerate(os.listdir(dir_name), start=1):
        print(f"----------------------------------page {i}---------------------------------------")
        # read image
        img_path = os.path.join(dir_name, file_name)
        img = cv2.imread(img_path)

        # get ocr result
        start_time = time.time()
        result = ocr.ocr(img, cls=False)
        end_time = time.time()
        print(f"time lapse: {end_time - start_time} sec")
        result = result[0]

        # print ocr results
        # for ele in result:
        #     bbox, txt_with_conf = ele
        #     x, y = [coor[0] for coor in bbox], [coor[1] for coor in bbox]
        #     tl, br = (int(min(x)), int(min(y))), (int(max(x)), int(max(y)))
        #     txt, conf = txt_with_conf
        #     print(f"bbox: {bbox}, tl br: {tl} {br}, txt: {txt}, conf: {conf}")

        # output image with ocr result
        print("outputting image...")
        ocr_img = visualizeOcrResults(img, result)
        cv2.imwrite(f"out/ocr_page_{i}.png", ocr_img)
        print("finished outputting image!")

# Test size on process time and performance. The image is resized to several sizes from mult_start to mult_end.
# params:
# - mult_start: max percentage to resize the image to
# - mult_end: min percentage to resize the. image to
# - steps: number of different percentage between mult_start and mult_end to resize the image
def test_diff_size_img(img_path: str, mult_start: float, mult_end, steps = 5):
    for multiplier in np.linspace(mult_start, mult_end, steps):
        # resize the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=None, fx=multiplier, fy=multiplier)
        h, w = img.shape[:2]
        print(f"-------------------------size multiplier: {multiplier}, size: {w} * {h}--------------------------")

        # compute results
        start_time = time.time()
        result = ocr.ocr(img, cls=False)
        end_time = time.time()
        print(f"time lapse: {end_time - start_time} sec")
        result = result[0]

        # output image with ocr result
        print("outputting image...")
        ocr_img = visualizeOcrResults(img, result)
        cv2.imwrite(f"out/ocr_multiplier_{multiplier}.png", ocr_img)
        print("finished outputting image!")

def main():
    test_diff_size_img("img/regal-3.png", 1, 0.5)

if __name__ == "__main__":
    main()