import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


# 輪郭の頂点の数が最大のものを,最大の輪郭として採用している
def get_max_index(contours):
    max_num = 0
    max_i = -1
    for i, cnt in enumerate(contours):
        cnt_num = len(cnt)
        if cnt_num > max_num:
            max_num = cnt_num
            max_i = i
    return max_i


ix, iy, ex, ey = -1, -1, -1, -1
fill_x, fill_y = -1, -1
is_rect = False


def mouse_input(event, x, y, flag, param):
    global ix, iy, ex, ey, is_rect, img_temp

    if not is_rect:
        img_temp = img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = x, y
        # elif event == cv2.EVENT_MOUSEMOVE:
        elif event == cv2.EVENT_LBUTTONUP:
            ex, ey = x, y
            cv2.rectangle(img_temp, (ix, iy), (ex, ey), (0, 255, 0), 1)
            is_rect = True


def make_roi(img):
    global ix, iy, ex, ey
    w, h = abs(ix - ex), abs(iy - ey)
    if ix < ex:
        if iy < ey:
            pass
        else:
            iy -= h
            ey += h
    else:
        if iy < ey:
            ix -= w
            ex += w
        else:
            tmp = (ex, ey)
            ex, ey = ix, iy
            ix, iy = tmp

    return img[iy:ey, ix:ex]


def inPainting(img):
    # cv2.imshow('img', img)
    global img_result
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rows, cols, c = img_hsv.shape
    binary_threshold = 100
    kernel_close_parm = 3
    kernel_open_parm = 13

    padding = 10

    # roi(範囲画像)を作成し、HSVのチャンネルごとに分ける
    # roi_bgr = img[220:480, 270:520]
    roi_bgr = make_roi(img)
    roi_bgr_padding = cv2.copyMakeBorder(roi_bgr, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                         value=(255, 255, 255))
    roi_hsv = cv2.cvtColor(roi_bgr_padding, cv2.COLOR_BGR2HSV)
    # cv2.imshow('roi_bgr', roi_bgr)

    h = roi_hsv[:, :, 0]
    s = roi_hsv[:, :, 1]
    v = roi_hsv[:, :, 2]

    # cv2.imshow('H', h)
    # cv2.imshow('S', s)
    # cv2.imshow('V', v)

    # vの画像がコントラスト高めなので採用
    # vのヒストグラムを正規化して、更にコントラスを上げる
    equ_v = cv2.equalizeHist(v)
    # cv2.imshow('equ_v', equ_v)

    # 以下はヒストグラムの確認をしている
    # plt.hist(v.ravel(), 256, [0, 256]);
    # plt.hist(equ_v.ravel(), 256, [0, 256]);
    # plt.show()

    # 2値化する
    _, roi_binary, = cv2.threshold(equ_v, binary_threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow('roi_binary', roi_binary)

    # モーフィング処理
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_close_parm, kernel_close_parm))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_open_parm, kernel_open_parm))
    closing = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, kernel_close)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
    opening = cv2.bitwise_not(opening)
    # cv2.imshow('close)', closing)
    # cv2.imshow('open', opening)

    # 輪郭抽出して、マスク画像生成
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_index = get_max_index(contours)
    mask = np.zeros(opening.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, max_index, (255, 255, 255), thickness=-1, lineType=1)

    cv2.namedWindow("process_roi", cv2.WINDOW_NORMAL)
    process_roi = cv2.hconcat([v, equ_v, roi_binary, opening, mask])
    cv2.imshow("process_roi", process_roi)

    # マスクのPadding消去, 膨張, 入力画像のサイズに合わせる
    w, h = mask.shape
    mask = mask[padding:w - padding, padding:h - padding]
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.dilate(mask, kernel_dilate)
    mask_full = np.zeros((rows, cols), dtype=np.uint8)
    mask_full[iy:ey, ix:ex] = mask
    # cv2.imshow('mask', mask_full)

    # maskを用いて、元画像からオブジェクトを消去する
    roi_masked = cv2.bitwise_and(roi_bgr, roi_bgr, mask=cv2.bitwise_not(mask))
    img_masked = img.copy()
    img_masked[iy:ey, ix:ex] = roi_masked
    # cv2.imshow('img_masked', img_masked)

    # InPaint(画像修復する)
    img_inpainted = cv2.inpaint(img, mask_full, 5, cv2.INPAINT_TELEA)

    cv2.namedWindow("process_full", cv2.WINDOW_NORMAL)
    process_full = cv2.hconcat([img, img_masked, img_inpainted])
    cv2.imshow("process_full", process_full)

    # cv2.imshow('img', img)
    # cv2.imshow('roi_bgr', roi_bgr)
    # cv2.imshow('V', v)
    # cv2.imshow('equ_v', equ_v)
    # cv2.imshow('roi_binary', roi_binary)
    # cv2.imshow('close)', closing)
    # cv2.imshow('open', opening)
    # cv2.imshow('img_masked', img_masked)
    # cv2.imshow('mask', mask_full)
    cv2.imshow('img_inPainted', img_inpainted)
    img_result = img_inpainted

    key = cv2.waitKey(0) & 0xff
    if key == 27:
        cv2.destroyAllWindows()
        exit()


if __name__ == '__main__':
    args = sys.argv
    result_path = ""
    if len(args) == 1:
        src = cv2.imread('../image/test1.jpg', 1)
    elif len(args) == 2:
        src = cv2.imread(args[1], 1)
    elif len(args) == 3:
        src = cv2.imread(args[1], 1)
        result_path = args[2]
    else:
        print("Params Error")
        exit(-1)

    if src.shape[0] > 1000:
        img = cv2.resize(src, None, fx=0.5, fy=0.5)
    else:
        img = src.copy()

    img_temp = img.copy()
    img_result = img.copy()
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', mouse_input)

    while 1:
        cv2.imshow('input', img_temp)
        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        elif key == ord('r'):
            ix, iy, ex, ey = -1, -1, -1, -1
            is_rect = False
            cv2.imshow('input', img)
        elif key == ord('p'):
            print(ix, iy, ex, ey)
        elif key == ord('c'):
            # roi = make_roi(img, ix, iy, ex, ey)
            # cv2.namedWindow('output')
            # cv2.setMouseCallback('output', mouse_output)
            # print(fill_x, fill_y)
            inPainting(img)
        elif key == ord('s'):
            if result_path:
                cv2.imwrite(result_path, img_result)
            else:
                cv2.imwrite('../image/result.jpg', img_result)
            print("saved")

    cv2.destroyAllWindows()
    exit()