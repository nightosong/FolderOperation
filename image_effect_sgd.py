"""
Created on 2018.11.02

@author: songguodong
"""
import os
import cv2
import math
import numpy as np
import numpy.matlib


def shadow(img, add_width=20, add_height=20, style='normal'):
    """
    阴影效果
    :param img: 输入图像
    :param add_width: 垂直方向阴影距离
    :param add_height: 水平方向阴影距离
    :param style: 图片格式
    :return:
    """

    width, height, _ = img.shape
    assert abs(add_width) < width // 2 and abs(add_height) < height

    # 二值化
    if style == 'normal':
        alpha = img[:, :, 3]
        _, img_bin = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 平移特征图像
    img_transform = img_bin
    add_row = np.zeros((abs(add_width), height), dtype=np.uint8)
    add_col = np.zeros((width + abs(add_width), abs(add_height)), dtype=np.uint8)
    if add_width > 0:
        img_transform = np.row_stack((add_row, img_transform))
        sw, ew = 0, width
    else:
        img_transform = np.row_stack((img_transform, add_row))
        sw, ew = -add_width, width - add_width
    if add_height > 0:
        img_transform = np.column_stack((add_col, img_transform))
        sh, eh = 0, height
    else:
        img_transform = np.column_stack((img_transform, add_col))
        sh, eh = -add_height, height - add_height

    # 融合原始图像和阴影
    img_transform = img_transform[sw:ew, sh:eh]
    img_transform = img_transform // 2
    ret = np.empty(img.shape, np.float32)
    ret[:] = img[:]
    ret[:, :, 3] = ret[:, :, 3] + img_transform[:, :]
    ret[np.where(ret > 255)] = 255
    ret = ret.astype(np.uint8)
    return ret


def emboss(img, style='concave'):
    """
    浮雕效果
    :param img: 输入图像
    :param style: 呈现方式[concave（凹），convex（凸），bump（凹凸贴图）]
    :return:
    """

    width, height, _ = img.shape
    ret = np.empty(img.shape, np.uint8)
    ret[:, :, 3] = img[:, :, 3]
    # 灰度化
    gray = np.empty(img.shape[0:2], np.uint8)
    gray[:, :] = 0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]
    # 浮雕效果
    if style == 'concave' or style == 'bump':
        ret[:width - 1, :height - 1, 0] = 0.5 * (gray[:width - 1, :height - 1] - gray[1:width, 1:height]) + 128
        ret[:width - 1, :height - 1, 2] = ret[:width - 1, :height - 1, 1] = ret[:width - 1, :height - 1, 0]
    else:
        ret[1:width, 1:height, 0] = 0.5 * (gray[1:width, 1:height] - gray[:width - 1, :height - 1]) + 128
        ret[1:width, 1:height, 2] = ret[1:width, 1:height, 1] = ret[1:width, 1:height, 0]

    ret[np.where(ret < 0)] = 0
    ret[np.where(ret > 255)] = 255

    if style == 'bump':
        ret[:, :, :3] = 0.7 * img[:, :, :3] + 0.5 * ret[:, :, :3]
    return ret


def color_style(img, style='normal'):
    """
    颜色效果
    :param img: 输入图像
    :param style: 呈现方式[green（碧绿），green0（前一算法的另一种变现），brown（怀旧），
                ice（冰冻），fire（熔铸），dark（暗调），anti-color（反色）]
    :return:
    """
    """
    颜色效果
    """

    width, height, _ = img.shape
    ret = np.empty(img.shape, np.float32)
    ret[:, :, 3] = img[:, :, 3]
    if style == 'anti-color':
        ret[:, :, :3] = 255 - img[:, :, :3]
    elif style == 'dark':
        ret[:, :, :3] = np.power(1.0 * img[:, :, :3], 2) / 255.
    elif style == 'fire':
        gbplus = np.empty(img.shape, np.float32)
        gbplus[:, :, 0] = 1.0 + img[:, :, 1] + img[:, :, 2]
        gbplus[:, :, 1] = gbplus[:, :, 2] = gbplus[:, :, 0]
        ret[:, :, :3] = np.divide(1.0 * img[:, :, :3], gbplus[:, :, :3]) * 128.0
    elif style == 'ice':
        ret[:, :, :3] = abs(np.dot(1.0 * img[:, :, :3], [[-1., 1., 1.], [1., -1., 1.], [1., 1., -1.]]) * 1.5)
    elif style == 'brown':
        ret[:, :, :3] = np.dot(1.0 * img[:, :, :3],
                               [[0.393, 0.349, 0.272], [0.769, 0.686, 0.534], [0.189, 0.168, 0.131]])
    elif style == 'green0':
        ret[:, :, :3] = (np.dot(img[:, :, :3], [[0., -1., 1.], [-1., 0., 1.], [1., -1., 0.]]) ** 2) / 128.
    elif style == 'green':
        ret = ret.astype(np.uint8)
        ret[:, :, :3] = (np.dot(img[:, :, :3], [[0., -1., 1.], [-1., 0., 1.], [1., -1., 0.]]) ** 2) / 128.
    ret[np.where(ret < 0)] = 0
    ret[np.where(ret > 255)] = 255
    ret = ret.astype(np.uint8)
    return ret


def lighting(img, brightness=0.05):
    """
    改变图像亮度
    :param img: 输入图像
    :param brightness: 提升的亮度系数
    :return:
    """

    width, height, _ = img.shape
    mean_rgb = [0., 0., 0.]
    for i in range(3):
        mean_rgb[i] = np.sum(img[:, :, i]) / (width * height)
    ret = np.empty(img.shape, np.uint8)
    ret[:, :, 3] = img[:, :, 3]
    ret[:, :, :3] = mean_rgb + (img[:, :, :3] - mean_rgb) * (1 + brightness)
    ret[np.where(ret < 0)] = 0
    ret[np.where(ret > 255)] = 255
    ret = ret.astype(np.uint8)
    return ret


def line_style(img, line_width=5, direction='horizontal', color=None, style='equal'):
    """
    透明线条风格
    :param img: 输入图像
    :param line_width: 线条宽度
    :param direction: 线条方向[horizontal(水平), vertical(垂直), oblique(倾斜)]
    :param color: 线条颜色
    :param style: 线条格式[equal(等间距), unequal(非等距), pure(纯色)]
    :return:
    """

    width, height, channel = img.shape
    if color is None:
        color = [0] * 4
    if channel == 3:
        color = color[:3]
    cycle = line_width * 3
    ret = np.empty(img.shape, np.uint8)
    ret[:] = img[:]
    if style == 'pure':
        ret[:, :, :3] = [15, 19, 36]
    if direction == 'oblique':
        cycle = line_width * 3
        for t in range(line_width):
            for i in range(t, width, cycle):
                ret[range(width - i), range(width - 1 - i, -1, -1), :] = color
                ret[range(i + line_width, width), range(width - 1, i - 1 + line_width, -1), :] = color
    else:
        if style == 'unequal':
            for i in range(line_width):
                if direction == 'horizontal':
                    ret[::(cycle - i), :, :] = color
                else:
                    ret[:, ::(cycle - i), :] = color
        else:
            c = 2 * line_width
            for i in range(line_width):
                if direction == 'horizontal':
                    ret[i::c, :, :] = color[:]
                else:
                    ret[:, i::c, :] = color[:]

    return ret


def sphere_style(img):
    """
    鱼眼效果
    :param img:
    :return:
    """
    width, height, _ = img.shape
    radius1 = height / 2
    radius2 = radius1 * radius1
    middle2 = 2 * radius1 / math.pi
    ret = np.zeros((width, height, 4), dtype=np.uint8)
    for j in range(height):
        for i in range(width):
            dis = pow(j - height / 2, 2) + pow(i - width / 2, 2)
            if dis < radius2:
                ret[i, j, 3] = 255
                x1 = i - width / 2
                y1 = height / 2 - j
                if x1 != 0:
                    oa = middle2 * math.asin(math.sqrt(y1 * y1 + x1 * x1) / radius1)
                    ang1 = math.atan2(y1, x1)
                    x = math.cos(ang1) * oa
                    y = math.sin(ang1) * oa
                else:
                    y = math.asin(y1 / radius1) * middle2
                    x = 0
                ori_j = int(height / 2 - y)
                ori_i = int(x + width / 2)
                dis_j = (height / 2 - y) - ori_j
                dis_i = (x + width / 2) - ori_i
                plus_i = ori_i + 1
                plus_j = ori_j + 1
                if j == height - 1:
                    plus_j = ori_j
                if i == width - 1:
                    plus_i = ori_i

                ret[i, j, :3] = \
                    img[ori_i, ori_j, :3] * (1 - dis_i) * (1 - dis_j) + img[ori_i, plus_j, :3] * dis_i * (1 - dis_j) + \
                    img[plus_i, ori_j, :3] * dis_j * (1 - dis_i) + img[plus_i, plus_j, :3] * dis_j * dis_i
    return ret


def knit_initializer(style='hole', hole_width=10, belt_width=30):
    """
    初始化编织图案
    :param style: 呈现方式[hole(带孔洞的编织效果), no-hole(不带黑孔的编织效果),
                block(方格孔洞效果)]
    :param hole_width: 孔洞宽度
    :param belt_width: 编织带宽度
    :return:
    """

    hw, bw = hole_width, belt_width
    cycle = 2 * (hole_width + belt_width)
    img_shape = (512, 512, 4)
    width, height = img_shape[0], img_shape[1]

    # 计算亮度调节幅度-
    def cal_knit_bright(x):
        return -(x - hw / 2.) * (x - bw - hw * 1.5 + 1) / bw / 1.5

    def cal_texture_belt():
        higher = cal_knit_bright(hw + (bw - 1) / 2.)
        lower = cal_knit_bright(0)
        stretch = higher - lower
        temp_img = np.zeros(img_shape, dtype=np.uint8)
        for i in range(width):
            ic = i % cycle
            for j in range(height):
                jc = j % cycle
                if (ic < hw and jc < hw) or (hw + bw - 1 < ic < 2 * hw + bw and jc < hw) or \
                        (hw + bw - 1 < jc < 2 * hw + bw and ic < hw) or \
                        (hw + bw - 1 < ic < 2 * hw + bw and hw + bw - 1 < jc < 2 * hw + bw):
                    temp_img[i, j, :] = [0, 0, 0, 128]
                    continue
                elif hw - 1 < ic < hw + bw and (jc < hw or (hw + bw - 1 < jc < 2 * hw + 2 * bw)):
                    if jc < hw:
                        filled = (cal_knit_bright(hw - jc - 1) - lower) / stretch * 255
                    else:
                        filled = (cal_knit_bright(jc - hw - bw) - lower) / stretch * 255
                elif ic < 2 * hw + bw and (hw - 1 < jc < hw + bw):
                    filled = (cal_knit_bright(ic) - lower) / stretch * 255
                elif ic > 2 * hw + bw - 1 and jc < 2 * hw + bw:
                    filled = (cal_knit_bright(jc) - lower) / stretch * 255
                else:
                    if ic < hw:
                        filled = (cal_knit_bright(hw - ic - 1) - lower) / stretch * 255
                    else:
                        filled = (cal_knit_bright(ic - hw - bw) - lower) / stretch * 255
                temp_img[i, j, :] = [filled, filled, filled, 255]
        return temp_img

    def cal_texture_block():
        higher = 255
        lower = 0
        cur_cycle = 2 * belt_width
        cur_temp = np.zeros(img_shape, dtype=np.uint8)
        for i in range(width):
            ic = i % cur_cycle
            for j in range(height):
                jc = j % cur_cycle
                if (ic < belt_width and jc < belt_width) or \
                        (ic >= belt_width and jc >= belt_width):
                    cur_temp[i, j] = higher
                elif (cur_cycle - hole_width > ic >= belt_width + hole_width and
                      belt_width - hole_width > jc >= hole_width) or \
                        (belt_width - hole_width > ic >= hole_width and
                         cur_cycle - hole_width > jc >= belt_width + hole_width):
                    cur_temp[i, j] = (higher + lower) / 2
                else:
                    cur_temp[i, j] = lower
        return cur_temp

    if style == 'block':
        return cal_texture_block()
    else:
        return cal_texture_belt()


def knitting(img, temp_pic, percent=0.3, style='hole'):
    """
    编织效果
    :param img: 输入图像
    :param temp_pic: 编织带图案
    :param percent: 编织带特征占比
    :param style: 呈现方式[hole(带孔洞的编织效果), no-hole(不带黑孔的编织效果),
                block(方格孔洞效果)]
    :return:
    """

    width, height, _ = img.shape
    ret = np.empty(img.shape, np.uint8)
    ret[:, :, 3] = img[:, :, 3]
    ret[:, :, :3] = temp_pic[:width, :height, :3]
    ret[:, :, :3] = (1 - percent) * img[:, :, :3] + percent * ret[:, :, :3]

    if style != 'hole':
        return ret
    # 添加孔洞
    pos = np.where(temp_pic == 128)
    len_ls = len(pos[0])
    for i in range(3):
        pos_ls = list(pos)
        pos_ls[2] = [i] * len_ls
        ret[tuple(pos_ls)] = 0
    return ret


def warp_affine(img, style='rt', degree=30):
    """
    仿射变换
    :param img: 输入图像
    :param style: 呈现方式[rt(旋转平移), convex(凸透镜),
            distort(扭曲), vortex(旋涡)]
    :param degree: vortex中指旋转系数
    :return:
    """

    row, col, _ = img.shape
    if style == 'rt':
        pts1 = np.float32([[0, 0], [row - 1, 0], [0, col - 1]])
        pts2 = np.float32([[row * 0.0, col * 0.1], [row * 0.9, col * 0.2], [row * 0.1, col * 0.9]])
        mat = cv2.getAffineTransform(pts1, pts2)
        ret = cv2.warpAffine(img, mat, (row, col))
    elif style == 'convex':
        ret = np.zeros(img.shape, np.float32)
        cur_radius = min(row, col) / 2
        gamma = 1.5
        center_x = (col - 1) / 2
        center_y = (row - 1) / 2
        xx = np.arange(col)
        yy = np.arange(row)
        x_mask = np.matlib.repmat(xx, row, 1)
        y_mask = np.matlib.repmat(yy, col, 1)
        y_mask = np.transpose(y_mask)
        xx_dif = x_mask - center_x
        yy_dif = center_y - y_mask
        r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
        theta = np.arctan(yy_dif / xx_dif)
        mask_1 = xx_dif < 0
        theta = theta * (1 - mask_1) + (theta + math.pi) * mask_1
        r_new = cur_radius * np.power(r / cur_radius, gamma)
        x_new = r_new * np.cos(theta) + center_x
        y_new = center_y - r_new * np.sin(theta)
        int_x = np.floor(x_new)
        int_x = int_x.astype(int)
        int_y = np.floor(y_new)
        int_y = int_y.astype(int)
        int_x[np.where(int_x >= col)] = col - 1
        int_y[np.where(int_y >= row)] = row - 1
        int_x[np.where(int_x < 0)] = 0
        int_y[np.where(int_y < 0)] = 0
        ret[:, :] = img[int_y[:, :], int_x[:, :]]
        ret = ret.astype(np.uint8)
    elif style == 'distort':
        ret = np.empty(img.shape, np.uint8)
        for i in range(row):
            offset_x = int(20.0 * math.cos(2 * math.pi * i / 180.))
            if offset_x >= 0:
                ret[i, range(col - offset_x)] = img[i, range(offset_x, col)]
            else:
                ret[i, range(-offset_x, col)] = img[i, range(col + offset_x)]
    elif style == 'vortex':
        ret = np.empty(img.shape, np.uint8)
        center_x = (col - 1) / 2
        center_y = (row - 1) / 2
        xx = np.arange(col)
        yy = np.arange(row)
        x_mask = np.matlib.repmat(xx, row, 1)
        y_mask = np.matlib.repmat(yy, col, 1)
        y_mask = np.transpose(y_mask)
        xx_dif = x_mask - center_x
        yy_dif = center_y - y_mask

        r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
        theta = np.arctan(yy_dif / xx_dif)
        mask_1 = xx_dif < 0
        theta = theta * (1 - mask_1) + (theta + math.pi) * mask_1
        theta = theta + r / degree
        x_new = r * np.cos(theta) + center_x
        y_new = center_y - r * np.sin(theta)
        int_x = np.floor(x_new).astype(int)
        int_y = np.floor(y_new).astype(int)
        int_x[np.where(int_x >= row)] = row - 1
        int_y[np.where(int_y >= col)] = col - 1
        int_x[np.where(int_x < 0)] = 0
        int_y[np.where(int_y < 0)] = 0
        ret[:, :] = img[int_x[:, :], int_y[:, :]]
    else:
        ret = img

    return ret


def shade_initializer(shade_style, cycle=40):
    """
    阴影效果图案
    :param shade_style: 呈现方式[sin_h(水平波纹), sin_v(垂直波纹),
            sin_s or sin_n(交汇波纹), wave(同心波纹), triangle(三角格子)]
    :param cycle:
    :return:
    """
    width, height = 512, 512

    def sin_bright(x):
        return 255. * (math.sin(2 * math.pi * x / cycle) + 1) / 2

    def cos_bright(x):
        return 255. * (math.cos(2 * math.pi * x / cycle) + 1) / 2

    def triangle_bright(ic, jc):
        tn = cycle * 1.414 / 4.
        ln = (ic + jc) / 1.414 - 1.414 * jc
        return (1 - ln / tn) * 255.

    def wave_bright(ic, jc):
        ln = math.sqrt(pow(ic - width / 2, 2) + pow(jc - height / 2, 2))
        ln = ln % cycle
        return 255. * (math.sin(2 * math.pi * ln / cycle) + 1) / 2

    def cal_bright(ic, jc):
        ic = ic % cycle
        jc = jc % cycle

        if shade_style == 'sin_h':
            return sin_bright(ic)
        elif shade_style == 'sin_v':
            return sin_bright(jc)
        elif shade_style == 'sin_s':
            return 0.5 * sin_bright(ic) + 0.5 * sin_bright(jc)
        elif shade_style == 'sin_n':
            return 0.5 * sin_bright(ic) + 0.5 * cos_bright(jc)
        elif shade_style == 'triangle':
            if ic < cycle / 2 and jc < cycle / 2:
                return triangle_bright(max(ic, jc), min(ic, jc))
            elif jc < cycle / 2 <= ic:
                return triangle_bright(max(cycle - ic, jc), min(cycle - ic, jc))
            elif ic < cycle / 2 < jc:
                return triangle_bright(max(cycle - jc, ic), min(cycle - jc, ic))
            else:
                return triangle_bright(max(cycle - jc, cycle - ic), min(cycle - jc, cycle - ic))
        elif shade_style == 'wave':
            return wave_bright(ic, jc)

    temp_img = np.empty((width, height, 4), np.uint8)
    for i in range(width):
        for j in range(height):
            br = cal_bright(i, j)
            temp_img[i, j, :3] = [br] * 3
    return temp_img


def shade_effect(img, temp_param, percent=0.4):
    """
    明暗效果
    :param img: 输入图像
    :param temp_param: 明暗纹理图案
    :param percent: 纹理颜色占比
    :return:
    """

    width, height, _ = img.shape
    ret = np.empty(img.shape, np.uint8)
    ret[:, :, 3] = img[:, :, 3]
    ret[:, :, :3] = (1 - percent) * img[:, :, :3] + percent * temp_param[:width, :height, :3]
    ret = ret.astype(np.uint8)
    return ret


def cal_contour(img):
    # 边缘提取
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[:, :] = 0.5 * gray[:, :] + 0.5 * img[:, :, 3]
    canny = cv2.Canny(gray, 50, 200)
    # 标记轮廓连通域
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    len_c = len(contours)
    cc = np.zeros(gray.shape, np.uint8)
    for i in range(len_c):
        cx = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(cx, contours[i], 1)
        cc[:] = cc[:] + cx[:]
    cc[:] = cc[:] / 2

    cc[np.where(cc % 2 == 1)] = 127
    cc[np.where(cc % 2 == 0)] = 255
    return cc


def template_maps(img, style='bg0'):
    """
    模板贴图
    :param img: 输入图像
    :param style: 呈现方式[style.png 代表模板图像]
    :return:
    """

    row, col, _ = img.shape
    if style == 'bg0':
        img_temp = image_utils.read('./maps_bg/bg0.png')
        row_temp, col_temp, _ = img_temp.shape

        ret = np.empty(img.shape, np.uint8)
        for i in range(row):
            for j in range(col):
                new_i = i / row * row_temp
                new_j = j / col * col_temp
                ret[i, j] = img_temp[int(new_i), int(new_j)]
        for i in range(row_temp):
            for j in range(col_temp):
                img_temp[i, j] = img[int(i / row_temp * row), int(j / col_temp * col)]
        # 倾斜并添加阴影
        pts1 = np.float32([[0, 0], [row_temp - 1, 0], [0, col_temp - 1]])
        pts2 = np.float32([[0, 0], [row_temp - 1, 0], [0, (col_temp - 1) * 0.9]])
        mat = cv2.getAffineTransform(pts1, pts2)
        ret2 = cv2.warpAffine(img_temp, mat, (row_temp, col_temp))
        img_temp = shadow(ret2, 5, 5)
        for i in range(row):
            if row / 4 < i < row * 3 / 4:
                for j in range(col):
                    if col / 4 < j < col * 3 / 4:
                        if img_temp[int(i - row / 4), int(j - col / 4), 3] == 0:
                            continue
                        ret[i, j, :3] = 0.5 * ret[i, j, :3] + 0.5 * img_temp[int(i - row / 4), int(j - col / 4), :3]
        return ret
    elif style in ['bg1', 'bg4', 'bg5', 'bg6', 'bg8']:
        back_img = image_utils.read('./maps_bg/' + style + '.png')

        # 放缩
        row_scale, col_scale = 0.5, 0.5
        ret = np.empty((math.ceil(row * row_scale), math.ceil(col * col_scale), 4), np.uint8)
        if style == 'bg6':
            brown1 = color_style(img, 'brown')
            warp1 = warp_affine(brown1, 'warp')
            ret2 = shadow1 = None
        elif style == 'bg8':
            shadow1 = shadow(img, 15, 15)
            ret2 = np.empty((math.ceil(row * row_scale), math.ceil(col * col_scale), 4), np.uint8)
            warp1 = None
        else:
            ret2 = warp1 = shadow1 = None
        for i in range(math.ceil(row * row_scale)):
            for j in range(math.ceil(col * col_scale)):
                new_i = i / row_scale
                new_j = j / col_scale
                if style == 'bg6':
                    ret[i, j] = warp1[int(new_i), int(new_j)]
                elif style == 'bg8':
                    ret[i, j] = shadow1[int(new_i), int(new_j)]
                    ret2[i, j] = img[int(new_i), int(new_j)]
                else:
                    ret[i, j] = img[int(new_i), int(new_j)]
        # 灰度化+边缘检测
        gray_img = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray_img, 10, 220)
        # 镶嵌
        if style == 'bg1':  # scale=0.5
            row_min = row / 3
            row_max = row_min + row * row_scale
            col_min = col / 4
            col_max = col_min + col * col_scale
        elif style in ['bg4', 'bg5', 'bg8']:  # scale=0.5
            row_min = row / 4
            row_max = row_min + row * row_scale
            col_min = col / 4
            col_max = col_min + col * col_scale
        elif style == 'bg6':
            row_min = row / 2
            row_max = row_min + row * row_scale
            col_min = col * 0.38
            col_max = col_min + col * col_scale
        else:
            row_min = col_min = 0
            row_max = col_max = row
        for i in range(row):
            if row_min < i < row_max:
                for j in range(col):
                    if col_min < j < col_max:
                        ii = int(i - row_min)
                        jj = int(j - col_min)
                        if ret[ii, jj, 3] > 0:
                            if canny[ii, jj] == 0:
                                if style == 'bg6':
                                    back_img[i, j, :3] = ret[ii, jj, :3]
                                elif style == 'bg8':
                                    back_img[i, j] = [222, 222, 222, 255]
                                    if ret2[ii, jj, 3] == 0:
                                        back_img[i, j, :3] = [115, 20, 40]
                                else:
                                    back_img[i, j, :3] = [255] * 3
                            else:
                                if style == 'bg8':
                                    back_img[i, j, :3] = [200] * 3

        return back_img
    elif style in ['bg2', 'bg3']:
        back_img = image_utils.read('./maps_bg/' + style + '.png')
        ret = np.empty(img.shape, np.uint8)
        img_bump = emboss(img, 'bump')
        for i in range(row):
            for j in range(col):
                if style == 'bg2' and row * 0.2 < i < row * 0.8 and col * 0.2 < j < col * 0.8:
                    ret[i, j] = img_bump[int((i - row * 0.2) * 5 / 3), int((j - col * 0.2) * 5 / 3)]
                if style == 'bg3' and row * 0.3 < i < row * 0.7 and col * 0.3 < j < col * 0.7:
                    ret[i, j] = img_bump[int(2.5 * (i - row * 0.3)), int(2.5 * (j - col * 0.3))]
                if back_img[i, j, 3] > 0:
                    if style == 'bg2':
                        if i < row / 2 and j > col / 2:
                            if ret[i, j, 3] == 0:
                                ret[i, j] = [227, 207, 87, 200]
                        else:
                            ret[i, j] = [227, 207, 87, 200]
                    elif style == 'bg3':
                        if ret[i, j, 3] == 0 and back_img[i, j, 3] > 0:
                            ret[i, j] = back_img[i, j]
        return ret
    elif style == 'bg7':
        bg1 = image_utils.read('./maps_bg/' + style + '.png')
        bg2 = image_utils.read('./maps_bg/' + style + '1.png')
        # 放缩
        row_scale, col_scale = 0.6, 0.6
        ret = np.empty((math.ceil(row * row_scale), math.ceil(col * col_scale), 4), np.uint8)
        for i in range(math.ceil(row * row_scale)):
            for j in range(math.ceil(col * col_scale)):
                new_i = i / row_scale
                new_j = j / col_scale
                ret[i, j] = img[int(new_i), int(new_j)]
        # # 灰度化+边缘检测
        # gray_img = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
        # 镶嵌
        if style == 'bg7':  # scale=0.5
            row_min = row / 4
            row_max = row_min + row * row_scale
            col_min = col / 5
            col_max = col_min + col * col_scale
        else:
            row_min = col_min = 0
            row_max = col_max = row
        for i in range(row):
            if row_min < i < row_max:
                for j in range(col):
                    if col_min < j < col_max:
                        if bg2[i, j] > 0:
                            continue
                        ii = int(i - row_min)
                        jj = int(j - col_min)
                        if ret[ii, jj, 3] > 0:
                            if i + j > (row + col) / 2:
                                ratio = (i + j) * 2 / (row + col) - 1
                                ratio = 2 * ratio
                                bg1[i, j, :3] = (1 - ratio) * ret[ii, jj, :3] + ratio * bg1[i, j, :3]
                            else:
                                bg1[i, j] = ret[ii, jj]

        return bg1
    elif style == 'bg9':
        back_img = image_utils.read('./maps_bg/' + style + '.png')
        edge_color = back_img[5, 5, :3]
        cc = cal_contour(img)
        for i in range(row):
            for j in range(col):
                if img[i, j, 3] > 0:
                    if cc[i, j] == 255:
                        back_img[i, j, :3] = edge_color
                    elif cc[i, j] == 127:
                        back_img[i, j, :3] = 0.5 * edge_color + 0.5 * back_img[i, j, :3]

        return back_img
    elif style == 'bg10':
        back_img = img
        # edge_color = back_img[5, 5, :3]
        cc = cal_contour(img)
        ret = np.zeros(img.shape, np.uint8)
        method = 'separate'
        if method == 'union':
            ret[:] = back_img[:]

        for i in range(row):
            for j in range(col):
                if img[i, j, 3] > 0:
                    if method == 'union':
                        if cc[i, j] == 9:
                            ret[i, j, :3] = [189, 182, 204]
                        elif cc[i, j] == 10:
                            ret[i, j, :3] = [21, 23, 63]
                        elif cc[i, j] == 11:
                            ret[i, j, :3] = [40, 150, 224]
                    else:
                        if cc[i, j] == 9:
                            ret[i - 5, j - 5] = [189, 182, 204, 255]
                        elif cc[i, j] == 10:
                            ret[i + 5, j + 5] = [21, 23, 63, 255]
                        elif cc[i, j] == 11:
                            ret[i, j] = [40, 150, 224, 255]

        return ret


def deal(path, save_dir, modes='shadow', temp_param=None):
    """
    图像特效
    :param path: 图像路径
    :param save_dir: 储存路径
    :param modes: 特效类型  shadow（阴影），emboss（凹凸浮雕），color-style（颜色效果），
            lighting（亮度调整），line-style（线条），sphere-style（立体球），
            knit-style（编织），warp-affine（仿射变换），shade-style（明暗效果）
    :param temp_param: 额外参数
    :return:
    """

    img_load = image_utils.read(path)
    if modes == 'shadow':
        ret = shadow(img_load, add_width=20, add_height=20)
    elif modes == 'emboss':
        ret = emboss(img_load, style='convex')
    elif modes == 'color-style':
        ret = color_style(img_load, style='green')
    elif modes == 'lighting':
        ret = lighting(img_load)
    elif modes == 'line':
        ret = line_style(img_load, 10, direction='oblique', style='unequal')
    elif modes == 'sphere-style':
        ret = sphere_style(img_load)
    elif modes == 'knit':
        ret = knitting(img_load, temp_param, percent=0.4, style='hole')
    elif modes == 'warp-affine':
        ret = warp_affine(img_load, style='distort')
    elif modes == 'shade':
        ret = shade_effect(img_load, temp_param)
    elif modes == 'maps':
        ret = template_maps(img_load, style='bg10')
    else:
        ret = img_load
    image_utils.save(save_dir, ret)


if __name__ == '__main__':
    load_path = './pack_50-Free-Filled-Outline_1368/'
    save_path = './style/zzz-test/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    images = os.listdir(load_path)
    mode = 'knit'
    temp = None
    if mode == 'knit':
        temp = knit_initializer('hole', hole_width=10, belt_width=10)
    elif mode == 'shade':
        temp = shade_initializer('triangle')

    for im in range(len(images)):
        deal(load_path + images[im], save_path + mode + '_' + images[im], mode, temp)
        if im >= 9:
            break
    assert 1 == 0
    # cnt = multiprocessing.cpu_count()
    # eve = math.ceil(10 / cnt)
    # for th in range(cnt):
    #     for x in range(eve):
    #         if eve * th + x >= 10:
    #             break
    #         img = images[eve * th + x]
    #         t = threading.Thread(target=deal,
    #                              args=(load_path + img,
    #                                    save_path + mode + '0_' + img,
    #                                    mode, temp))
    #         t.start()
