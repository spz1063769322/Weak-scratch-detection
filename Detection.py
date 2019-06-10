import numpy
import pandas
import cv2
import math
import copy
from Weak_Scratch_Detection.RANSAC import *

class Detection():
    def __init__(self, image_path, T1=False, T2=10, T3=40, T4=32, XI=5, n=1, m=1):
        self.T1 = 40
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.XI = XI
        self.src_image = cv2.imread(image_path, 0)
        self.image_info = self.src_image.shape
        self.image_height = self.image_info[0]
        self.image_width = self.image_info[1]
        if(T1):
            n1 = int(self.image_width / n)
            n2 = int(self.image_height / m)
            for i in range(n1):
                for j in range(n2):
                    self.T1 += self.src_image[n * i, m * j]
            self.T1 = self.T1 / (n1 * n2) + XI

    """Rules of coarse detection"""
    """
        粗检测中的四条规则，判断是否满足，如果不满足返回True，否则返回False
        
        Return：bool
    """
    def judge(self, lc, ll, lr):
        if(lc > self.T1):
            if((lc > (ll + self.T2)) and ((lc > (lr + self.T2)))):
                if(abs(int(ll) - int(lr)) < self.T3):
                    if((ll > self.T4) or (lr > self.T4)):
                        return False
        return True

    """Determine whether to merge the same line segments"""
    """
        精检测阶段判断是否满足合并两条线段的三条规则，如果满足返回True，否则返回False
        
        Return：bool
    """
    def merge(self, line_1, line_2, theta=12, d=3, g=200):
        k1 = (line_1[3] - line_1[1]) / -(line_1[2] - line_1[0])
        k2 = (line_2[3] - line_2[1]) / -(line_2[2] - line_2[0])
        angle = math.atan((k1 - k2) / (1 + k1 * k2))
        # print(angle)
        if(angle <= theta):
            gap = math.sqrt((line_2[0] - line_1[0]) ** 2 + (line_2[1] - line_1[1]) ** 2)
            # print(gap)
            # print("-" * 40)
            if(gap < g):
                return True
        return False

    def sort_line(self, lines):
        min_value = 10000000
        max_value = 0
        if(lines[1] < min_value):
            min_value = lines[1]
        if (lines[3] < min_value):
            min_value = lines[3]
            lines[0], lines[1], lines[2], lines[3] = lines[2], lines[3], lines[0], lines[1]
        if(lines[3] > max_value):
            max_value = lines[3]
        if (lines[1] > max_value):
            max_value = lines[1]
            lines[0], lines[1], lines[2], lines[3] = lines[2], lines[3], lines[0], lines[1]

    """Returns the maximum or minimum point in the line segment set"""
    """
        lines：检测的到的所有线段->[[x1, y1, x2, y2],....]
        
        Return：线段集合中最小点下标和最大点下标
    """
    def get_minmax_line(self, lines):
        min_index, min_value = 0, 10000000
        max_index, max_value = 0, 0
        for i in range(len(lines)):
            if(lines[i][1] < min_value):
                min_index = i
                min_value = lines[i][1]
            if (lines[i][3] < min_value):
                min_index = i
                min_value = lines[i][3]
            if(lines[i][3] > max_value):
                max_index = i
                max_value = lines[i][3]
            if (lines[i][1] > max_value):
                max_index = i
                max_value = lines[i][1]
        return min_index, max_index

    """Avoid overflow"""
    """
        x：数值
        low：下界
        high：上界
        
        Return：溢出检测后的x值
    """
    def crop(self, x, low, high):
        if(x > high):
            x = high
        elif(x < low):
            x = low
        return x

    """Coordinate affine transformation to find ROI region"""
    """
        x, y：原图中点的坐标
        angle：进行仿射变换时需要图片变换的角度，使用的是直线与水平方向夹角
        scale：仿射变换时缩放的尺度
        
        Return：仿射变换后的坐标
    """
    def coordinate_transform(self, x, y, angle, scale):
        x = x - self.image_height * 0.5
        y = y - self.image_width * 0.5
        aff_center_x = round(x * math.cos(angle) * scale + y * math.sin(angle) * scale + self.image_height * 0.5)
        aff_center_x = self.crop(aff_center_x, 0, self.image_height)
        aff_center_y = round(-x * math.sin(angle) * scale + y * math.cos(angle) * scale + self.image_width * 0.5)
        aff_center_y = self.crop(aff_center_y, 0, self.image_width)
        return aff_center_x, aff_center_y

    """Returns the ROI region coordinate points and discards points that are not in the region"""
    """
        min_point：一条线段集合中的最小点[x1, y1, x2, y2]
        max_point：一条线段集合中的最大点[x1, y1, x2, y2]
        point：线段集合中的所有点[[x1, y1, x2, y2],....]
        drop_index：需要丢弃的点下标，用来精检测第三步丢弃点
        
        Rrturn：ROI区域四个坐标点(左上, 左下, 右上, 右下)
    """
    def get_ROI_drop(self, min_point, max_point, point):
        drop_index = []
        k = (max_point[3] - min_point[1]) / (max_point[2] - min_point[0])
        center_x = (max_point[2] - min_point[0]) / 2
        center_y = (max_point[3] - min_point[1]) / 2
        angle = math.atan(k)
        aff_center_x, aff_center_y = self.coordinate_transform(center_x, center_y, -angle, 0.5)
        aff_ROI_top_left_x = aff_center_x - 2.5
        aff_ROI_top_left_y = aff_center_y - 25
        aff_ROI_down_left_x = aff_center_x + 2.5
        aff_ROI_down_left_y = aff_center_y - 25
        aff_ROI_top_right_x = aff_center_x - 2.5
        aff_ROI_top_right_y = aff_center_y + 25
        aff_ROI_down_right_x = aff_center_x + 2.5
        aff_ROI_down_right_y = aff_center_y + 25
        print(aff_ROI_top_left_x, aff_ROI_top_left_y, aff_ROI_down_left_x, aff_ROI_down_left_y, aff_ROI_top_right_x, aff_ROI_top_right_y, aff_ROI_down_right_x, aff_ROI_down_right_y)
        for i in range(len(point)):
            aff_start_point_x, aff_start_point_y = self.coordinate_transform(point[i][0], point[i][1], -angle, 0.5)
            aff_end_point_x, aff_end_point_y = self.coordinate_transform(point[i][2], point[i][3], -angle, 0.5)
            if((aff_start_point_x < aff_ROI_top_left_x) or (aff_start_point_x > aff_ROI_down_left_x)):
                drop_index.append(i)
            elif((aff_start_point_y < aff_ROI_top_left_y) or (aff_start_point_y > aff_ROI_top_right_y)):
                drop_index.append(i)
            if ((aff_end_point_x < aff_ROI_top_left_x) or (aff_end_point_x > aff_ROI_down_left_x)):
                drop_index.append(i)
            elif ((aff_end_point_y < aff_ROI_top_left_y) or (aff_end_point_y > aff_ROI_top_right_y)):
                drop_index.append(i)
        ROI_top_left_x, ROI_top_left_y = self.coordinate_transform(aff_ROI_top_left_x, aff_ROI_top_left_y, angle, 2)
        ROI_down_left_x, ROI_down_left_y = self.coordinate_transform(aff_ROI_down_left_x, aff_ROI_down_left_y, angle, 2)
        ROI_top_right_x, ROI_top_right_y = self.coordinate_transform(aff_ROI_top_right_x, aff_ROI_top_right_y, angle, 2)
        ROI_down_right_x, ROI_down_right_y = self.coordinate_transform(aff_ROI_down_right_x, aff_ROI_down_right_y, angle, 2)
        print(ROI_top_left_x, ROI_top_left_y, ROI_down_left_x, ROI_down_left_y, ROI_top_right_x, ROI_top_right_y, ROI_down_right_x, ROI_down_right_y)
        return [ROI_top_left_x, ROI_top_left_y, ROI_down_left_x, ROI_down_left_y, ROI_top_right_x, ROI_top_right_y, ROI_down_right_x, ROI_down_right_y], drop_index

    """粗检测"""
    def Coarse_Detection(self):
        print("start coarse detection")
        dst_image = numpy.zeros(self.image_info)
        for i in range(3, self.image_height - 3):
            for j in range(3, self.image_width - 3):
                flag = True
                Lc = self.src_image[i, j]
                Ll_0 = self.src_image[i - 3, j]
                Lr_0 = self.src_image[i + 3, j]
                flag = self.judge(Lc, Ll_0, Lr_0)
                if(flag):
                    Ll_45 = self.src_image[i - 3, j - 3]
                    Lr_45 = self.src_image[i + 3, j + 3]
                    flag = self.judge(Lc, Ll_45, Lr_45)
                if (flag):
                    Ll_90 = self.src_image[i, j - 3]
                    Lr_90 = self.src_image[i, j + 3]
                    flag = self.judge(Lc, Ll_90, Lr_90)
                if (flag):
                    Ll_135 = self.src_image[i + 3, j - 3]
                    Lr_135 = self.src_image[i - 3, j + 3]
                    flag = self.judge(Lc, Ll_135, Lr_135)
                if(flag == False):
                    dst_image[i, j] = 255
        return dst_image

    """精检测"""
    def Fine_Detection(self, binary_image_path):
        print("start fine detection")
        binary_image = cv2.imread(binary_image_path, 0)
        dst_image = numpy.zeros(binary_image.shape)
        """step1: LSD检测并合并线段，获得同类线段集合"""
        print("step 1")
        ls = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD, _scale=0.8, _sigma_scale=0.6, _quant=2.0, _ang_th=22.5, _density_th=0.7, _n_bins=1024)
        line_result, _, _, _ = ls.detect(binary_image)
        merge_result, group_result = [], []
        line_result = line_result.reshape((-1, 4))
        line_result = line_result.tolist()
        line_result = list(line_result)
        for i in range(len(line_result)):
            self.sort_line(line_result[i])
            cv2.line(dst_image, (int(line_result[i][0]), int(line_result[i][1])), (int(line_result[i][2]), int(line_result[i][3])), (255, 255, 255))
        # dst_image = cv2.resize(dst_image, (500, 500), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("dst image", dst_image)
        # cv2.waitKey(1000)
        index = numpy.zeros((len(line_result)))
        q = 1
        while(True):
            for i in range(len(line_result)):
                if(index[i] == 0):
                    group_result.append(line_result[i])
                    index[i] = q
                    break
            while(True):
                end_flag = True
                for i in range(len(line_result)):
                    for j in range(len(group_result)):
                        flag = self.merge(group_result[j], line_result[i])
                        if(flag and (index[i] == 0)):
                            index[i] = q
                            group_result.append(line_result[i])
                            end_flag = False
                if(end_flag):
                    merge_result.append(copy.deepcopy(group_result))
                    group_result.clear()
                    q += 1
                    break
            if(index.min() != 0):
                break
        # print(merge_result)
        """step2：根据第一步的结果设置ROI区域并将区域外点和线段丢弃"""
        print("step 2")
        # ROI_results, drop_indexs = [], []
        # for i in range(len(merge_result)):
        #     min_index, max_index = self.get_minmax_line(merge_result[i])
        #     line = copy.deepcopy(merge_result[i])
        #     ROI_result, drop_index = self.get_ROI_drop(merge_result[i][min_index], merge_result[i][max_index], line)
        #     ROI_results.append(ROI_result)
        #     drop_indexs.append(copy.deepcopy(drop_index))
        # # print(merge_result)
        # # print(ROI_results)
        # print(drop_indexs)
        # for i in range(len(merge_result)):
        #     for j in range(len(drop_indexs[i])):
        #         _ = merge_result[i].pop[drop_indexs[i][j]]
        # points = []
        # for i in range(len(merge_result)):
        #     point = []
        #     for j in range(len(merge_result[i])):
        #         point.append([merge_result[i][j][0], merge_result[i][j][1]])
        #         point.append([merge_result[i][j][2], merge_result[i][j][3]])
        #     points.append(point)
        # points = numpy.array(points)
        """step3：根据第二步结果，使用RANSAC拟合直线，将最相关的点找出"""
        points = []
        for i in range(len(merge_result)):
            point = []
            for j in range(len(merge_result[i])):
                point.append(numpy.array([merge_result[i][j][0], merge_result[i][j][1]]))
                point.append(numpy.array([merge_result[i][j][2], merge_result[i][j][3]]))
            points.append(numpy.array(point))
        points = numpy.array(points)
        print(points)
        print("step 3")
        n_input, n_output = 1, 1
        input_columns, output_columns = range(n_input), [n_input + i for i in range(n_output)]
        model = LinearLeastSquareModel(input_columns, output_columns, debug=False)
        RANSAC_data = []
        for i in range(len(points)):
            # n = int(len(points) * 0.2)
            # d = int(len(points) * 0.5)
            if(len(points[i]) < 10):
                continue
            ransac_fit, ransac_data = ransac(points[i], model, 5, 1000, 7e3, 10, debug=False, return_all=True)
            data = numpy.array(ransac_data['inliers'])
            for j in range(data.shape[0] - 1):
                cv2.line(dst_image, (int(points[i][data[j]][0]), int(points[i][data[j]][1])), (int(points[i][data[j + 1]][0]), int(points[i][data[j + 1]][1])), (255, 0, 0))
        cv2.imshow("step 3", dst_image)
        cv2.waitKey(5000)
        cv2.imwrite("step3_image.bmp", dst_image)
        """step4：根据第三步结果，对相关点得到的二值化图像进行概率霍夫变换进行直线检测"""
        print("step 4")
        image = cv2.imread("step3_image.bmp", 0)
        lines = cv2.HoughLinesP(image, 1, numpy.pi/180, 160, maxLineGap=100)
        print(lines)
        return lines
