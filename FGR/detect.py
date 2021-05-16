# import mayavi.mlab
import traceback
import argparse
from sklearn.cluster import KMeans
from multiprocessing import Pool
import math as m

from utils.kitti_utils_official import *
from utils.visual import *
from utils import iou_3d_python
import pickle
import yaml
from easydict import EasyDict


def Find_2d_box(maximum, KeyPoint_3d, box_2d, p2, corner, detect_config, 
                total_pc=None, cluster=None, truncate=False, sample_points=None):

    # ignore less point cases
    if len(KeyPoint_3d) < 10:
        return None, None, None, None, None, None, None, None
    
    img = np.zeros((700, 700, 3), 'f4')
    KeyPoint = KeyPoint_3d[:, [0, 2]].copy()

    left_point  = np.linalg.inv(p2[:, [0,1,2]]).dot(np.array([box_2d[0], 0, 1]).copy().T)[[0, 2]]
    right_point = np.linalg.inv(p2[:, [0,1,2]]).dot(np.array([box_2d[2], 0, 1]).copy().T)[[0, 2]]

    mat_1 = np.array([[left_point[0], right_point[0]], [left_point[1], right_point[1]]])

    KeyPoint_for_draw = KeyPoint_3d[:, [0, 2]].copy()
    AverageValue_x = np.mean(KeyPoint[:, 0]) * 100
    AverageValue_y = np.mean(KeyPoint[:, 1]) * 100
    
    # start our pipeline
    # 1. find minimum bbox with special consideration: maximize pc number between current bbox
    #    and its 0.8 times bbox with same bbox center and orientation

    current_angle = 0.0
    min_x = 0
    min_y = 0
    max_x = 100
    max_y = 100

    final = None
    seq = np.arange(0, 90.5 * np.pi / 180, 0.5 * np.pi / 180)
    FinalPoint = np.array([0., 0.])

    if maximum:
        cut_times = max(int(len(KeyPoint) * detect_config.CUT_RATE_MAX), 1)
    else:
        cut_times = min(int(len(KeyPoint) * detect_config.CUT_RATE_MIN), 1)

    while True:
        minValue = -1
        for i in seq:
            try:
                RotateMatrix = np.array([[np.cos(i), -1 * np.sin(i)],
                                         [np.sin(i), np.cos(i)]])
                temp = np.dot(KeyPoint, RotateMatrix)
                current_min_x, current_min_y = np.amin(temp, axis=0)
                current_max_x, current_max_y = np.amax(temp, axis=0)

                # construct a sub-rectangle smaller than bounding box, whose x_range and y_range is defined below:
                thresh_min_x = current_min_x + detect_config.RECT_SHRINK_THRESHOLD * (current_max_x - current_min_x)
                thresh_max_x = current_max_x - detect_config.RECT_SHRINK_THRESHOLD * (current_max_x - current_min_x)
                thresh_min_y = current_min_y + detect_config.RECT_SHRINK_THRESHOLD * (current_max_y - current_min_y)
                thresh_max_y = current_max_y - detect_config.RECT_SHRINK_THRESHOLD * (current_max_y - current_min_y)

                thresh_filter_1 = (temp[:, 0] >= thresh_min_x) & (temp[:, 0] <= thresh_max_x)
                thresh_filter_2 = (temp[:, 1] >= thresh_min_y) & (temp[:, 1] <= thresh_max_y)
                thresh_filter = (thresh_filter_1 & thresh_filter_2).astype(np.uint8)

                # calculate satisfying point number between original bbox and shrinked bbox
                CurrentValue = np.sum(thresh_filter) / temp.shape[0]

            except:
                return None, None, None, None, None, None, None, None

            if CurrentValue < minValue or minValue < 0:
                final = temp
                minValue = CurrentValue
                current_angle = i
                min_x = current_min_x
                min_y = current_min_y
                max_x = current_max_x
                max_y = current_max_y

        box = np.array([[min_x, min_y],
                        [min_x, max_y],
                        [max_x, max_y],
                        [max_x, min_y]])  # rotate clockwise

        angle = current_angle

        # calculate satisfying bounding box
        box = np.dot(box, np.array([[np.cos(angle), np.sin(angle)],
                                    [-1 * np.sin(angle), np.cos(angle)]])).astype(np.float32)

        index_1, index_2, point_1, point_2, number_1, number_2 = find_key_vertex_by_pc_number(KeyPoint, box)
        
        # compare which side has the most points, then determine final diagonal, 
        # final key vertex (Current_FinalPoint) and its index (Current_Index) in bbox
        if number_1 < number_2:
            Current_FinalPoint = point_2
            Current_Index = index_2
        else:
            Current_FinalPoint = point_1
            Current_Index = index_1

        # quitting this loop requires:
        # 1. deleting point process has not stopped (cut_times is not positive)
        # 2. after deleting points, key vertex point's location is almost same as that before deleting points   
        if cut_times == 0 and (Current_FinalPoint[0] - FinalPoint[0]) ** 2 + \
                              (Current_FinalPoint[1] - FinalPoint[1]) ** 2 < detect_config.KEY_VERTEX_MOVE_DIST_THRESH:
            break
        else:
            if cut_times == 0:
                # the end of deleting point process, re-calculate new cut_times with lower number of variable [KeyPoint]
                FinalPoint = Current_FinalPoint
                if maximum:
                    cut_times = max(int(len(KeyPoint) * detect_config.CUT_RATE_MAX_2), 1)
                else:
                    cut_times = min(int(len(KeyPoint) * detect_config.CUT_RATE_MIN), 1)

            else:
                # continue current deleting point process
                cut_times -= 1
                
                # avoid too fierce deleting
                if KeyPoint.shape[0] < detect_config.THRESH_MIN_POINTS_AFTER_DELETING:
                    return None, None, None, None, None, None, None, None
                
                index, KeyPoint, final = delete_noisy_point_cloud(final, Current_Index, KeyPoint, 
                                                                  detect_config.DELETE_TIMES_EVERY_EPOCH)

    # while the loop is broken, the variable [box] is the final selected bbox for car point clouds
    index_1, index_2, point_1, point_2, number_1, number_2 = find_key_vertex_by_pc_number(KeyPoint, box)
    
    # here we get final key-vertex (FinalPoint) and its index in box (FinalIndex)
    if number_1 < number_2:
        FinalPoint = point_2
        FinalIndex = index_2
    else:
        FinalPoint = point_1
        FinalIndex = index_1
        
    # 2. calculate intersection from key-vertex to frustum [vertically]
    # mat_1: rotation matrix from  
    FinalPoint_Weight = np.linalg.inv(mat_1).dot(np.array([FinalPoint[0], FinalPoint[1]]).T)

    top_1 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[0], box_2d[1], 1]).T)
    top_2 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[2], box_2d[1], 1]).T)
    bot_1 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[0], box_2d[3], 1]).T)
    bot_2 = np.linalg.inv(p2[:, [0, 1, 2]]).dot(np.array([box_2d[2], box_2d[3], 1]).T)
    
    if truncate == False:
        y_min, y_max = Calculate_Height(top_1, top_2, bot_1, bot_2, FinalPoint)
    else:
        # for truncate cases, calculating height from frustum may fail if key-vertex is not inside frustum area
        
        y_min = np.min(KeyPoint_3d[:, 1])
        plane = fitPlane(sample_points)
        eps = 1e-8
        sign = np.sign(np.sign(plane[1]) + 0.5)
        try:
            y_max = -1 * (plane[0] * FinalPoint[0] + plane[2] * FinalPoint[1] - 1) / (plane[1] + eps * sign)
        except:
            y_max = np.max(KeyPoint_3d[:, 1])

    # filter cars with very bad height
    if np.abs(y_max - y_min) < detect_config.MIN_HEIGHT_NORMAL or \
       np.abs(y_max - y_min) > detect_config.MAX_HEIGHT_NORMAL or \
       (truncate == True and (y_max < detect_config.MIN_TOP_TRUNCATE or 
                              y_max > detect_config.MAX_TOP_TRUNCATE or 
                              y_min < detect_config.MIN_BOT_TRUNCATE or 
                              y_min > detect_config.MAX_BOT_TRUNCATE)):
        
        error_message = "top: %.4f, bottom: %.4f, car height: %.4f, deprecated" % (y_min, y_max, np.abs(y_max - y_min))
        return None, None, None, None, None, None, error_message, 0

    # 3. calculate intersection from key-vertex to frustum [horizontally], to get car's length and width
    if truncate == True or FinalPoint_Weight[0] < detect_config.FINAL_POINT_FLIP_THRESH or \
                           FinalPoint_Weight[1] < detect_config.FINAL_POINT_FLIP_THRESH:
        loc1 = box[FinalIndex - 1]
        loc2 = box[(FinalIndex + 1) % 4]
        loc3 = np.array([0., 0.])
        loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
        loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]
    else:
        loc1, loc2, loc3, angle_1, angle_2 = Find_Intersection_Point(box=box, right_point=right_point,
                                                                        left_point=left_point,
                                                                        FinalIndex=FinalIndex, FinalPoint=FinalPoint,
                                                                        shape=KeyPoint.shape[0])
        
        weight = np.linalg.inv(mat_1).dot(np.array([loc3[0], loc3[1]]).T)
        
        # correct some cases with failed checking on key-vertex (very close to frustum's left/right side)
        if weight[0] <= detect_config.FINAL_POINT_FLIP_THRESH or weight[1] <= detect_config.FINAL_POINT_FLIP_THRESH:
            if FinalIndex == index_1:
                FinalIndex = index_2
                FinalPoint = point_2
            else:
                FinalIndex = index_1
                FinalPoint = point_1
            
            # re-calculate intersection
            loc1, loc2, loc3, angle_1, angle_2 = Find_Intersection_Point(box=box, right_point=right_point,
                                                                            left_point=left_point,
                                                                            FinalIndex=FinalIndex, FinalPoint=FinalPoint,
                                                                            shape=KeyPoint.shape[0])

        # if the angle between bounding box and frustum radiation lines is smaller than detect_config.ANCHOR_FIT_DEGREE_THRESH,
        # ignore the intersection strategy, and use anchor box 
        # (with pre-defined length-width rate, which is medium value in total KITTI dataset)
        
        loc1, loc2, loc3 = check_anchor_fitting(box, loc1, loc2, loc3, angle_1, angle_2, 
                                                FinalIndex, FinalPoint, y_max, y_min,
                                                anchor_fit_degree_thresh=detect_config.ANCHOR_FIT_DEGREE_THRESH, 
                                                height_width_rate=detect_config.HEIGHT_WIDTH_RATE, 
                                                height_length_rate=detect_config.HEIGHT_LENGTH_RATE,
                                                length_width_boundary=detect_config.LENGTH_WIDTH_BOUNDARY)
        
    # 4. filter cases with still bad key-vertex definition,
    # we assume that key-vertex must be in the top 2 nearest to camera along z axis
    
    z_less_than_finalpoint = 0
    for i in range(len(box)):
        if i == FinalIndex:
            continue
        if box[i, 1] < box[FinalIndex, 1]:
            z_less_than_finalpoint += 1
    
    if z_less_than_finalpoint >= 2:
        error_message = "keypoint error, deprecated."
        return None, None, None, None, None, None, error_message, 2

    len_1 = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
    len_2 = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)

    car_length = max(len_1, len_2)
    car_width  = min(len_1, len_2)

    # define max(len_1, len_2) as length of the car, and min(len_1, len_2) as width of the car
    # length of the car is 3.0-5.0m, and width of the car is 1.2-2.2m
    # filter cars with very bad length or height
    
    if not (detect_config.MIN_WIDTH  <= car_width  <= detect_config.MAX_WIDTH) or \
       not (detect_config.MIN_LENGTH <= car_length <= detect_config.MAX_LENGTH):
        error_message = "length: %.4f, width: %.4f, deprecated" % (car_length, car_width)
        return None, None, None, None, None, None, error_message, 1

    KeyPoint_side = KeyPoint_3d[:, [0, 1]].copy()
    img_side = np.zeros((700, 700, 3), 'f4')
    
    # draw gt bounding box from 3D to 2D plane
    img = draw_bbox_3d_to_2d_gt(img, corner, AverageValue_x, AverageValue_y)

    # draw frustum's left and right line in 2D plane
    img = draw_frustum_lr_line(img, left_point, right_point, AverageValue_x, AverageValue_y)
    
    # draw psuedo bounding box from 3D to 2D plane [before] calculating intersection
    img = draw_bbox_3d_to_2d_psuedo_no_intersection(img, box, AverageValue_x, AverageValue_y)

    # draw psuedo bounding box from 3D to 2D plane [after] calculating intersection
    img = draw_bbox_3d_to_2d_psuedo_with_key_vertex(img, FinalPoint, loc1, loc2, loc3, AverageValue_x, AverageValue_y)
    
    # draw car point clouds after region growth
    img = draw_point_clouds(img, KeyPoint_for_draw, AverageValue_x, AverageValue_y)
    
    return img, img_side, FinalPoint, loc1, loc3, loc2, y_max, y_min


def delete_noisy_point_cloud(final, Current_Index, KeyPoint, delete_times_every_epoch=2):
                
    # re-calculate key-vertex's location
    # KeyPoint: original point cloud
    # final: [rotated] point cloud
    # deleting method: from KeyPoint, calculate the point with maximum/minimum x and y,
    # extract their indexes, and delete them from numpy.array
    # one basic assumption on box's location order is: 0 to 3 => left-bottom to left_top (counter-clockwise)
    if Current_Index == 2 or Current_Index == 3:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.max(final[:, 0], axis=0))
            KeyPoint = np.delete(KeyPoint, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)

    if Current_Index == 0 or Current_Index == 1:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.min(final[:, 0], axis=0))
            KeyPoint = np.delete(KeyPoint, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)

    if Current_Index == 1 or Current_Index == 2:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.max(final[:, 1], axis=0))
            KeyPoint = np.delete(KeyPoint, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)

    if Current_Index == 0 or Current_Index == 3:
        for _ in range(delete_times_every_epoch):
            index = np.where(final == np.min(final[:, 1], axis=0))
            KeyPoint = np.delete(KeyPoint, index[0][0], axis=0)
            final = np.delete(final, index[0][0], axis=0)
    
    return index, KeyPoint, final


def find_key_vertex_by_pc_number(KeyPoint, box):
    # first diagonal: (box[1], box[3]), corresponding points: box[0] / box[2]
    # (... > 0) is the constraint for key vertex's side towards the diagnoal
    if box[0][0] * (box[1][1] - box[3][1]) - box[0][1] * (box[1][0] - box[3][0]) + (
            box[1][0] * box[3][1] - box[1][1] * box[3][0]) > 0:
        index_1 = 0
    else:
        index_1 = 2

    # first diagonal: (box[1], box[3]), and calculate the point number on one side of this diagonal,
    # (... > 0) to constraint the current side, which is equal to the side of key vertex (box[index_1]) 
    filter_1 = (KeyPoint[:, 0] * (box[1][1] - box[3][1]) - KeyPoint[:, 1] * (box[1][0] - box[3][0]) + (
                box[1][0] * box[3][1] - box[1][1] * box[3][0]) > 0)
    number_1 = np.sum(filter_1)
        
    # find which side contains more points, record this side and corresponding point number, 
    # and key vertex, towards current diagonal (box[1], box[3])
        
    # number_1: most point number
    # index_1:  corresponding key vertex's index of bbox points
    # point_1:  corresponding key vertex
        
    if number_1 < KeyPoint.shape[0] / 2:
        number_1 = KeyPoint.shape[0] - number_1
        index_1 = (index_1 + 2) % 4

    point_1 = box[index_1]

    # second diagonal: (box[0], box[2]), corresponding points: box[1] / box[3]
    # (... > 0) to constraint the current side, which is equal to the side of key vertex (box[index_2]) 
    if box[1][0] * (box[0][1] - box[2][1]) - box[2][1] * (box[0][0] - box[2][0]) + (
            box[0][0] * box[2][1] - box[0][1] * box[2][0]) > 0:
        index_2 = 1
    else:
        index_2 = 3

    # find which side contains more points, record this side and corresponding point number, 
    # and key vertex, towards current diagonal (box[0], box[2])
        
    # number_2: most point number
    # index_2:  corresponding key vertex's index of bbox points
    # point_2:  corresponding key vertex
        
    filter_2 = (KeyPoint[:, 0] * (box[0][1] - box[2][1]) - KeyPoint[:, 1] * (box[0][0] - box[2][0]) + (
                box[0][0] * box[2][1] - box[0][1] * box[2][0]) > 0)
    number_2 = np.sum(filter_2)

    if number_2 < KeyPoint.shape[0] / 2:
        number_2 = KeyPoint.shape[0] - number_2
        index_2 = (index_2 + 2) % 4

    point_2 = box[index_2]
    
    return index_1, index_2, point_1, point_2, number_1, number_2


def check_anchor_fitting(box, loc1, loc2, loc3, angle_1, angle_2, FinalIndex, FinalPoint, y_max, y_min,
                         anchor_fit_degree_thresh=10,
                         height_width_rate=0.9305644265920366,
                         height_length_rate=0.3969212090597959, 
                         length_width_boundary=2.2):
        
    if loc1[0] == box[FinalIndex - 1][0] or angle_1 * 180 / np.pi < anchor_fit_degree_thresh or (
            loc1[0] - FinalPoint[0] > 0 and box[FinalIndex - 1][0] - FinalPoint[0] < 0) or \
            (loc1[0] - FinalPoint[0] < 0 and box[FinalIndex - 1][0] - FinalPoint[0] > 0):
        current_distance = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)
            
        # if current_distance is larger than 2.2, we assume current boundary is length, otherwise width,
        # then use length-width rate to calculate another boundary
        if current_distance > length_width_boundary:
            current_distance = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
            # ... np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2) * width_length_rate ....
            loc1[0] = FinalPoint[0] + (loc1[0] - FinalPoint[0]) * np.abs(
                y_max - y_min) / height_width_rate / current_distance
            loc1[1] = FinalPoint[1] + (loc1[1] - FinalPoint[1]) * np.abs(
                y_max - y_min) / height_width_rate / current_distance
        else:
            current_distance = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
            loc1[0] = FinalPoint[0] + (loc1[0] - FinalPoint[0]) * np.abs(
                y_max - y_min) / height_length_rate / current_distance
            loc1[1] = FinalPoint[1] + (loc1[1] - FinalPoint[1]) * np.abs(
                y_max - y_min) / height_length_rate / current_distance

        loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
        loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]

    # check another boundary radiated from key vertex
    elif loc2[0] == box[(FinalIndex + 1) % 4][0] or angle_2 * 180 / np.pi < anchor_fit_degree_thresh or (
            loc2[0] - FinalPoint[0] > 0 and box[(FinalIndex + 1) % 4][0] - FinalPoint[0] < 0) or \
            (loc2[0] - FinalPoint[0] < 0 and box[(FinalIndex + 1) % 4][0] - FinalPoint[0] > 0):
        current_distance = np.sqrt((loc1[0] - FinalPoint[0]) ** 2 + (loc1[1] - FinalPoint[1]) ** 2)
        if current_distance > length_width_boundary:
            current_distance = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)
            loc2[0] = FinalPoint[0] + (loc2[0] - FinalPoint[0]) * np.abs(
                y_max - y_min) / height_width_rate / current_distance
            loc2[1] = FinalPoint[1] + (loc2[1] - FinalPoint[1]) * np.abs(
                y_max - y_min) / height_width_rate / current_distance
        else:
            current_distance = np.sqrt((loc2[0] - FinalPoint[0]) ** 2 + (loc2[1] - FinalPoint[1]) ** 2)
            loc2[0] = FinalPoint[0] + (loc2[0] - FinalPoint[0]) * np.abs(
                y_max - y_min) / height_length_rate / current_distance
            loc2[1] = FinalPoint[1] + (loc2[1] - FinalPoint[1]) * np.abs(
                y_max - y_min) / height_length_rate / current_distance

        loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
        loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]
    
    return loc1, loc2, loc3

def iou_3d(box_3d, loc0, loc1, loc2, loc3, y_min, y_max):

    # use official code: iou_3d_python.py to calculate 3d iou
    std_box_3d = np.array([[loc1[0], y_max, loc1[1]],
                           [loc0[0], y_max, loc0[1]],
                           [loc3[0], y_max, loc3[1]],
                           [loc2[0], y_max, loc2[1]],
                           [loc1[0], y_min, loc1[1]],
                           [loc0[0], y_min, loc0[1]],
                           [loc3[0], y_min, loc3[1]],
                           [loc2[0], y_min, loc2[1]]])
    std_iou, iou_2d = iou_3d_python.box3d_iou(box_3d, std_box_3d)
    return None, std_iou

def Calculate_Height(top_1, top_2, bot_1, bot_2, keypoint):

    # calculate the [vertical] height in frustum at key vertex (input variable [keypoint])

    # because top and bottom plane of frustum crosses (0, 0, 0), we assume the plane equation: Ax + By + 1 * z = 0
    # |x1 y1| |A|     |-1|         |A|     |x1 y1| -1    |-1|
    # |     | | |  =  |  |     =>  | |  =  |     |    *  |  |
    # |x2 y2| |B|     |-1|         |B|     |x2 y2|       |-1|

    mat_1 = np.array([[top_1[0], top_1[1]], [top_2[0], top_2[1]]])
    mat_2 = np.array([[bot_1[0], bot_1[1]], [bot_2[0], bot_2[1]]])
    mat_3 = np.array([-1., -1.]).T

    top_plane_info = np.linalg.inv(mat_1).dot(mat_3)
    bot_plane_info = np.linalg.inv(mat_2).dot(mat_3)

    top_y = -1 * (keypoint[0] * top_plane_info[0] + keypoint[1] * 1) / top_plane_info[1]
    bot_y = -1 * (keypoint[0] * bot_plane_info[0] + keypoint[1] * 1) / bot_plane_info[1]

    return top_y, bot_y

def Find_Intersection_Point(box, FinalIndex, right_point, left_point, FinalPoint, shape):

    # calculate the [expanded] bounding box from input variable [box], 
    # with intersection line radiated from key vertex (FinalPoint)
    # calculate two line's intersection point by line function:
    # y1 = k * x1 + b
    # y2 = k * x2 + b, 
    # solve these equations

    equation_1_left = np.array(
        [[box[FinalIndex - 1][1] - box[FinalIndex][1], box[FinalIndex][0] - box[FinalIndex - 1][0]],
         [left_point[1], -1 * left_point[0]]])
    equation_1_right = np.array(
        [box[FinalIndex][0] * box[FinalIndex - 1][1] - box[FinalIndex - 1][0] * box[FinalIndex][1], 0])

    try:
        loc1 = np.linalg.inv(equation_1_left).dot(equation_1_right.T)
    except:
        # if there are two parallel lines, np.linalg.inv will fail, so deprecate this case
        return None, None, None

    # determine how to intersect
    # the line radiated from key vertex may cross left frustum and right frustum at the same time, causing two intersection points,
    # so just check which intersection point is right
    # solve this matter still by line equations
    if (loc1[0] - FinalPoint[0]) * (box[FinalIndex - 1][0] - FinalPoint[0]) + \
            (loc1[1] - FinalPoint[1]) * (box[FinalIndex - 1][1] - FinalPoint[1]) > 0:
        vector_1 = np.array([loc1[0] - FinalPoint[0], loc1[1] - FinalPoint[1]])
        angle_1 = np.abs(np.arcsin((left_point[0] * vector_1[1] - left_point[1] * vector_1[0]) / 
                                   (np.linalg.norm(vector_1) * np.linalg.norm(left_point))))
    else:
        equation_2_left = np.array(
            [[box[FinalIndex - 1][1] - box[FinalIndex][1], box[FinalIndex][0] - box[FinalIndex - 1][0]],
             [right_point[1], -1 * right_point[0]]])
        equation_2_right = np.array(
            [box[FinalIndex][0] * box[FinalIndex - 1][1] - box[FinalIndex - 1][0] * box[FinalIndex][1], 0])

        loc1 = np.linalg.inv(equation_2_left).dot(equation_2_right.T)
        vector_1 = np.array([loc1[0] - FinalPoint[0], loc1[1] - FinalPoint[1]])
        angle_1 = np.abs(np.arcsin((right_point[0] * vector_1[1] - right_point[1] * vector_1[0]) / (
                    np.linalg.norm(vector_1) * np.linalg.norm(right_point))))

        if (loc1[0] - FinalPoint[0]) * (box[FinalIndex - 1][0] - FinalPoint[0]) + \
           (loc1[1] - FinalPoint[1]) * (box[FinalIndex - 1][1] - FinalPoint[1]) < 0:
            loc1 = box[FinalIndex - 1].copy()

    equation_1_left = np.array(
        [[box[(FinalIndex + 1) % 4][1] - box[FinalIndex][1], box[FinalIndex][0] - box[(FinalIndex + 1) % 4][0]],
         [right_point[1], -1 * right_point[0]]])
    equation_1_right = np.array(
        [box[FinalIndex][0] * box[(FinalIndex + 1) % 4][1] - box[(FinalIndex + 1) % 4][0] * box[FinalIndex][1], 0])

    loc2 = np.linalg.inv(equation_1_left).dot(equation_1_right.T)

    if (loc2[0] - FinalPoint[0]) * (box[(FinalIndex + 1) % 4][0] - FinalPoint[0]) + \
       (loc2[1] - FinalPoint[1]) * (box[(FinalIndex + 1) % 4][1] - FinalPoint[1]) > 0:
        vector_2 = np.array([loc2[0] - FinalPoint[0], loc2[1] - FinalPoint[1]])
        angle_2 = np.abs(np.arcsin((right_point[0] * vector_2[1] - right_point[1] * vector_2[0]) / (
                np.linalg.norm(vector_2) * np.linalg.norm(right_point))))
    else:
        equation_2_left = np.array(
            [[box[(FinalIndex + 1) % 4][1] - box[FinalIndex][1], box[FinalIndex][0] - box[(FinalIndex + 1) % 4][0]],
             [left_point[1], -1 * left_point[0]]])
        equation_2_right = np.array(
            [box[FinalIndex][0] * box[(FinalIndex + 1) % 4][1] - box[(FinalIndex + 1) % 4][0] * box[FinalIndex][1], 0])

        loc2 = np.linalg.inv(equation_2_left).dot(equation_2_right.T)
        vector_2 = np.array([loc2[0] - FinalPoint[0], loc2[1] - FinalPoint[1]])
        angle_2 = np.abs(np.arcsin((left_point[0] * vector_2[1] - left_point[1] * vector_2[0]) / (
                np.linalg.norm(vector_2) * np.linalg.norm(left_point))))

        if (loc2[0] - FinalPoint[0]) * (box[(FinalIndex + 1) % 4][0] - FinalPoint[0]) + \
                (loc2[1] - FinalPoint[1]) * (box[(FinalIndex + 1) % 4][1] - FinalPoint[1]) < 0:
            loc2 = box[(FinalIndex + 1) % 4].copy()


    # infer the last point location (loc3) from other 3 points (loc1, loc2, FinalPoint)
    loc3 = np.array([0., 0.])
    loc3[0] = loc1[0] - FinalPoint[0] + loc2[0]
    loc3[1] = loc1[1] - FinalPoint[1] + loc2[1]

    return loc1, loc2, loc3, angle_1, angle_2


def generate_result(base, pickle_save_path, image_save_path, label_save_path, detect_config, save_det_image):

    try:
        seq = base

        if not os.path.exists(os.path.join(pickle_save_path, base + '.pickle')):
            assert False, "no such file: %s" % (os.path.join(pickle_save_path, base + '.pickle'))

        with open(os.path.join(label_save_path, "%s.txt" % base), 'w') as fp:
            pass

        # read pickle file as a dict
        with open(os.path.join(pickle_save_path, base + '.pickle'), 'rb') as fp:
            dic = pickle.load(fp)

        calib = dic['calib']
        objects = {k: dic['sample'][k]['object'] \
                    for k in dic['sample'].keys()}


        if len(list(objects.keys())) == 0:
            print("%s: no valid cars" % seq)
            return

        iou_collection = []
        total_object_number = 0
        for i in objects.keys():
            pc = dic['sample'][i]['pc']

            # ignore bad region grow result (with too many points), which may lead to process stuck in deleting points
            if len(pc) > 4000:
                continue

            # for standard data, y_min and y_max is a float number; 
            # while meeting bugs, y_max is an error message while y_min is error code
            img_down, img_side, loc0, loc1, loc2, loc3, y_max, y_min = Find_2d_box(True, pc, objects[i].boxes[0].box,
                                                                                   calib.p2, objects[i].corners, detect_config,
                                                                                   truncate=dic['sample'][i]['truncate'],
                                                                                   sample_points=dic['ground_sample'])

            if img_down is None:
                img_down, img_side, loc0, loc1, loc2, loc3, y_max, y_min = Find_2d_box(False, pc,
                                                                                       objects[i].boxes[0].box,
                                                                                       calib.p2, objects[i].corners, detect_config,
                                                                                       truncate=dic['sample'][i]['truncate'],
                                                                                       sample_points=dic['ground_sample'])
                if img_down is None:
                    continue

            if loc2 is not None:
                _, std_iou = iou_3d(objects[i].corners, loc0, loc1, loc2, loc3, y_max=y_max, y_min=y_min)
                iou_collection.append(std_iou)

                if save_det_image == True:
                    cv2.imwrite("{:s}/{:.4f}_picture_{:s}_object_{:d}.png".format(image_save_path, std_iou, seq, i), img_down)

                std_box_3d = np.array([[loc0[0], y_max, loc0[1]],
                                        [loc1[0], y_max, loc1[1]],
                                        [loc2[0], y_max, loc2[1]],
                                        [loc3[0], y_max, loc3[1]],
                                        [loc0[0], y_min, loc0[1]],
                                        [loc1[0], y_min, loc1[1]],
                                        [loc2[0], y_min, loc2[1]],
                                        [loc3[0], y_min, loc3[1]]])
                CurrentObject = KittiObject()
                CurrentObject.truncate = objects[i].truncate
                CurrentObject.occlusion = objects[i].occlusion
                CurrentObject.corners = std_box_3d
                CurrentObject.boxes = objects[i].boxes
                CurrentObject.orientation = objects[i].orientation
                write_result_to_label(seq, i, CurrentObject, calib, label_save_path)
                total_object_number += 1
            else:
                continue

        print(base)
        
    except:
        print(traceback.format_exc())
        
        
def write_result_to_label(picture_number, object_number, single_object, calib, label_save_path):
    
    # write detected box information to txt, same format as KITTI
    
    """
    :param picture_number: e.g. 000123
    :param object_number: seems to have no effect...
    :param single_object: an example of KittiObject class
    :param calib: record location transformed from cam0 to cam2
    :return: No return value
    """

    bias_x = calib.t_cam2_cam0[0]
    center_x = 0.5 * (single_object.corners[0][0] + single_object.corners[2][0])
    center_y = single_object.corners[0][1]
    center_z = 0.5 * (single_object.corners[0][2] + single_object.corners[2][2])

    single_object.orientation -= 0.5 * np.pi

    length = np.sqrt((single_object.corners[0][0] - single_object.corners[1][0]) ** 2 +
                     (single_object.corners[0][2] - single_object.corners[1][2]) ** 2)
    width  = np.sqrt((single_object.corners[2][0] - single_object.corners[1][0]) ** 2 +
                     (single_object.corners[2][2] - single_object.corners[1][2]) ** 2)
    height = np.abs(single_object.corners[0][1] - single_object.corners[4][1])
    orientation = m.atan2(-1 * (single_object.corners[0][2] - single_object.corners[1][2]),
                           1 * (single_object.corners[0][0] - single_object.corners[1][0]))

    if length < width:
        length, width = width, length
        orientation = m.atan2(-1 * (single_object.corners[2][2] - single_object.corners[1][2]),
                               1 * (single_object.corners[2][0] - single_object.corners[1][0]))

    if orientation > 0:
        angle = -1 * np.pi
    else:
        angle = np.pi

    if np.abs(orientation - single_object.orientation) > np.abs(orientation + angle - single_object.orientation):
        orientation += angle

    alpha = orientation + 0.5 * np.pi - m.atan2(center_z, -1 * center_x)

    label = "Car "
    label += "%.2f %d " % (single_object.truncate, single_object.occlusion)
    label += "%.2f " % alpha
    label += "%.2f %.2f %.2f %.2f " % (single_object.boxes[0].box[0], single_object.boxes[0].box[1],
                              single_object.boxes[0].box[2], single_object.boxes[0].box[3])
    label += "%.2f %.2f %.2f " % (height, width, length)
    label += "%.2f %.2f %.2f " % (center_x - bias_x, center_y, center_z)
    label += "%.2f" % orientation
    
    with open("%s/%s.txt" % (label_save_path, picture_number), "a+") as fp:
        fp.write(label + '\n')
        

def merge_validation_to_labels(valid_split_path, det_label_save_path, gt_label_save_path):
    with open(args.valid_split_path, 'r') as fp:
        valid_index_collection = fp.readlines()

    for valid_idx in valid_index_collection:
        valid_idx = valid_idx.strip()
        assert len(valid_idx) == 6
        
        shutil.copy(os.path.join(gt_label_save_path,  '%s.txt' % valid_idx), 
                    os.path.join(det_label_save_path, '%s.txt' % valid_idx))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--process', 
        default=16, 
        help="number of parallel process"
    )
    parser.add_argument(
        '--kitti_dataset_path', 
        default='/mnt1/yiwei/kitti_dataset', 
        help="path to store kitti dataset"
    )
    parser.add_argument(
        '--pickle_save_path', 
        default='./region_grow_result', 
        help="path to store region growth files"
    )
    parser.add_argument(
        '--final_save_path', 
        default='./detection_result', 
        help="path to save detection results"
    )
    parser.add_argument(
        '--train_split_path',
        default='../split/train.txt', 
        help="path to save train-split file numbers"
    )
    parser.add_argument(
        '--valid_split_path', 
        default='../split/val.txt', 
        help="path to save validation-split file numbers"
    )
    parser.add_argument(
        '--save_det_image', 
        action='store_true', 
        help='whether to store visualized detection result with iou'
    )
    parser.add_argument(
        '--not_merge_valid_labels', 
        action='store_true', 
        help='merge gt labels to detection labels for later training'
    )
    parser.add_argument(
        '--detect_config_path',
        type=str,
        default='../configs/config.yaml',
        help='pre-defined parameters for running detect.py'
    )
    
    args = parser.parse_args()

    assert os.path.isdir(args.kitti_dataset_path), \
          "Can't find kitti dataset along this path: %s ! Please modify ArgumentParser or check your kitti dataset." % args.kitti_dataset_path
    
    assert os.path.isdir(args.pickle_save_path), \
          "Can't find region grow result along this path: %s ! Please create region growth result and set to this path." \
                % args.pickle_save_path
    
    assert os.path.isfile(args.train_split_path), \
          "Can't find training-split text file path: %s ! Please copy one to current path." % args.train_split_path
    
    assert os.path.isfile(args.detect_config_path), \
          "Can't find [necessary] config file for running detect.py in: %s!" % args.detect_config_path
    
    if os.path.isdir(args.final_save_path):
        ans = input("Saved label directory already exists: %s , Remove it ? [y/n] " % args.final_save_path)
        if ans == 'y' or ans == "Y" or ans == "yes" or ans == "Yes" or ans == "True" or ans == "1":
            os.system('rm -rf %s' % args.final_save_path)
        else:
            exit(0)
    
    os.system('rm -rf %s' % args.final_save_path)
    
    os.makedirs(args.final_save_path, exist_ok=False)
    
    if args.save_det_image:
        image_save_path = os.path.join(args.final_save_path, 'images')
        os.makedirs(image_save_path, exist_ok=False)
    else:
        image_save_path = None
        
    label_save_path = os.path.join(args.final_save_path, 'labels')
    os.makedirs(label_save_path, exist_ok=False)
    
    # load config file
    with open(args.detect_config_path, 'r') as fp:
        detect_config = yaml.load(fp, Loader=yaml.FullLoader)
        detect_config = EasyDict(detect_config)

    # multiprocessing
    start = time.time()
    pool = Pool(processes = int(args.process))
    
    with open(args.train_split_path, 'r') as fp:
        train_index_collection = fp.readlines()

    for train_idx in train_index_collection:
        train_idx = train_idx.strip()
           
        assert len(train_idx) == 6
        pool.apply_async(generate_result, (train_idx, args.pickle_save_path, image_save_path, 
                                           label_save_path, detect_config, args.save_det_image))

    pool.close()
    pool.join()
    
    if not args.not_merge_valid_labels:
        assert os.path.isfile(args.valid_split_path), \
            "Can't find validation split text file in: %s, since you use --merge_valid_labels !" % args.valid_split_path
        
        print("merge validation labels ...")
        gt_label_save_path = os.path.join(args.kitti_dataset_path, 'data_object_label_2', 'training', 'label_2')
        merge_validation_to_labels(args.valid_split_path, label_save_path, gt_label_save_path)

    print('runtime(s):', time.time() - start)
    print('done.')
    exit(0)