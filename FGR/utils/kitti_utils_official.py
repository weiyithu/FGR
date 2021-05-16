import math as m
import numpy as np
import csv
import os
import os.path
import math as mcalcu
import cv2
import json
import shutil
import time

# process all the data in camera 2 frame, the kitti raw data is on camera 0 frame
#------------------------------------------------------ Class define ------------------------------------------------------#
class Box3d:
    def __init__(self, loc0, loc1, loc2, loc3, y_max, y_min):
        self.loc0 = loc0
        self.loc1 = loc1
        self.loc2 = loc2
        self.loc3 = loc3
        self.y_max = y_max
        self.y_min = y_min

class Box2d:
    def __init__(self):
        self.box = []               # left, top, right bottom in 2D image
        self.keypoints = []         # holds the u coordinates of 4 keypoints, -1 denotes the invisible one
        self.visible_left = 0       # The left side is visible (not occluded) by other object
        self.visible_right = 0      # The right side is visible (not occluded) by other object

class KittiObject:
    def __init__(self):
        self.cls = ''               # Car, Van, Truck
        self.truncate = 0           # float 0(non-truncated) - 1(totally truncated)
        self.occlusion = 0          # integer 0, 1, 2, 3
        self.alpha = 0              # viewpoint angle -pi - pi
        self.boxes_origin = (Box2d(), Box2d(), Box2d())
        self.boxes = (Box2d(),
             Box2d(), Box2d())      # Box2d list, default order: box_left, box_right, box_merge
        self.pos = []               # x, y, z in cam2 frame
        self.dim = []               # width(x), height(y), length(z)
        self.orientation = 0        # [-pi - pi]
        self.R = []                 # rotation matrix in cam2 frame
        self.corners = []

class FrameCalibrationData:
    '''Frame Calibration Holder
        p0-p3      Camera P matrix. Contains extrinsic 3x4
                   and intrinsic parameters.
        r0_rect    Rectification matrix, required to transform points 3x3
                   from velodyne to camera coordinate frame.
        tr_velodyne_to_cam0     Used to transform from velodyne to cam 3x4
                                coordinate frame according to:
                                Point_Camera = P_cam * R0_rect *
                                                Tr_velo_to_cam *
                                                Point_Velodyne.
    '''

    def __init__(self):
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.p2_3 = []
        self.r0_rect = []
        self.t_cam2_cam0 = []
        self.tr_velodyne_to_cam0 = []

#------------------------------------------------------ Math operation ------------------------------------------------------#

def region_grow_visual_1(pc, mask_search, mask_origin, thresh):
    pc_search = pc[mask_search == 1]
    mask = mask_origin.copy()
    best_len = 0
    mask_best = np.zeros((pc.shape[0]))
    seed_list_return = []
    mask_list = []
    flag_list = []
    count = 0
    while mask.sum() > 0:
        seed = pc[mask == 1][0]
        seed_list_return.append(seed)
        seed_mask = np.zeros((pc_search.shape[0]))
        seed_mask_all = np.zeros((pc.shape[0]))
        seed_list = [seed]
        flag = 1
        while len(seed_list) > 0:
            temp = seed_list.pop(0)
            dis = np.linalg.norm(pc_search - temp, axis=-1)
            index = np.argmin(dis)
            seed_mask[index] = 1
            valid_mask = (dis < thresh) * (1 - seed_mask) * mask_origin[mask_search == 1]
            seed_list += list(pc_search[valid_mask == 1])
            seed_mask[valid_mask == 1] = 1
            seed_mask_all[mask_search == 1] = seed_mask

        seed_list = [seed]
        while len(seed_list) > 0:
            temp = seed_list.pop(0)
            dis = np.linalg.norm(pc_search - temp, axis=-1)
            index = np.argmin(dis)
            seed_mask[index] = 1
            valid_mask = (dis < thresh) * (1 - seed_mask)
            seed_list += list(pc_search[valid_mask == 1])
            seed_mask[valid_mask == 1] = 1
            seed_mask_all[mask_search == 1] = seed_mask
            if (seed_mask_all * mask_origin).sum() / seed_mask.sum().astype(np.float32) < 0.8:
                flag = 0

        if flag == 1:
            if seed_mask.sum() > best_len:
                best_len = seed_mask.sum()
                mask_best = seed_mask_all
                idx_best = count
        mask_list.append(seed_mask_all)
        flag_list.append(flag)
        mask *= (1 - seed_mask_all)
        count += 1

    return count, mask_list, seed_list_return, flag_list

def region_grow_visual(pc, mask_search, mask_origin, thresh):
    pc_search = pc[mask_search == 1]
    mask = mask_origin.copy()
    best_len = 0
    mask_best = np.zeros((pc.shape[0]))
    mask_list = []
    seed_list_final = []
    flag_list = []
    count = 0
    while mask.sum() > 0:
        seed = pc[mask == 1][0]
        seed_list_final.append(seed)
        seed_mask = np.zeros((pc_search.shape[0]))
        seed_mask_all = np.zeros((pc.shape[0]))
        seed_list = [seed]
        flag = 1
        while len(seed_list) > 0:
            temp = seed_list.pop(0)
            dis = np.linalg.norm(pc_search - temp, axis=-1)
            index = np.argmin(dis)
            seed_mask[index] = 1
            valid_mask = (dis < thresh) * (1 - seed_mask)
            seed_list += list(pc_search[valid_mask == 1])
            seed_mask[valid_mask == 1] = 1
            seed_mask_all[mask_search == 1] = seed_mask
            if (seed_mask_all * mask_origin).sum() / seed_mask.sum().astype(np.float32) < thresh:
                flag = 0
                if (seed_mask_all * mask_origin).sum() / seed_mask.sum().astype(np.float32) < 0.2:
                    break
        if flag == 1:
            if seed_mask.sum() > best_len:
                best_len = seed_mask.sum()
                mask_best = seed_mask_all
                idx_best = count
        mask_list.append(seed_mask_all)
        flag_list.append(flag)
        mask *= (1 - seed_mask_all)
        count += 1

    return count, mask_list, seed_list_final, flag_list

def E2R(Ry, Rx, Rz):
    '''Combine Euler angles to the rotation matrix (right-hand)

        Inputs:
            Ry, Rx, Rz : rotation angles along  y, x, z axis
                         only has Ry in the KITTI dataset
        Returns:
            3 x 3 rotation matrix

    '''
    R_yaw = np.array([[ m.cos(Ry), 0 ,m.sin(Ry)],
                      [ 0,         1 ,     0],
                      [-m.sin(Ry), 0 ,m.cos(Ry)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, m.cos(Rx), -m.sin(Rx)],
                        [0, m.sin(Rx), m.cos(Rx)]])
    #R_roll = np.array([[[m.cos(Rz), -m.sin(Rz), 0],
    #                    [m.sin(Rz), m.cos(Rz), 0],
    #                    [ 0,         0 ,     1]])
    return R_pitch.dot(R_yaw)

def Space2Image(P0, pts3):
    ''' Project a 3D point to the image

        Inputs:
            P0 : Camera intrinsic matrix 3 x 4
            pts3 : 4-d homogeneous coordinates
        Returns:
            image uv coordinates

    '''

    pts2_norm = P0.dot(pts3)

    pts2 = np.array([(pts2_norm[0]/pts2_norm[2]), (pts2_norm[1]/pts2_norm[2])])
    return pts2

def NormalizeVector(P):
    return np.append(P, [1])

#------------------------------------------------------ Data reading ------------------------------------------------------#

def read_obj_calibration(CALIB_PATH):

    '''
        Reads in Calibration file from Kitti Dataset.

        Inputs:
        CALIB_PATH : Str PATH of the calibration file.

        Returns:
        frame_calibration_info : FrameCalibrationData
                                Contains a frame's full calibration data.
        ^ z        ^ z                                      ^ z         ^ z
        | cam2     | cam0                                   | cam3      | cam1
        |-----> x  |-----> x                                |-----> x   |-----> x
    '''

    frame_calibration_info = FrameCalibrationData()

    data_file = open(CALIB_PATH, 'r')

    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]                                  
        p = [float(p[i]) for i in range(len(p))]   
        p = np.reshape(p, (3, 4))                  
                                                   
        p_all.append(p)                            

    frame_calibration_info.p0 = p_all[0]
    frame_calibration_info.p1 = p_all[1]
    frame_calibration_info.p2 = p_all[2]
    frame_calibration_info.p3 = p_all[3]

    frame_calibration_info.p2_2 = np.copy(p_all[2])
    frame_calibration_info.p2_2[0,3] = frame_calibration_info.p2_2[0,3] - frame_calibration_info.p2[0,3]

    frame_calibration_info.p2_3 = np.copy(p_all[3])
    frame_calibration_info.p2_3[0,3] = frame_calibration_info.p2_3[0,3] - frame_calibration_info.p2[0,3]
    frame_calibration_info.t_cam2_cam0 = np.zeros(3)
    frame_calibration_info.t_cam2_cam0[0] = (frame_calibration_info.p2[0,3] - frame_calibration_info.p0[0,3]) / frame_calibration_info.p2[0,0]

    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calibration_info.r0_rect = np.reshape(tr_rect, (3, 3))

    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calibration_info.tr_velodyne_to_cam0 = np.reshape(tr_v2c, (3, 4))

    return frame_calibration_info

def read_obj_data(LABEL_PATH, calib=None, im_shape=None):

    '''
        Reads in object label file from Kitti Object Dataset.

        Inputs:
            LABEL_PATH : Str PATH of the label file.
        Returns:
            List of KittiObject : Contains all the labeled data
    '''

    used_cls = ['Car']#, 'Van' ,'Truck', 'Misc']
    objects = []

    detection_data = open(LABEL_PATH, 'r')
    detections = detection_data.readlines()
    for object_index in range(len(detections)):

        data_str = detections[object_index]
        data_list = data_str.split()

        if data_list[0] not in used_cls:
            continue

        object_it = KittiObject()

        object_it.cls = data_list[0]

        object_it.truncate = float(data_list[1])

        object_it.occlusion = int(data_list[2])

        object_it.alpha = float(data_list[3])

        object_it.dim = np.array([data_list[9], data_list[8], data_list[10]]).astype(float)

        object_it.pos = np.array(data_list[11:14]).astype(float) + calib.t_cam2_cam0

        # The orientation definition is inconsistent with right-hand coordinates in kitti
        object_it.orientation = float(data_list[14]) + m.pi/2

        object_it.R = E2R(object_it.orientation, 0, 0)

        pts3_c_o = []  # 3D location of 3D bounding box corners
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, -object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], 0, object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], 0, -object_it.dim[2]])/2.0)
        
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], -2.0*object_it.dim[1], -object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([-object_it.dim[0], -2.0*object_it.dim[1], object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], -2.0*object_it.dim[1], object_it.dim[2]])/2.0)
        pts3_c_o.append(object_it.pos + object_it.R.dot([object_it.dim[0], -2.0*object_it.dim[1], -object_it.dim[2]])/2.0)
        
        object_it.boxes_origin[0].box = np.array(data_list[4:8]).astype(float)
        object_it.boxes_origin[1].box = np.array(data_list[4:8]).astype(float)
        object_it.boxes_origin[2].box = np.array(data_list[4:8]).astype(float)

        object_it.boxes[0].box = np.array([10000, 10000, 0, 0]).astype(float)
        object_it.boxes[1].box = np.array([10000, 10000, 0, 0]).astype(float)
        object_it.boxes[2].box = np.array([0.0, 0.0, 0.0, 0.0]).astype(float)
        object_it.boxes[0].keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)

        object_it.boxes[1].keypoints = np.array([-1.0, -1.0, -1.0, -1.0]).astype(float)
        object_it.corners = np.array(pts3_c_o)

        for j in range(2): # left and right boxes
            for i in range(8):
                if pts3_c_o[i][2] < 0:
                    continue
                if j == 0:
                    # project 3D corner to left image
                    pt2 = Space2Image(calib.p2_2, NormalizeVector(pts3_c_o[i]))
                elif j == 1:
                    # project 3D corner to right image
                    pt2 = Space2Image(calib.p2_3, NormalizeVector(pts3_c_o[i]))
                if i < 4:
                    object_it.boxes[j].keypoints[i] = pt2[0]

                object_it.boxes[j].box[0] = min(object_it.boxes[j].box[0], pt2[0])
                object_it.boxes[j].box[1] = min(object_it.boxes[j].box[1], pt2[1])
                object_it.boxes[j].box[2] = max(object_it.boxes[j].box[2], pt2[0])
                object_it.boxes[j].box[3] = max(object_it.boxes[j].box[3], pt2[1])

            object_it.boxes[j].box[0] = max(object_it.boxes[j].box[0], 0)
            object_it.boxes[j].box[1] = max(object_it.boxes[j].box[1], 0)

            if im_shape is not None:
                object_it.boxes[j].box[2] = min(object_it.boxes[j].box[2], im_shape[1]-1)
                object_it.boxes[j].box[3] = min(object_it.boxes[j].box[3], im_shape[0]-1)

            # deal with invisible keypoints
            left_keypoint, right_keypoint = 5000, 0
            left_inx, right_inx = -1, -1

            # 1. Select keypoints that lie on the left and right side of the 2D box
            for i in range(4):
                if object_it.boxes[j].keypoints[i] < left_keypoint:
                    left_keypoint = object_it.boxes[j].keypoints[i]
                    left_inx = i
                if object_it.boxes[j].keypoints[i] > right_keypoint:
                    right_keypoint = object_it.boxes[j].keypoints[i]
                    right_inx = i

            # 2. For keypoints between left and right side, select the visible one
            for i in range(4):
                if i == left_inx or i == right_inx:
                    continue

                if pts3_c_o[i][2] > object_it.pos[2]:
                    object_it.boxes[j].keypoints[i] = -1

        # calculate the union of the left and right box
        object_it.boxes[2].box[0] = min(object_it.boxes[1].box[0], object_it.boxes[0].box[0])
        object_it.boxes[2].box[1] = min(object_it.boxes[1].box[1], object_it.boxes[0].box[1])
        object_it.boxes[2].box[2] = max(object_it.boxes[1].box[2], object_it.boxes[0].box[2])
        object_it.boxes[2].box[3] = max(object_it.boxes[1].box[3], object_it.boxes[0].box[3])

        objects.append(object_it)

    return objects

def project_to_image_for_box(box_list, p):
    if not box_list:
        return None
    final_box_list = []
    for box in box_list:
        corner = box.corners.T
        pts_2d = np.dot(p, np.append(corner,
                                     np.ones((1, corner.shape[1])),
                                     axis=0))

        pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
        pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
        pts_2d = np.delete(pts_2d, 2, 0)
        final_box_list.append(pts_2d.T)

    return final_box_list

def project_to_image(point_cloud, p):
    ''' Projects a 3D point cloud to 2D points for plotting

        Inputs:
            point_cloud: 3D point cloud (3, N)
            p: Camera matrix (3, 4)
        Return:
            pts_2d: the image coordinates of the 3D points in the shape (2, N)

    '''

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d

def point_in_2Dbox(points_im, obj):
    '''Select points contained in object 2D box

        Inputs:
            points_im: N x 2 numpy array in image
            obj: KittiObject
        Return:
            pointcloud indexes

    '''

    point_filter = (points_im[:, 0] > obj.box[0]) & \
                   (points_im[:, 0] < obj.box[2]) & \
                   (points_im[:, 1] > obj.box[1]) & \
                   (points_im[:, 1] < obj.box[3])
    return point_filter

def lidar_to_cam_frame(xyz_lidar, frame_calib):
    '''Transforms the pointclouds to the camera 2 frame.

        Inputs:
            xyz_lidar : N x 3  x,y,z coordinates of the pointcloud in lidar frame
            frame_calib : FrameCalibrationData
        Returns:
            ret_xyz : N x 3  x,y,z coordinates of the pointcloud in cam2 frame

    '''
    # Pad the r0_rect matrix to a 4x4
    r0_rect_mat = frame_calib.r0_rect

    r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)),
                         'constant', constant_values=0)
    r0_rect_mat[3, 3] = 1

    # Pad the tr_vel_to_cam matrix to a 4x4
    tf_mat = frame_calib.tr_velodyne_to_cam0
    tf_mat = np.pad(tf_mat, ((0, 1), (0, 0)),
                    'constant', constant_values=0)
    tf_mat[3, 3] = 1

    # Pad the t_cam2_cam0 matrix to a 4x4
    t_cam2_cam0 = np.identity(4)
    t_cam2_cam0[0:3, 3] = frame_calib.t_cam2_cam0

    # Pad the pointcloud with 1's for the transformation matrix multiplication
    one_pad = np.ones(xyz_lidar.shape[0]).reshape(-1, 1)
    xyz_lidar = np.append(xyz_lidar, one_pad, axis=1)

    # p_cam = P2 * R0_rect * Tr_velo_to_cam * p_velo
    rectified = np.dot(r0_rect_mat, tf_mat)
    to_cam2 = np.dot(t_cam2_cam0, rectified)
    ret_xyz = np.dot(to_cam2, xyz_lidar.T)

    # Change to N x 3 array for consistency.
    return ret_xyz[0:3].T

def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]
    
def check_parallel(points):
    a = np.linalg.norm(points[0] - points[1])
    b = np.linalg.norm(points[1] - points[2])
    c = np.linalg.norm(points[2] - points[0])
    p = (a + b + c) / 2
    
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))
    if area < 1e-2:
        return True
    else:
        return False

def calculate_ground_for_mask(LIDAR_PATH, frame_calib, image_shape=None, thresh_ransac=0.15, back_cut=True):
    ''' Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        Inputs:
            LIDAR_PATH: string specify the lidar file path
            frame_calib: FrameCalibrationData
        Return:
            (3, N) point_cloud in the form [[x,...][y,...][z,...]]

    '''
    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0],[0],[0]])
    if image_shape is not None:
        im_size = [image_shape[1], image_shape[0]]

    else:
        im_size = [1242, 375]

    with open(LIDAR_PATH, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    
    if flag == 1:
        xyzi = data_array.reshape(-1, 4)
    else:
        xyzi = data_array.reshape(-1, 3)
        # print("shape in calculate_ground:", data_array.shape)
    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    # i = xyzi[:, 3]

    # Calculate the point cloud
    point_cloud = np.vstack((x, y, z))
    point_cloud = lidar_to_cam_frame(point_cloud.T, frame_calib)  # N x 3

    # Only keep points in front of camera (positive z)
    if back_cut == True:
        point_cloud = point_cloud[point_cloud[:,2] > 0]   # camera frame 3 x N
    planeDiffThreshold = thresh_ransac
    
    temp = np.sort(point_cloud[:,1])[int(point_cloud.shape[0]*0.75)]
    cloud = point_cloud[point_cloud[:,1]>temp]
    points_np = point_cloud
    mask_all = np.ones(points_np.shape[0])
    for i in range(5):
         best_len = 0
         for iteration in range(min(cloud.shape[0], 100)):
             sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]

             plane = fitPlane(sampledPoints)
             diff = np.abs(np.matmul(points_np, plane) - np.ones(points_np.shape[0])) / np.linalg.norm(plane)
             inlierMask = diff < planeDiffThreshold
             numInliers = inlierMask.sum()
             if numInliers > best_len and np.abs(np.dot(plane/np.linalg.norm(plane),np.array([0,1,0])))>0.9:
                 mask_ground = inlierMask
                 best_len = numInliers
                 best_plane = plane
         mask_all *= 1 - mask_ground
    return mask_all

def calculate_ground(LIDAR_PATH, frame_calib, image_shape=None, thresh_ransac=0.15, back_cut=True, back_cut_z=-5.0):
    ''' Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        Inputs:
            LIDAR_PATH: string specify the lidar file path
            frame_calib: FrameCalibrationData
        Return:
            (3, N) point_cloud in the form [[x,...][y,...][z,...]]
 
    '''
    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0],[0],[0]])
    if image_shape is not None:
        im_size = [image_shape[1], image_shape[0]]

    else:
        im_size = [1242, 375]

    with open(LIDAR_PATH, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    xyzi = data_array.reshape(-1, 4)
    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    i = xyzi[:, 3]

    # Calculate the point cloud
    point_cloud = np.vstack((x, y, z))
    point_cloud = lidar_to_cam_frame(point_cloud.T, frame_calib)  # N x 3

    # Only keep points in front of camera (positive z)
    if back_cut:
        point_cloud = point_cloud[point_cloud[:,2] > back_cut_z]   # camera frame 3 x N
    planeDiffThreshold = thresh_ransac
    temp = np.sort(point_cloud[:,1])[int(point_cloud.shape[0]*0.75)]
    cloud = point_cloud[point_cloud[:,1]>temp]
    points_np = point_cloud
    mask_all = np.ones(points_np.shape[0])
    final_sample_points = None
    for i in range(5):
         best_len = 0
         for iteration in range(min(cloud.shape[0], 100)):
             sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]
                
             while check_parallel(sampledPoints) == True:
                sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]
                continue

             plane = fitPlane(sampledPoints)
             diff = np.abs(np.matmul(points_np, plane) - np.ones(points_np.shape[0])) / np.linalg.norm(plane)
             inlierMask = diff < planeDiffThreshold
             numInliers = inlierMask.sum()
             if numInliers > best_len and np.abs(np.dot(plane/np.linalg.norm(plane),np.array([0,1,0])))>0.9:
                 mask_ground = inlierMask
                 best_len = numInliers
                 best_plane = plane
                 final_sample_points = sampledPoints
         mask_all *= 1 - mask_ground
    return mask_all, final_sample_points


def calculate_gt_point_number(pc, corner):
    p1 = (corner[1, 0] - corner[0, 0], corner[1, 2] - corner[0, 2])
    p2 = (corner[3, 0] - corner[0, 0], corner[3, 2] - corner[0, 2])

    # a·b = |a||b|cos(x)
    # target: 0 < |a|cos(x) < |b| -> (a·b) / |b| < |b| -> (a·b) < |b|·|b|
    
    filter_1 = np.logical_and(
        (pc[:, 0] - corner[0, 0]) * p1[0] + (pc[:, 2] - corner[0, 2]) * p1[1] >= 0.0 * (p1[0] ** 2 + p1[1] ** 2),
        (pc[:, 0] - corner[0, 0]) * p1[0] + (pc[:, 2] - corner[0, 2]) * p1[1] <= 1.0 * (p1[0] ** 2 + p1[1] ** 2)
    )
    
    filter_2 = np.logical_and(
        (pc[:, 0] - corner[0, 0]) * p2[0] + (pc[:, 2] - corner[0, 2]) * p2[1] >= 0.0 * (p2[0] ** 2 + p2[1] ** 2),
        (pc[:, 0] - corner[0, 0]) * p2[0] + (pc[:, 2] - corner[0, 2]) * p2[1] <= 1.0 * (p2[0] ** 2 + p2[1] ** 2)
    )
    
    filter_3 = np.logical_and(pc[:, 1] >= corner[4, 1] * 1, pc[:, 1] <= corner[0, 1] * 1)
    # print(len(np.where(filter_1)[0]), len(np.where(filter_2)[0]), len(np.where(filter_3)[0]))

    filter_final = np.logical_and(filter_1, filter_2, filter_3)
    
    return filter_final


def check_truncate(img_shape, box, threshold=2):
    if min(box[0], box[1]) < 1 or box[2] > img_shape[1] - 2 or box[3] > img_shape[0] - 2:
        return True
    else:
        return False
    
    
def get_point_cloud(LIDAR_PATH, frame_calib, image_shape=None, objects=None):
    ''' Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        Inputs:
            LIDAR_PATH: string specify the lidar file path
            frame_calib: FrameCalibrationData
        Return:
            (3, N) point_cloud in the form [[x,...][y,...][z,...]]

    '''

    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0],[0],[0]])

    if image_shape is not None:
        im_size = [image_shape[1], image_shape[0]]
    else:
        im_size = [1242, 375]

    with open(LIDAR_PATH, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    xyzi = data_array.reshape(-1, 4)

    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    # i = xyzi[:, 3]

    # Calculate the point cloud
    point_cloud = np.vstack((x, y, z))
    point_cloud = lidar_to_cam_frame(point_cloud.T, frame_calib)  # N x 3

    # Only keep points in front of camera (positive z)
    point_cloud = point_cloud[point_cloud[:,2] > 0].T   # camera frame 3 x N

    # Project to image frame
    point_in_im = project_to_image(point_cloud, p=frame_calib.p2).T

    # Filter based on the given image size
    image_filter = (point_in_im[:, 0] > 0) & \
                    (point_in_im[:, 0] < im_size[0]) & \
                    (point_in_im[:, 1] > 0) & \
                    (point_in_im[:, 1] < im_size[1])

    object_filter = np.zeros(point_in_im.shape[0], dtype=bool)

    if objects is not None:
        for i in range(len(objects)):
            object_filter = np.logical_or(point_in_2Dbox(point_in_im, objects[i]), object_filter)

        object_filter = np.logical_and(image_filter, object_filter)
    else:
        object_filter = image_filter

    point_cloud = point_cloud.T[object_filter].T

    return point_cloud

def get_point_cloud_without_image_filter(LIDAR_PATH, frame_calib, image_shape=None, objects=None):
    ''' Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        Inputs:
            LIDAR_PATH: string specify the lidar file path
            frame_calib: FrameCalibrationData
        Return:
            (3, N) point_cloud in the form [[x,...][y,...][z,...]]

    '''
    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0],[0],[0]])
    if image_shape is not None:
        im_size = [image_shape[1], image_shape[0]]
    else:
        im_size = [1242, 375]

    with open(LIDAR_PATH, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    xyzi = data_array.reshape(-1, 4)
    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    i = xyzi[:, 3]

    # Calculate the point cloud
    point_cloud = np.vstack((x, y, z))
    point_cloud = lidar_to_cam_frame(point_cloud.T, frame_calib)  # N x 3

    # Only keep points in front of camera (positive z)
    # point_cloud = point_cloud[point_cloud[:,2] > 0].T   # camera frame 3 x N
    point_cloud = point_cloud[point_cloud[:, 2] > -5].T   # camera frame 3 x N

    # Project to image frame
    point_in_im = project_to_image(point_cloud, p=frame_calib.p2).T

    # Filter based on the given image size

    """
    image_filter = (point_in_im[:, 0] > 0) & \
                    (point_in_im[:, 0] < im_size[0]) & \
                    (point_in_im[:, 1] > 0) & \
                    (point_in_im[:, 1] < im_size[1])
    """

    object_filter = np.zeros(point_in_im.shape[0], dtype=bool)
    if objects is not None:
        for i in range(len(objects)):
            object_filter = np.logical_or(point_in_2Dbox(point_in_im, objects[i]), object_filter)

        # object_filter = np.logical_and(image_filter, object_filter)
    else:
        object_filter = image_filter

    #point_cloud = point_cloud.T[object_filter]

    return point_cloud.T, object_filter


def get_point_cloud_my_version(LIDAR_PATH, frame_calib, image_shape=None, objects=None, back_cut=True, back_cut_z=-5.0):
    ''' Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        Inputs:
            LIDAR_PATH: string specify the lidar file path
            frame_calib: FrameCalibrationData
        Return:
            (3, N) point_cloud in the form [[x,...][y,...][z,...]]

    '''
    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0],[0],[0]])
    if image_shape is not None:
        im_size = [image_shape[1], image_shape[0]]
    else:
        im_size = [1242, 375]

    with open(LIDAR_PATH, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    xyzi = data_array.reshape(-1, 4)
    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    i = xyzi[:, 3]

    # Calculate the point cloud
    point_cloud = np.vstack((x, y, z))
    point_cloud = lidar_to_cam_frame(point_cloud.T, frame_calib)  # N x 3

    # Only keep points in front of camera (positive z)
    if back_cut: 
        point_cloud = point_cloud[point_cloud[:,2] > back_cut_z].T   # camera frame 3 x N
    else:
        point_cloud = point_cloud.T

    # Project to image frame
    point_in_im = project_to_image(point_cloud, p=frame_calib.p2).T

    # Filter based on the given image size

    image_filter = (point_in_im[:, 0] > 0) & \
                    (point_in_im[:, 0] < im_size[0]) & \
                    (point_in_im[:, 1] > 0) & \
                    (point_in_im[:, 1] < im_size[1])

    object_filter = np.zeros(point_in_im.shape[0], dtype=bool)
    if objects is not None:
        for i in range(len(objects)):
            object_filter = np.logical_or(point_in_2Dbox(point_in_im, objects[i]), object_filter)

        object_filter = np.logical_and(image_filter, object_filter)
    else:
        object_filter = image_filter

    #point_cloud = point_cloud.T[object_filter]

    return point_cloud.T, object_filter

def get_point_cloud_for_mask(LIDAR_PATH, frame_calib, image_shape=None, objects=None):
    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0],[0],[0]])
    if image_shape is not None:
        im_size = [image_shape[1], image_shape[0]]
    else:
        im_size = [1242, 375]
        
    point_cloud = np.fromfile(LIDAR_PATH, dtype = np.float).reshape(-1, 3)
    point_cloud = point_cloud[point_cloud[:, 2] > 0].T
    
    point_in_im = project_to_image(point_cloud, p=frame_calib.p2).T
    
    image_filter = (point_in_im[:, 0] > 0) & \
                   (point_in_im[:, 0] < im_size[0]) & \
                   (point_in_im[:, 1] > 0) & \
                   (point_in_im[:, 1] < im_size[1])
    
    object_filter = np.zeros(point_in_im.shape[0], dtype=bool)
    
    if objects is not None:
        for i in range(len(objects)):
            object_filter = np.logical_or(point_in_2Dbox(point_in_im, objects[i]), object_filter)

        object_filter = np.logical_and(image_filter, object_filter)
    else:
        object_filter = image_filter
    
    return point_cloud.T, object_filter

def get_point_cloud_for_box2d(LIDAR_PATH, frame_calib, box_2d, objects=None):
    ''' Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        Inputs:
            LIDAR_PATH: string specify the lidar file path
            frame_calib: FrameCalibrationData
        Return:
            (3, N) point_cloud in the form [[x,...][y,...][z,...]]

    '''
    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0],[0],[0]])

    with open(LIDAR_PATH, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    xyzi = data_array.reshape(-1, 4)

    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    # i = xyzi[:, 3]

    # Calculate the point cloud
    point_cloud = np.vstack((x, y, z))
    point_cloud = lidar_to_cam_frame(point_cloud.T, frame_calib)  # N x 3

    # Only keep points in front of camera (positive z)
    point_cloud = point_cloud[point_cloud[:,2] > 0].T   # camera frame 3 x N

    # Project to image frame
    point_in_im = project_to_image(point_cloud, p=frame_calib.p2).T

    # Filter based on the given image size
    image_filter = (point_in_im[:, 0] > box_2d[0]) & \
                    (point_in_im[:, 0] < box_2d[2]) & \
                    (point_in_im[:, 1] > box_2d[1]) & \
                    (point_in_im[:, 1] < box_2d[3])

    object_filter = np.zeros(point_in_im.shape[0], dtype=bool)

    if objects is not None:
        for i in range(len(objects)):
            object_filter = np.logical_or(point_in_2Dbox(point_in_im, objects[i]), object_filter)

        object_filter = np.logical_and(image_filter, object_filter)
    else:
        object_filter = image_filter

    point_cloud = point_cloud.T[object_filter].T

    return point_cloud

def infer_boundary(im_shape, boxes_left):
    ''' Approximately infer the occlusion border for all objects
        accoording to the 2D bounding box

        Inputs:
            im_shape: H x W x 3
            boxes_left: rois x 4
        Return:
            left_right: left and right borderline for each object rois x 2
    '''
    left_right = np.zeros((boxes_left.shape[0],2), dtype=np.float32)
    depth_line = np.zeros(im_shape[1]+1, dtype=float)
    for i in range(boxes_left.shape[0]):
        for col in range(int(boxes_left[i,0]), int(boxes_left[i,2])+1):
            pixel = depth_line[col]
            depth = 1050.0/boxes_left[i,3]
            if pixel == 0.0:
                depth_line[col] = depth
            elif depth < depth_line[col]:
                depth_line[col] = (depth+pixel)/2.0

    for i in range(boxes_left.shape[0]):
        left_right[i,0] = boxes_left[i,0]
        left_right[i,1] = boxes_left[i,2]
        left_visible = True
        right_visible = True
        if depth_line[int(boxes_left[i,0])] < 1050.0/boxes_left[i,3]:
            left_visible = False
        if depth_line[int(boxes_left[i,2])] < 1050.0/boxes_left[i,3]:
            right_visible = False

        if right_visible == False and left_visible == False:
            left_right[i,1] = boxes_left[i,0]

        for col in range(int(boxes_left[i,0]), int(boxes_left[i,2])+1):
            if left_visible and depth_line[col] >= 1050.0/boxes_left[i,3]:
                left_right[i,1] = col
            elif right_visible and depth_line[col] < 1050.0/boxes_left[i,3]:
                left_right[i,0] = col
    return left_right

def region_grow_my_version(pc, mask_search, mask_origin, thresh, ratio=0.8):
    pc_search = pc[mask_search==1]
    mask = mask_origin.copy()
    best_len = 0
    mask_best = np.zeros((pc.shape[0]))
    while mask.sum() > 0:
        seed = pc[mask==1][0]
        seed_mask = np.zeros((pc_search.shape[0]))
        seed_mask_all = np.zeros((pc.shape[0]))
        seed_list = [seed]
        flag = 1
        while len(seed_list) > 0:
            temp = seed_list.pop(0)
            dis = np.linalg.norm(pc_search - temp, axis=-1)
            index = np.argmin(dis)
            seed_mask[index] = 1
            valid_mask = (dis < thresh) * (1 - seed_mask)
            seed_list += list(pc_search[valid_mask==1])
            seed_mask[valid_mask == 1] = 1
            seed_mask_all[mask_search==1] = seed_mask
            if ratio is not None and (seed_mask_all*mask_origin).sum()/seed_mask.sum().astype(np.float32)<ratio:
                flag = 0
                break
        if flag == 1:
            if seed_mask.sum() > best_len:
                best_len = seed_mask.sum()
                mask_best = seed_mask_all
        mask *= (1 - seed_mask_all)

    if ratio is not None:
        return mask_best*mask_origin
    else:
        return mask_best

#---------------------------------------------------- Data Writing -------------------------------------------------#
def write_detection_results(result_dir, file_number, calib, box_left, pos, dim, orien, score):
    '''One by one write detection results to KITTI format label files.
    '''
    if result_dir is None: return
    result_dir = result_dir + '/data'

    # convert the object from cam2 to the cam0 frame
    dis_cam02 = calib.t_cam2_cam0[0]

    output_str = 'Car -1 -1 '
    alpha = orien - m.pi/2 + m.atan2(-pos[0], pos[2])
    output_str += '%f %f %f %f %f ' % (alpha, box_left[0],box_left[1],box_left[2],box_left[3])
    output_str += '%f %f %f %f %f %f %f %f \n' % (dim[1],dim[0],dim[2],pos[0]-dis_cam02,pos[1],
                                                  pos[2],orien-1.57,score)

    # Write TXT files
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    pred_filename = result_dir + '/' + file_number + '.txt'
    with open(pred_filename, 'a') as det_file:
        det_file.write(output_str)