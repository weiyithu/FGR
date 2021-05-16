from .kitti_utils_official import *
import traceback


def draw_bbox_3d_to_2d_gt(img, corner, AverageValue_x, AverageValue_y):
    
    gt_box_3d = corner.copy()
    for point in gt_box_3d:
        point[0] = point[0] * 100 + 250 - AverageValue_x
        point[2] = point[2] * 100 + 250 - AverageValue_y

    cv2.line(img, (int(gt_box_3d[0][0]), int(gt_box_3d[0][2])), (int(gt_box_3d[1][0]), int(gt_box_3d[1][2])), (0, 255, 255), 1, 4)
    cv2.line(img, (int(gt_box_3d[1][0]), int(gt_box_3d[1][2])), (int(gt_box_3d[2][0]), int(gt_box_3d[2][2])), (0, 255, 255), 1, 4)
    cv2.line(img, (int(gt_box_3d[2][0]), int(gt_box_3d[2][2])), (int(gt_box_3d[3][0]), int(gt_box_3d[3][2])), (0, 255, 255), 1, 4)
    cv2.line(img, (int(gt_box_3d[3][0]), int(gt_box_3d[3][2])), (int(gt_box_3d[0][0]), int(gt_box_3d[0][2])), (0, 255, 255), 1, 4)
    
    return img


def draw_frustum_lr_line(img, left_point, right_point, AverageValue_x, AverageValue_y):
    
    left_point_draw = np.array([0., 0.])
    left_point_draw[0] = (left_point[0] * 20000 + 250 - AverageValue_x)
    left_point_draw[1] = (left_point[1] * 20000 + 250 - AverageValue_y)

    right_point_draw = np.array([0., 0.])
    right_point_draw[0] = (right_point[0] * 20000 + 250 - AverageValue_x)
    right_point_draw[1] = (right_point[1] * 20000 + 250 - AverageValue_y)

    initial_point_draw = np.array([0., 0.])
    initial_point_draw[0] = 250 - AverageValue_x
    initial_point_draw[1] = 250 - AverageValue_y

    cv2.line(img, tuple(initial_point_draw.astype(np.float32)), tuple(left_point_draw.astype(np.float32)),
             (255, 255, 0), 1, 4)
    cv2.line(img, tuple(initial_point_draw.astype(np.float32)), tuple(right_point_draw.astype(np.float32)),
             (255, 255, 0), 1, 4)

    return img


def draw_bbox_3d_to_2d_psuedo_no_intersection(img, box_no_intersection, AverageValue_x, AverageValue_y):
    
    box_draw = box_no_intersection.copy()
    
    for var in box_draw:
        var[0] = var[0] * 100 + 250 - AverageValue_x
        var[1] = var[1] * 100 + 250 - AverageValue_y
    
    cv2.line(img, tuple(box_draw[0]), tuple(box_draw[1]), (255, 0, 255), 1, 4)
    cv2.line(img, tuple(box_draw[1]), tuple(box_draw[2]), (255, 0, 255), 1, 4)
    cv2.line(img, tuple(box_draw[2]), tuple(box_draw[3]), (255, 0, 255), 1, 4)
    cv2.line(img, tuple(box_draw[3]), tuple(box_draw[0]), (255, 0, 255), 1, 4)
    
    return img


def draw_bbox_3d_to_2d_psuedo_with_key_vertex(img, FinalPoint, loc1, loc2, loc3, AverageValue_x, AverageValue_y):
    
    loc1_draw = np.array([0., 0.])
    loc2_draw = np.array([0., 0.])
    loc3_draw = np.array([0., 0.])
    loc1_draw[0] = loc1[0] * 100 + 250 - AverageValue_x
    loc1_draw[1] = loc1[1] * 100 + 250 - AverageValue_y
    loc2_draw[0] = loc2[0] * 100 + 250 - AverageValue_x
    loc2_draw[1] = loc2[1] * 100 + 250 - AverageValue_y
    loc3_draw[0] = loc3[0] * 100 + 250 - AverageValue_x
    loc3_draw[1] = loc3[1] * 100 + 250 - AverageValue_y
    
    # draw key vertex with larger point than normal point cloud
    FinalPoint_draw = np.array([0., 0.])
    FinalPoint_draw[0] = FinalPoint[0] * 100 + 250 - AverageValue_x
    FinalPoint_draw[1] = FinalPoint[1] * 100 + 250 - AverageValue_y
    cv2.circle(img, tuple(FinalPoint_draw.astype(np.float32)), 3, (0, 255, 0), 4)

    cv2.line(img, tuple(loc1_draw.astype(np.float32)), tuple(FinalPoint_draw.astype(np.float32)), (0, 0, 255), 1, 4)
    cv2.line(img, tuple(loc1_draw.astype(np.float32)), tuple(loc3_draw.astype(np.float32)), (0, 0, 255), 1, 4)
    cv2.line(img, tuple(loc3_draw.astype(np.float32)), tuple(loc2_draw.astype(np.float32)), (0, 0, 255), 1, 4)
    cv2.line(img, tuple(loc2_draw.astype(np.float32)), tuple(FinalPoint_draw.astype(np.float32)), (0, 0, 255), 1, 4)
    
    return img

def draw_point_clouds(img, KeyPoint_for_draw, AverageValue_x, AverageValue_y):

    for point in KeyPoint_for_draw:
        a = point[0] * 100 + 250 - AverageValue_x
        b = point[1] * 100 + 250 - AverageValue_y
        cv2.circle(img, (int(a), int(b)), 1, (255, 255, 255), 2)
    
    return img


def draw_2d_box_in_2d_image(KeyPoint_3d, box_2d, p2):

    img = np.zeros((1000, 1000, 3), 'f4')

    KeyPoint_for_draw = np.copy(KeyPoint_3d[:, [0, 2]])
    KeyPoint = KeyPoint_3d[:, [0, 2]]

    AverageValue_x = np.mean(KeyPoint[:, 0]) * 100
    AverageValue_y = np.mean(KeyPoint[:, 1]) * 100

    for point in KeyPoint_for_draw:
        a = point[0] * 50 + 500 - AverageValue_x
        b = point[1] * 50 + 100 - AverageValue_y
        cv2.circle(img, (int(a), int(b)), 1, (255, 255, 255), 0)

    left_point = np.array([box_2d[0], 0, 1])
    right_point = np.array([box_2d[2], 0, 1])

    left_point = np.linalg.inv(p2[:, [0, 1, 2]]).dot(left_point.T)
    right_point = np.linalg.inv(p2[:, [0, 1, 2]]).dot(right_point.T)

    left_point = left_point[[0, 2]]
    right_point = right_point[[0, 2]]

    left_point_draw = np.array([0., 0.])
    right_point_draw = np.array([0., 0.])

    while True:
        zoom_factor = 60000
        try:
            left_point_draw[0] = (left_point[0] * zoom_factor + 100 - AverageValue_x)
            left_point_draw[1] = (left_point[1] * zoom_factor + 100 - AverageValue_y)

            right_point_draw[0] = (right_point[0] * zoom_factor + 100 - AverageValue_x)
            right_point_draw[1] = (right_point[1] * zoom_factor + 100 - AverageValue_y)
        except OverflowError:
            zoom_factor /= 6
            continue
        else:
            break

    initial_point_draw = np.array([0., 0.])
    initial_point_draw[0] = 500 - AverageValue_x
    initial_point_draw[1] = 100 - AverageValue_y

    cv2.line(img, tuple(initial_point_draw.astype(np.float32)), tuple(left_point_draw.astype(np.float32)),
             (255, 255, 0), 1, 4)
    cv2.line(img, tuple(initial_point_draw.astype(np.float32)), tuple(right_point_draw.astype(np.float32)),
             (255, 255, 0), 1, 4)

    return img

def draw_3d_box_in_2d_image(img, box):

    # draw 3D box in 2D RGB image, if needed

    cv2.line(img, (int(box[0][0]), int(box[0][1])), (int(box[1][0]), int(box[1][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[1][0]), int(box[1][1])), (int(box[2][0]), int(box[2][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[2][0]), int(box[2][1])), (int(box[3][0]), int(box[3][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[3][0]), int(box[3][1])), (int(box[0][0]), int(box[0][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[0][0]), int(box[0][1])), (int(box[4][0]), int(box[4][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[1][0]), int(box[1][1])), (int(box[5][0]), int(box[5][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[2][0]), int(box[2][1])), (int(box[6][0]), int(box[6][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[3][0]), int(box[3][1])), (int(box[7][0]), int(box[7][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[4][0]), int(box[4][1])), (int(box[5][0]), int(box[5][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[5][0]), int(box[5][1])), (int(box[6][0]), int(box[6][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[6][0]), int(box[6][1])), (int(box[7][0]), int(box[7][1])), (0, 0, 255), 1, 4)
    cv2.line(img, (int(box[7][0]), int(box[7][1])), (int(box[4][0]), int(box[4][1])), (0, 0, 255), 1, 4)

    return img

def show_velodyne_in_camera(loc0, loc1, loc2, loc3, y_min, y_max):

    # use mayavi to draw 3D bbox 
    corners = np.array([[[loc0[0], loc1[0], loc2[0], loc3[0], loc0[0], loc1[0], loc2[0], loc3[0]],
                         [loc0[1], loc1[1], loc2[1], loc3[1], loc0[1], loc1[1], loc2[1], loc3[1]],
                         [y_min, y_min, y_min, y_min, y_max, y_max, y_max, y_max]]],
                       dtype=np.float32)

    for i in range(corners.shape[0]):
        corner = corners[i]
        idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
        x = corner[0, idx]
        y = corner[1, idx]
        z = corner[2, idx]
        mayavi.mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', representation='wireframe', line_width=5)