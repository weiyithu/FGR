import traceback
import argparse
from   multiprocessing import Pool
import os
import cv2
import time
import pickle
import numpy as np
import yaml
from easydict import EasyDict

from utils import kitti_utils_official


def save_result(seq, output_dir, kitti_dataset_path, region_growth_config):
    # print("%s: begin" % seq)
    thresh_ransac = region_growth_config.THRESH_RANSAC
    thresh_seg_max = region_growth_config.THRESH_SEG_MAX
    ratio = region_growth_config.REGION_GROWTH_RATIO

    total_object_number = 0
    iou_collection = []

    # path to save dataset
    img_path   = os.path.join(kitti_dataset_path, 'data_object_image_2/training/image_2/%s.png'   % seq)
    lidar_path = os.path.join(kitti_dataset_path, 'data_object_velodyne/training/velodyne/%s.bin' % seq)
    calib_path = os.path.join(kitti_dataset_path, 'data_object_calib/training/calib/%s.txt'       % seq)
    label_path = os.path.join(kitti_dataset_path, 'data_object_label_2/training/label_2/%s.txt'   % seq)

    img = cv2.imread(img_path)
    calib = kitti_utils_official.read_obj_calibration(calib_path)
    objects = kitti_utils_official.read_obj_data(label_path, calib, img.shape)

    dic = {'calib': calib,
           'shape': img.shape,
           'sample': {}}

    if len(objects) == 0:

        with open(os.path.join(output_dir, "%s.pickle" % seq), 'wb') as fp:
            pickle.dump(dic, fp)

        print("%s: empty" % seq)
        return

    pc_all, object_filter_all = kitti_utils_official.get_point_cloud_my_version(lidar_path, calib, img.shape, back_cut=False)
    mask_ground_all, ground_sample_points = kitti_utils_official.calculate_ground(lidar_path, calib, img.shape, 0.2, back_cut=False)

    z_list = []
    index_list = []
    valid_list = []

    valid_index = []

    for i in range(len(objects)):
        total_object_number += 1
        flag = 1

        _, object_filter = kitti_utils_official.get_point_cloud_my_version(lidar_path, calib, img.shape, [objects[i].boxes[0]], back_cut=False)
        pc = pc_all[object_filter == 1]
        
        filter_sample = kitti_utils_official.calculate_gt_point_number(pc, objects[i].corners)
        pc_in_box_3d = pc[filter_sample]
        if len(pc_in_box_3d) < 30:
            flag = 0

        if flag == 1:
            valid_list.append(i)

        z_list.append(np.median(pc[:, 2]))
        index_list.append(i)

    sort = np.argsort(np.array(z_list))
    object_list = list(np.array(index_list)[sort])

    mask_object = np.ones((pc_all.shape[0]))

    # [add] dict to be saved
    dic = {'calib': calib,
           'shape': img.shape,
           'ground_sample': ground_sample_points,
           'sample': {}}

    for i in object_list:
        result = np.zeros((7, 2))
        count = 0
        mask_seg_list = []

        for j in range(thresh_seg_max):
            thresh = (j + 1) * 0.1
            _, object_filter = kitti_utils_official.get_point_cloud_my_version(
                lidar_path, calib, img.shape, [objects[i].boxes[0]], back_cut=False)
            
            filter_z = pc_all[:, 2] > 0
            mask_search = mask_ground_all * object_filter_all * mask_object * filter_z
            mask_origin = mask_ground_all * object_filter * mask_object * filter_z
            mask_seg = kitti_utils_official.region_grow_my_version(pc_all.copy(), 
                                                                   mask_search, mask_origin, thresh, ratio)
            if mask_seg.sum() == 0:
                continue

            if j >= 1:
                mask_seg_old = mask_seg_list[-1]
                if mask_seg_old.sum() != (mask_seg * mask_seg_old).sum():
                    count += 1
            result[count, 0] = j  
            result[count, 1] = mask_seg.sum()
            mask_seg_list.append(mask_seg)
            
        best_j = result[np.argmax(result[:, 1]), 0]
        try:
            mask_seg_best = mask_seg_list[int(best_j)]
            mask_object *= (1 - mask_seg_best)
            pc = pc_all[mask_seg_best == 1].copy()
        except IndexError:
            # print("bad region grow result! deprecated")
            continue
        if i not in valid_list:
            continue

        if kitti_utils_official.check_truncate(img.shape, objects[i].boxes_origin[0].box):
            # print('object %d truncates in %s, with bbox %s' % (i, seq, str(objects[i].boxes_origin[0].box)))

            mask_origin_new = mask_seg_best
            mask_search_new = mask_ground_all
            thresh_new      = (best_j + 1) * 0.1

            mask_seg_for_truncate = kitti_utils_official.region_grow_my_version(pc_all.copy(),
                                                                                mask_search_new,
                                                                                mask_origin_new,
                                                                                thresh_new,
                                                                                ratio=None)
            pc_truncate = pc_all[mask_seg_for_truncate == 1].copy()

            dic['sample'][i] = {
                'truncate': True,
                'object': objects[i],
                'pc': pc_truncate
            }

        else:
            dic['sample'][i] = {
                'truncate': False,
                'object': objects[i],
                'pc': pc
            }


    with open(os.path.join(output_dir, "%s.pickle" % seq), 'wb') as fp:
        pickle.dump(dic, fp)

    print("%s: end" % seq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_dir',
        default="../split/train.txt",
        help="input directory of raw dataset"
    )
    parser.add_argument(
        '--output_dir',
        default="./region_grow_result",
        help="output directory to store region-grow results"
    )
    parser.add_argument(
        '--process',
        default=8,
        type=int,
        help="number of parallel processes"
    )
    parser.add_argument(
        '--kitti_dataset_path',
        default='/mnt1/yiwei/kitti_dataset',
        type=str,
        help='path to store official kitti dataset'
    )
    parser.add_argument(
        '--region_growth_config_path',
        type=str,
        default='../configs/config.yaml',
        help='pre-defined parameters for running save_region_growth.py'
    )

    args = parser.parse_args()
    
    assert os.path.isfile(args.root_dir), "Can't find train sequence text file in: %s" % args.root_dir
    assert os.path.isfile(args.region_growth_config_path), "Can't find region growth config file in: %s" % args.region_growth_config_path
    assert os.path.isdir(args.kitti_dataset_path), "Can't find kitti dataset in: %s" % args.kitti_dataset_path

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # load config file
    with open(args.region_growth_config_path, 'r') as fp:
        region_growth_config = yaml.load(fp, Loader=yaml.FullLoader)
        region_growth_config = EasyDict(region_growth_config)
    
    pool = Pool(processes=args.process)

    assert args.root_dir.endswith(".txt"), "input training label must be sequences in train.txt!"
    with open(args.root_dir, 'r') as fp:
        seq_collection = fp.readlines()

    for i in range(len(seq_collection)):
        seq_collection[i] = seq_collection[i].strip()
        assert len(seq_collection[i]) == 6

    print("find %d training samples." % len(seq_collection))
    print("result will be stored in %s" % args.output_dir)

    start = time.time()
    for seq in seq_collection:
        # save_result(seq)
        pool.apply_async(save_result, (seq, args.output_dir, args.kitti_dataset_path, region_growth_config))

    pool.close()
    pool.join()
    
    print("runtime: %.4fs" % (time.time() - start))
