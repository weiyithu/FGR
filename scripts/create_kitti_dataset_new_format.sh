set -x
set -e

rm -rf $2/data/kitti/training $2/data/kitti/testing

mkdir -p $2/data/kitti/training $2/data/kitti/testing 
ln -s $1/data_object_image_2/training/image_2 $2/data/kitti/training/image_2
ln -s $1/data_object_label_2/training/label_2 $2/data/kitti/training/label_2
ln -s $1/data_object_calib/training/calib $2/data/kitti/training/calib
ln -s $1/data_object_velodyne/training/velodyne $2/data/kitti/training/velodyne

ln -s $1/data_object_image_2/testing/image_2 $2/data/kitti/testing/image_2
ln -s $1/data_object_calib/testing/calib $2/data/kitti/testing/calib
ln -s $1/data_object_velodyne/testing/velodyne $2/data/kitti/testing/velodyne
