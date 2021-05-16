FGR
===
This repository contains the python implementation for paper "FGR: Frustum-Aware Geometric Reasoning for Weakly Supervised 3D Vehicle Detection"(ICRA 2021)\[[arXiv](https://arxiv.org/abs/2012.00987)\]

<img src="./imgs/FGR.png">

## Installation

### Prerequisites
- Python 3.6
- scikit-learn, opencv-python, numpy, easydict, pyyaml

```shell
conda create -n FGR python=3.6
conda activate FGR
pip install -r requirements.txt
```

## Usage
### Data Preparation

Please download the KITTI 3D object detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize them as follows:

```text
${Root Path To Your KITTI Dataset}
├── data_object_image_2
│   ├── training
│   │   └── image_2
│   └── testing (optional)
│       └── image_2
│
├── data_object_label_2
│   └── training
│       └── label_2
│
├── data_object_calib
│   ├── training
│   │   └── calib
│   └── testing (optional)
│       └── calib
│
└── data_object_velodyne
    ├── training
    │   └── velodyne
    └── testing (optional)
        └── velodyne
```

### Retrieving psuedo labels
#### Stage I: Getting Region-Grow Result
We set this stage to store middle result, with selected point clouds for each possible cars and other basic information. Please run the following command:

```shell
python save_region_grow_result.py --kitti_dataset_dir ${Path To Your KITTI Dataset} --output_dir ${Path To Store Region-Growth Result}
```

- This Python file uses multiprocessing.Pool, which requires the number of parallel processes to execute. Default process is 8, so change this number by adding extra parameter "--process ${Process Number You Want}" in above command if needed. 
- The space of region-growth result takes about **170M**, and the execution time is about **3** hours when using process=8 (default)

#### Stage II: Iterative alorithm execution and getting psuedo labels with KITTI format
In this stage, psuedo labels with KITTI format will be calculated and stored. Please run the following command:

```shell
python detect.py --kitti_dataset_dir ${Path To Your KITTI Dataset} --final_save_dir ${Path To Store Psuedo Labels} --pickle_save_path ${Path To Save Region-Growth Result}
```

- For convenience, we simply provide the psuedo labels create by this repo in ./FGR/detection_result.zip, with pusedo training labels and GT validation labels.
- The multiprocessing.Pool is also used, with default process **16**. Change it by adding extra parameter "--process ${Process Number You Want}" in above command if needed. 
- Add "--not_merge_valid_labels" to ignore validation labels. We only create psuedo labels in training dataset, for further training in deep models, we simply copy groundtruth validation labels to saved path. If you just want to preserve psuedo labels from training dataset, just add this parameter
- Add "--save_det_image" if you want to visualize the calculated bbox. By this command, the 'image' directory will be created under "final_save_dir", with visualized png files. 
- One visualized sample is as follows: 
    - **white**  points record the point clouds of one car based on region-growth result 
    - **cyan**   lines record left/right side of frustum
    - **green**  point records the **key vertex** described in our paper
    - **yellow** lines record GT bbox's 2D projection
    - **purple** box records initial bounding box based on point clouds
    - **red**    box records the intersection based on purple box, which is also psuedo 3D bbox's 2D projection

<img src="./imgs/sample_bbox.png" width = "600" height = "600" div align=center />

### Use psuedo labels to train deep models
#### 1. Getting Startted

Please refer to the OpenPCDet repo [here](https://github.com/open-mmlab/OpenPCDet) by Open-mmlab, complete all the required installation

After downloading the repo and completing all the installation, a small modification in original code is needed：

```text
--------------------------------------------------
pcdet.datasets.kitti.kitti_dataset:
1. line between 142 and 143, add: "if len(obj_list) == 0: return None"
2. line after 191, delete "return list(infos)", and add:

final_result = list(infos)
while None in final_result:
    final_result.remove(None)
            
return final_result
--------------------------------------------------
```

This is because when creating dataset, OpenPCDet (the repo) requires each label file to have at least one valid label. In our psuedo labels, however, some bad labels will be removed and the label file may be empty.

#### 2. Data Preparation

In this repo, the KITTI dataset storage is as follows:

```text
data/kitti
├── testing
│   ├── calib
│   ├── image_2
│   └── velodyne
└── training
    ├── calib
    ├── image_2
    ├── label_2
    └── velodyne
```

It's different from our dataset storage, so we provide a script to construct this structure based on **symlink**:

```shell
sh create_kitti_dataset_new_format.sh ${Path To KITTI Dataset} ${Path To OpenPCDet Directory}
```

#### 3. Start training

Please follow the OpenPCDet instructions and run PointRCNN models. The reminder is that if you want to train the model by newly created labels instead of groundtruth, please remove the symlink of 'training/label_2' temporarily, and add a new symlink to psuedo label path. If you still need to use GT labels later, please replace the symlink to groundtruth label path.

## Citation 
If you find our work useful in your research, please consider citing:
```
@inproceedings{wei2020fgr,
  title={{FGR: Frustum-Aware Geometric Reasoning for Weakly Supervised 3D Vehicle Detection}},
  author={Wei, Yi and Su, Shang and Lu, Jiwen and Zhou, Jie},
  booktitle={ICRA},
  year={2021}
}
```
