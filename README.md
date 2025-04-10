# COMP0248 GRPA

## Dataset

4 subfolders: 
 - CW2-Dataset : refer to coursework requirements.
 - realsense : [text](https://drive.google.com/file/d/1SzfCeA_qOrDVZgeYqhs0KyEmW27aEHnw/view?usp=drive_link)
 - realsense_raw : [text](https://drive.google.com/file/d/1LnpcNDON0TLvxWPAgkwbaZXPW-4Y0Ywx/view?usp=sharing) 
 - dataset ： [text](https://drive.google.com/file/d/1aT0Iam7XWZPmjvM55s06Zks6c4W4hmsc/view?usp=sharing)


Please place the dataset under folder `data`:
```
data/ 
├── CW2-Dataset/ # Public dataset provided by the course (CW2) 
│ └── data/ 
│ ├── mit_* # MIT training scenes in Sun 3D
│ ├── harvard_* # Harvard testing scenes in Sun 3D
│ └── ... # Each scene contains image/, depth/, labels/, intrinsics.txt 
├── realsense/ # Processed Realsense captures 
│ ├── image/ # RGB images 
│ ├── depth/ # Depth images (after inpainting) 
│ └── intrinsics.txt # Camera intrinsics file 
├── realsense_raw/ # Raw Realsense captures (unprocessed) 
│ └── [timestamp]_[scene_name]/ # Scene folders with original images and depth 
├── dataset/ # Final processed datasets used for training and evaluation 
│ ├── point_clouds/ # Raw point clouds with labels 
│ ├── point_clouds_C/ # Standardized 9D point clouds (xyz, rgb, normalized xyz) 
│ ├── depth_img/ # RGB-D pairs used for depth completion training 
│ ├── realsense_point_clouds_C/ # Processed Realsense point clouds (standard format) 
│ └── realsense_depth_img_C/ # RGB-D pairs from Realsense captures
```

Above folders could be generated automatically, following commands at the last part in this readme. 


## Pretrained Models

All evaluation results, model checkpoints, and performance plots are stored under the `log/` directory.

Online link: [text](https://drive.google.com/file/d/1hBNld0Vw1kuaOtz-9lvqHINvR3WARdsp/view?usp=sharing)

Please place the logs under folder `log`：
Organized by experiment name and pipeline type.

Each pipeline subfolder (e.g. `binary_pointnet2_pipeline_A`) contains:

| File / Folder                         | Description |
|--------------------------------------|-------------|
| `checkpoints/`                       | Stores the best model (e.g. `best_model.pth`) for later inference |
| `*_train.txt`, `*_test_metrics.txt` | Text logs showing training/test metrics like accuracy, IoU, F1 |
| `accuracy_curve.png`                | Accuracy trend over epochs |
| `loss_curve.png`                    | Loss trend over epochs |
| `confusion_matrix.png`              | Classification Confusion matrix for test data |
| `depth_evaluation.txt`              | Pipeline B Logs for evaluating the influence of depth preprocessing |
| `eval_results.csv`                  | Segmentation CSV-formatted evaluation scores (e.g., for real-world Realsense data) |
| `eval.txt`                          | Segmentation Summary of accuracy / IoU on test set |


## To run the code

Install Envroinment:
```
pip install -r requirements.txt
```

Below are the main components of the project and their purposes:

### 1. `depth2cloud.py`
Generate 3D point cloud from RGB & Depth images.  
Outputs `.npy` (point cloud with RGB and normalized coordinates) and `.ply` (for visualization).  
Used in all pipelines.

---

### 2. **Pipeline A** - Binary Classification of Table (using PointNet++)
- Training: `pipeline_A_pointnet2_train.py`
- Testing: `pipeline_A_pointnet2_test.py`
- Model: `pointnet2_cls.py`
- Task: Binary classification of “table” vs. “non-table”
- Data: Can be run on either `CW2` dataset or `Realsense` point clouds

---

### 3. **Pipeline B** - Depth Estimation + Classification
- Script: `pipeline_B_depth_cls.py`
- Steps:
  - Depth Estimation: Use DepthAnything2 with `vitl` checkpoints
  - Generate updated point cloud
  - Train/test a new PointNet++ model based on updated clouds
- Extra Logs: Includes `depth_evaluation.txt` for depth quality effects
- Data: `CW2` and `Realsense`

---

### 4. **Pipeline C** - Semantic Segmentation (PointNet++)
- Point Cloud Generation: `pipeline_C_depth2cloud.py`
  - Outputs **9D** feature point clouds (XYZ, RGB, normalized XYZ)
- Training: `pipeline_C_train.py`
- Testing:
  - On CW2: `pipeline_C_test.py`
  - On Realsense: `pipeline_C_test_realsense.py`
- Model: `pointnet2_seg.py`
- Output: Per-point segmentation (table / non-table)

---

## `src/` Structure Summary 

| Script                          | Function |
|--------------------------------|----------|
| `dataloader.py`                | Dataset loader for training/testing |
| `depth2cloud.py`               | RGB-D to point cloud generation (used in Pipeline A/B) |
| `pipeline_A_pointnet2_*`       | Binary classification with PointNet++ |
| `pipeline_B_depth_cls.py`      | Depth-influenced classification pipeline |
| `pipeline_C_depth2cloud.py`    | 9D point cloud generation |
| `pipeline_C_train.py`          | Train segmentation model |
| `pipeline_C_test.py`           | Test segmentation on CW2 |
| `pipeline_C_test_realsense.py` | Test segmentation on real Realsense data |
| `pointnet2_cls.py`             | PointNet++ classification model |
| `pointnet2_seg.py`             | PointNet++ segmentation model |

## Example Commands

---

### 🔹 Pipeline A - Binary Classification of Table

#### ▶ Train on CW2 dataset
```
python src/pipeline_A_pointnet2_train.py 
```
set `data="CW2"`
#### ▶ Test on CW2 dataset
```
python src/pipeline_A_pointnet2_test.py 
```
set `data="CW2"`
#### ▶ Test on Realsense dataset
```
python src/pipeline_A_pointnet2_test.py 
```
set `data="Realsense"`

---

### 🔹 Pipeline B - Depth-Aware Classification

#### ▶ Run training with depth refinement (CW2)
```
python src/pipeline_B_depth_cls.py 
```
set `data="CW2"`
#### ▶ Run testing on Realsense
```
python src/pipeline_B_depth_cls.py 
```
set `data="Realsense"`

---

### Pipeline C - Semantic Segmentation

#### ▶ Generate point cloud (9D) from RGB-D
```
python src/pipeline_C_depth2cloud.py 
```
#### ▶ Train segmentation model
```
python src/pipeline_C_train.py 
```
set `data="CW2"`
#### ▶ Test on CW2 dataset
```
python src/pipeline_C_test.py 
```
set `data="CW2"`
#### ▶ Test on Realsense
```
python src/pipeline_C_test_realsense.py 
```
set `data="Realsense"`