# Singapore-Maritime-Dataset-Plus(SMD-Plus)

### * Notice

#### A paper on SMD-PLUS was submitted to the Journal of Marine Science and Engineering.
### Paper accepted 6/March/2022
### Object Detection and Classification Based on YOLO-V5 with Improved Maritime Dataset


Original SMD Download in below url

https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset

We provide video and ObjectGT for VIS_Onshore and VIS_Onboard.
ObjectGT is identical to the .mat file format provided by the original SMD.

Link to Download in below url

https://drive.google.com/file/d/1yokFvx_cJu-Fl5ti1wgF1WPHao_2kcKE/view?usp=sharing

More detail explanations will update soon.

### Docker

```Bash
docker build -t smd_plus .
```

```Bash
docker run -it --ipc=host --runtime=nvidia --gpus all -v ${HOME}/Documents/datasets/smd_plus/:/datasets/smd_plus/  -v ${HOME}/git/SMD-Plus:/SMD-Plus smd_plus
```

### Scripts

**Disclaimer:** most part of the scripts were vibe coded use it with caution

Convert the `.mat` files to `.csv` so it is human readable:
```Bash
python3 mat_to_csv.py
```
**Note:** Modify the `INPUT_DIR` variable to the path in your machine

Convert the `.mat` files to `.txt` annotations so it can be used directly with yolo:
```Bash
python3 mat_to_yolo_annotation.py
```
**Note:** Modify the `MAT_DIR` and `OUTPUT_DIR` variable to the path in your machine

Convert the videos to images so it can be used directly with yolo:
```Bash
python3 video_to_frames.py
```
**Note:** Modify the `VIDEO_DIR` and `OUT_ROOT` variable to the path in your machine