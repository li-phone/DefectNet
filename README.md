# DefectNet
DefectNet: Towards Fast and Efficient Defect Detection.

## Installation
mirror: [https://gitee.com/liphone/DefectNet.git](https://gitee.com/liphone/DefectNet.git)


    git clone https://github.com/li-phone/DefectNet.git
    cd DefectNet
    pip install -r requirements.txt
    bash setup.sh
    
## Prepare Dataset

- **Fabric defect dataset**

|            | Total    | Normal   | Defective    | Normal Proportion |
|------------|:--------:|:--------:|:------------:|:-----------------:|
| all        | 8325     | 3663     | 4662         | 0.44              |
| train      | 6660     | 2913     | 3747         | 0.44              | 
| test       | 1665     | 750      | 915          | 0.45              |

- **Bottle defect dataset** is available at: [https://pan.baidu.com/s/1RH0-hqGOWa-sgbAUdRQmGg](https://pan.baidu.com/s/1RH0-hqGOWa-sgbAUdRQmGg)，提取码：yd4b 

|            | Total    | Normal   | Defective    | Normal Proportion |
|------------|:--------:|:--------:|:------------:|:-----------------:|
| all        | 4516     | 1146     | 3370         | 0.25              |
| train      | 3612     | 921      | 2691         | 0.25              | 
| test       | 904      | 225      | 679          | 0.25              |
    
## Train and Test

    cd tools
    # ln -s {data directory} data 
    python demo.py
    # wait...

## Results

Test on GTX 2080Ti GPU: 

- **Fabric defect dataset**

| Model            | mAP    | F1-score   | ATT(MS)    | Remark |
|------------|:--------:|:--------:|:------------:|:-----------------|
|one-model         | 0.198 | 0.859 | 67.768         |cascade_rcnn_r50_fpn_1x|
|two-model_small| 0.168 | 0.841 | **40.769**     |r50_e52+cascade_rcnn_r50_fpn_1x| 
|two-model_large| 0.194 | 0.892 | 61.225     |r50_e12+cascade_rcnn_r50_fpn_1x| 
|defectnet_const| **0.214** | **0.931** | 47.979     |defectnet_const+cascade_rcnn_r50_fpn_1x| 
|defectnet_linear| 0.152 | 0.879 | 43.660    |defectnet_linear+cascade_rcnn_r50_fpn_1x| 
|defectnet_inverse| 0.190 | 0.925 | 47.855     |defectnet_inverse+cascade_rcnn_r50_fpn_1x| 
|defectnet_exponential| 0.191 | 0.926 | 47.500     |defectnet_exponential+cascade_rcnn_r50_fpn_1x| 

- **Bottle defect dataset**

| Model            | mAP    | F1-score   | ATT(MS)    | Remark |
|------------|:--------:|:--------:|:------------:|:-----------------|
|one-model         | 0.487 | 0.872 | 76.509         |cascade_rcnn_r50_fpn_1x|
|two-model_small| 0.479 | 0.890 | 65.913     |r50_e52+cascade_rcnn_r50_fpn_1x| 
|two-model_large| 0.481 | 0.924 | 82.801     |r50_e12+cascade_rcnn_r50_fpn_1x| 
|defectnet_const| **0.491** | 0.930 | 63.625 |defectnet_const+cascade_rcnn_r50_fpn_1x| 
|defectnet_linear| 0.472 | 0.918 | 64.025    |defectnet_linear+cascade_rcnn_r50_fpn_1x| 
|defectnet_inverse| 0.487 | **0.934** | 63.463     |defectnet_inverse+cascade_rcnn_r50_fpn_1x| 
|defectnet_exponential| 0.486 | 0.930 | **62.860** |defectnet_exponential+cascade_rcnn_r50_fpn_1x| 

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

None

## Reference

- **MMDetection**

    https://github.com/open-mmlab/mmdetection
