# DefectNet
DefectNet: Towards Fast and Effective Defect Detection.

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

| Model            | mAP    | F1-score   | ATT(MS)  | ATT_normal(MS)    | ATT_defective(MS)   | Remark |
|------------|:--------:|:--------:|:------------:|:-----------------|:-----------------|:-----------------|
|one-model         | 0.198 | 0.859 | 67.768   |     68.281    |    67.142  |cascade_rcnn_r50_fpn_1x|
|two-model_small| 0.168 | 0.841 | **40.769**   |      64.404      |      11.934     |r50_e52+cascade_rcnn_r50_fpn_1x| 
|two-model_large| 0.194 | 0.892 | 61.225      |      87.360      |      29.340      |r50_e12+cascade_rcnn_r50_fpn_1x| 
|defectnet_const| **0.214** | **0.931** | 47.979    |    65.573       |     26.513      |defectnet_const+cascade_rcnn_r50_fpn_1x| 
|defectnet_linear| 0.152 | 0.879 | 43.660   |    59.611   |     24.199    |defectnet_linear+cascade_rcnn_r50_fpn_1x| 
|defectnet_inverse| 0.190 | 0.925 | 47.855   |      65.300        |   26.572      |defectnet_inverse+cascade_rcnn_r50_fpn_1x| 
|defectnet_exponential| 0.191 | 0.926 | 47.500  |      65.032    |     26.110       |defectnet_exponential+cascade_rcnn_r50_fpn_1x| 

- **Bottle defect dataset**

| Model            | mAP    | F1-score   | ATT(MS)  | ATT_normal(MS)    | ATT_defective(MS)      | Remark |
|------------|:--------:|:--------:|:------------:|:-----------------|:-----------------|:-----------------|
|one-model         | 0.487 | 0.872 | 76.509   |   76.614  |     76.194    |cascade_rcnn_r50_fpn_1x|
|two-model_small| 0.479 | 0.890 | 65.913   |  78.772  |   27.107  |r50_e52+cascade_rcnn_r50_fpn_1x| 
|two-model_large| 0.481 | 0.924 | 82.801 |   98.653   |   34.962     |r50_e12+cascade_rcnn_r50_fpn_1x| 
|defectnet_const| **0.491** | 0.930 | 63.625  |  74.671  |  30.291  |defectnet_const+cascade_rcnn_r50_fpn_1x| 
|defectnet_linear| 0.472 | 0.918 | 64.025   |   74.745    |   31.676    |defectnet_linear+cascade_rcnn_r50_fpn_1x| 
|defectnet_inverse| 0.487 | **0.934** | 63.463   |  74.622     |    29.787     |defectnet_inverse+cascade_rcnn_r50_fpn_1x| 
|defectnet_exponential| 0.486 | 0.930 | **62.860**      |   74.080  |  29.001 |defectnet_exponential+cascade_rcnn_r50_fpn_1x| 

Test on GTX 1080Ti GPU: 

- **Bottle defect dataset**

| Model            | mAP    | F1-score   | ATT(MS)    | Remark |
|------------|:--------:|:--------:|:------------:|:-----------------|
|one-model         |  0.490 |  0.870 |  81.186         |cascade_rcnn_r50_fpn_1x|
|two-model_small|  0.483 |  0.872 | **65.079**     |r50_e52+cascade_rcnn_r50_fpn_1x| 
|two-model_large| 0.485 |  0.917 | 94.617     |r50_e12+cascade_rcnn_r50_fpn_1x| 
|defectnet_const| 0.491 |  0.938 |  65.646 |defectnet_const+cascade_rcnn_r50_fpn_1x| 
|defectnet_linear|  0.471 |  0.926 | 68.651    |defectnet_linear+cascade_rcnn_r50_fpn_1x| 
|defectnet_inverse| **0.495** | **0.942** |  67.850     |defectnet_inverse+cascade_rcnn_r50_fpn_1x| 
|defectnet_exponential|  0.489 | 0.934 | 67.889 |defectnet_exponential+cascade_rcnn_r50_fpn_1x| 

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

None

## Reference

- **MMDetection**

    https://github.com/open-mmlab/mmdetection
