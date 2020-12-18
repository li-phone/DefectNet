# Bag Tricks

## Train and Test

    cd tools
    # ln -s {data directory} data 
    python bag_tricks_demo.py
    # wait...

## Results

Test on GTX 2080Ti GPU: 

#### **Fabric defect dataset**

- **Multi-scale Training**

|Input Size            | mAP    | AP@.5   | 
|:------------:|:--------:|:--------:|
|(1333, 800)                         |  0.203 | 0.434 | 
|(1223, 500)                         |  0.199 | 0.427 | 
|(2446, 1000)                        |  0.179 | 0.380 | 
|*Value*({(1223, 500), (1333, 800), (2446, 1000)})| 0.210 | 0.448 |
|*Range*(([1223, 2446], [500, 1000]))| **0.225** | **0.472** |
|*Range*(([667, 2000], [400, 1200]))| 0.220 | 0.454 |

<!-- 
|*Value*({(1223, 500), (1333, 800)})*| 0.213 | 0.447 |
|*Range*(([1223, 1333], [500, 800]))*| 0.220 | 0.455 | 
|*Range*(([612, 1835], [250, 750]))|  |  | 
-->

- **Dimension Clustering**

|k      | Anchor Ratio  |    Average IoU    |     mAP    | AP@.5   | 
|:------------:|:--------:|:--------:|:--------:|:--------:|
|3 | [0.21, 0.32, 1.83]       |     44.27     | 0.215 | 0.453 | 
|4 | [0.16, 0.30, 0.92, 3.62]  | 48.42 | 0.250| 0.525|
|5 | [0.05, 0.31, 0.92, 3.66, 8.04] | 53.93 | 0.272 | 0.543 |
|6 | [0.05, 0.19, 0.41, 1.00, 3.92, 8.07] | 56.64 | 0.266 | 0.542 |
|7 | [0.04, 0.18, 0.40, 0.77, 2.45, 5.25, 8.12] | 58.87 | 0.269 | 0.544 |
|8 | [0.04, 0.14, 0.29, 0.52, 0.85, 2.73, 5.77, 8.14] | 60.90 | **0.274** | 0.546 |
|9 | [0.04, 0.13, 0.28, 0.48, 0.73, 0.88, 2.91, 5.91, 8.17]| 62.21 | **0.274** | **0.549** |
|10| [0.04, 0.12, 0.20, 0.40, 0.50, 0.82, 1.30, 3.82, 5.93, 8.08]| **63.67** | 0.271 | 0.540 |

- **Soft-NMS With IoU Threshold**

|IoU Threshold            | mAP    | AP@.5   | 
|:------------:|:--------:|:--------:|
| 0.1 | 0.208 | **0.437** | 
| 0.2 | 0.208 | **0.437** | 
| 0.3 | 0.208 | 0.436 | 
| 0.4 | **0.209** | **0.437** | 
| 0.5 | 0.208 | **0.437** | 
| 0.6 | 0.207 | 0.434 | 
| 0.7 | 0.206 | 0.429 | 
| 0.8 | 0.202 | 0.414 | 
| 0.9 | 0.187 | 0.368 | 

- **Stacking Tricks**

|IoU Threshold            | mAP    | AP@.5   | 
|:------------|:--------:|:--------:|
| Baseline |   0.203 | 0.434 | 
| Baseline+Multi-scale | 0.225 | 0.472 | 
| Baseline+Multi-scale+Dimension Clustering | 0.295 | 0.570 | 
| Baseline+Multi-scale+Dimension Clustering+Soft-NMS | **0.301** | **0.573** | 

- **DefectNet + Stacking Tricks**

|IoU Threshold            | mAP    | AP@.5   |  F1-score |  ATT | ATT_d | ATT_n |
|:------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| baseline |   0.200 | 0.427 | 0.859 | 67.856 | 68.265 | 67.357 |
| defectnet_const |   0.197| 0.423 | 0.934 | 50.391 |69.017 | 27.668 |
| defectnet_linear | 0.155 | 0.343 | 0.923 | 49.491 | 67.712 | 27.261 |
| defectnet_inverse | 0.183 | 0.395 | 0.933 | 49.760 | 68.250 | 27.203 |
| defectnet_exponential | 0.188 | 0.404 | 0.921 | 49.021 | 67.108 | 26.955 |
| defectnet_const+Multi-scale | 0.200 | 0.424 | 0.932 | 49.243 | 67.857 | 26.535 |
| defectnet_linear+Multi-scale |0.172|0.383|0.931|49.424|67.968|26.801|
| defectnet_inverse+Multi-scale |0.204|0.434|0.940|49.456|68.166|26.630|
| defectnet_exponential+Multi-scale | 0.201|0.425|0.926|49.248|67.472|27.014|
| defectnet_const+Multi-scale+Dimension Clustering | 0.280 |0.548|0.940|48.845|67.785|25.739|
| defectnet_linear+Multi-scale+Dimension Clustering | 0.251|0.503|0.934|50.684|70.367|26.670|
| defectnet_inverse+Multi-scale+Dimension Clustering |0.282|0.543|0.940|50.632|70.358|26.567|
| defectnet_exponential+Multi-scale+Dimension Clustering |0.284|0.550|0.935|50.633|70.086|26.901|
| defectnet_const+Multi-scale+Dimension Clustering+Soft-NMS |0.287|0.549|0.940|53.348|76.071|25.625| 
| defectnet_linear+Multi-scale+Dimension Clustering+Soft-NMS | 0.257|0.505|0.934|53.463|76.042|25.917|
| defectnet_inverse+Multi-scale+Dimension Clustering+Soft-NMS | 0.288|0.544|0.940|53.388|76.079|25.705|
| defectnet_exponential+Multi-scale+Dimension Clustering+Soft-NMS | 0.290|0.552|0.935|53.416|75.703|26.227|

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

None

## Reference

- **MMDetection**

    https://github.com/open-mmlab/mmdetection
