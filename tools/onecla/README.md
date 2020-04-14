# OneCla
"OneClassification" is one label image classification.

#### Structure

* configs: for dataset choice configuration
* work_dirs: for the save directory of training result

#### Get Started
    
1. set your data set name, root, annotations, image directory
    
       
    data_name = 'your_data_set_name'
    
    data_root = 'your_data_set_root'
    
    raw_train_path = 'your_data_annotations'
    
    ann_file = 'your_data_annotations'
    
    img_prefix = 'your_image_directory'
        
        
2. start training

    
    python train.py


#### Test Result

    dataset name    score
    pneumonia       78.0543
    dofu_tomato     96.9436


#### License
This project is licensed under the MIT license.
