# install pycocotools
# shellcheck disable=SC2164
# install coco
cd dependence/cocoapi/PythonAPI
python setup.py develop

# install mmcv
cd ../../
cd mmcv/
python setup.py develop

# install mmdet
cd ../../
python setup.py develop
