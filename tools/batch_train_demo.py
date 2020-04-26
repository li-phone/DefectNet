from batch_train import BatchTrain
from onecla.batch_train import main as first_model_train


def train_models(test_status=-10):
    # train for one-model method without background
    BatchTrain(cfg_path='../configs/bottle/one_model_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train for one-model method with background
    BatchTrain(cfg_path='../configs/bottle/one_model_bg_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train first model of two-model method
    first_model_train(dataset_name='bottle')

    # train constant loss weight of Defect Net method
    # train for finding best constant defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_constant_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).find_best_constant_loss_weight()

    # train variable loss weight of Defect Net method
    # train for exponent defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_exponent_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train for inverse defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_inverse_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train for linear defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_linear_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()


def test_models(test_status=-10):
    # test different score threshold of one-model method without background
    one_model = BatchTrain(cfg_path='../configs/bottle/one_model_cascade_rcnn_r50_fpn_1x.py',
                           data_mode='test', train_sleep_time=0, test_sleep_time=test_status)
    one_model.score_threshold_test()

    # test two-model method using one-model method without background as second model
    one_model.two_model_test()

    # test different normal images proportion for inverse defect finding network loss weight
    BatchTrain(cfg_path='../configs/bottle/defectnet_inverse_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).normal_proportion_test()


def train_fabric_models(test_status=-10):
    # train for one-model method without background
    BatchTrain(cfg_path='../configs/fabric/one_model_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train first model of two-model method
    first_model_train(dataset_name='fabric')

    # train constant loss weight of Defect Net method
    # train for finding best constant defect finding network loss weight
    BatchTrain(cfg_path='../configs/fabric/defectnet_constant_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train variable loss weight of Defect Net method
    # train for exponent defect finding network loss weight
    BatchTrain(cfg_path='../configs/fabric/defectnet_exponent_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train for inverse defect finding network loss weight
    BatchTrain(cfg_path='../configs/fabric/defectnet_inverse_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train for linear defect finding network loss weight
    BatchTrain(cfg_path='../configs/fabric/defectnet_linear_cascade_rcnn_r50_fpn_1x.py',
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()


def main():
    train_models()
    train_fabric_models()
    test_models()
    pass


if __name__ == '__main__':
    main()
