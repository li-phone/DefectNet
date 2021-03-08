from batch_train import BatchTrain
from onecla.batch_train import main as first_model_train


def models_test(test_status=60, data_type="fabric", only_two_model=None):
    # test different score threshold of one-model method without background
    one_model = BatchTrain(cfg_path='../configs/{}/faster_rcnn_r50_fpn_1x_1_model.py'.format(data_type),
                           data_mode='test', train_sleep_time=0, test_sleep_time=test_status)

    # test two-model method using one-model method without background as second model
    one_model.two_model_test(data_type)


def models_train(test_status=-10, data_type="fabric", const_weights=None):
    # train for one-model method without background
    BatchTrain(cfg_path='../configs/{}/faster_rcnn_r50_fpn_1x_1_model.py'.format(data_type),
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()

    # train first model of two-model method
    first_model_train(dataset_name=data_type)

    # train constant loss weight of Defect Net method
    # train for finding best constant defect finding network loss weight
    BatchTrain(cfg_path='../configs/{}/faster_rcnn_r50_fpn_1x_defectnet_const.py'.format(data_type),
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).find_best_constant_loss_weight(
        const_weights)

    # train variable loss weight of Defect Net method
    # train for exponent defect finding network loss weight
    BatchTrain(cfg_path='../configs/{}/faster_rcnn_r50_fpn_1x_defectnet_exp.py'.format(data_type),
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()
    #
    # train for inverse defect finding network loss weight
    BatchTrain(cfg_path='../configs/{}/faster_rcnn_r50_fpn_1x_defectnet_inv.py'.format(data_type),
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()
    #
    # train for linear defect finding network loss weight
    BatchTrain(cfg_path='../configs/{}/faster_rcnn_r50_fpn_1x_defectnet_lin.py'.format(data_type),
               data_mode='test', train_sleep_time=0, test_sleep_time=test_status).common_train()
    pass


def main():
    # # # train fabric dataset
    # models_train(data_type="fabric", const_weights=[0.15])
    # # # test fabric dataset
    # models_train(test_status=60 * 1, data_type="fabric")
    # # models_test(test_status=60 * 1, data_type="fabric")

    # train bottle dataset
    # models_train(data_type="bottle", const_weights=[0.15])
    # test bottle dataset
    # models_train(test_status=60 * 1, data_type="bottle", const_weights=[0.15])
    models_test(test_status=60 * 1, data_type="bottle", only_two_model=True)
    pass


if __name__ == '__main__':
    main()
