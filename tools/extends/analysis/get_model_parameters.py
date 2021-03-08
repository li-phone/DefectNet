# get model parameters
import torch
import numpy as np
import torch
import numpy as np
import torch
from torch.autograd import Variable
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
from mmdet.apis.inference import LoadImage
from mmdet.apis.inference import init_detector, inference_detector
from torchstat import stat
from torchsummary import summary


def get_parameter_number(net):
    total = dict(bk=0, sub=0, dfn=0)
    trainable = dict(bk=0, sub=0, dfn=0)
    for k, v in net.items():
        if 'backbone.fc' in k:
            total['dfn'] += v.numel()
            if v.requires_grad:
                trainable['dfn'] += v.numel()
        elif 'backbone.' in k:
            total['bk'] += v.numel()
            if v.requires_grad:
                trainable['bk'] += v.numel()
        else:
            total['sub'] += v.numel()
            if v.requires_grad:
                trainable['sub'] += v.numel()
    return {'Total': total, 'Trainable': trainable}


def count_params(model, input_size=224):
    # param_sum = 0
    with open('models.txt', 'w') as fm:
        fm.write(str(model))

    # 计算模型的计算量
    calc_flops(model, input_size)

    # 计算模型的参数总量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('The network has {} params.'.format(params))

    # 计算模型的计算量


def fake_data(model, input=(3, 800, 1333)):
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    if isinstance(input, (list, tuple)):
        img = np.random.random(input[::-1])
    elif isinstance(input, str):
        img = input
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data


def calc_flops(model, input_size=800, USE_GPU=True):
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    if isinstance(input_size, list) or isinstance(input_size, tuple):
        input = fake_data(model, input_size)
    else:
        if '1.' in torch.__version__:
            if USE_GPU:
                input = torch.cuda.FloatTensor(torch.rand(2, 3, input_size, input_size).cuda())
            else:
                input = torch.FloatTensor(torch.rand(2, 3, input_size, input_size))
        else:
            input = Variable(torch.rand(2, 3, input_size, input_size), requires_grad=True)

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    if isinstance(result, int):
        result = 0
    else:
        result = 1
    print('[%d]  + Number of FLOPs: %.2fM' % (result, total_flops / 1e6 / 2))


def main():
    m = torch.load(
        '../../../work_dirs/fabric/defectnet_inverse_cascade_rcnn_r50_fpn_1x/epoch_12.pth')
    print(get_parameter_number(m['state_dict']))

    model = init_detector(
        '../../../configs/fabric/defectnet_inverse_cascade_rcnn_r50_fpn_1x.py',
        '../../../work_dirs/fabric/defectnet_inverse_cascade_rcnn_r50_fpn_1x/epoch_12.pth')

    # count_params(model, (3, 800, 1333))
    stat(model, fake_data(model, '../../../demo/normal_demo.jpg'))
    stat(model, fake_data(model, '../../../demo/defective_demo.jpg'))


if __name__ == '__main__':
    main()
