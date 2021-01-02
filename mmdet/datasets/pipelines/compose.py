import collections

from mmdet.utils import build_from_cfg
from ..registry import PIPELINES


@PIPELINES.register_module
class Compose(object):

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        last_ind = 0
        for i, t in enumerate(self.transforms):
            last_ind = i
            data = t(data)
            if isinstance(data, list):
                break
            if data is None:
                return None
        if isinstance(data, list):
            data, data_list = [], data
            for idx, d in enumerate(data_list):
                for i, t in enumerate(self.transforms[last_ind + 1:]):
                    d = t(d)
                    if d is None:
                        break
                data.append(d)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
