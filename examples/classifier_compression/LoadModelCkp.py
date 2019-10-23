import os
import torch

fileName = os.path.join('checkpoint', '20191022_resnet10_fp32_-128_127_224x224_resize', 'checkpoint_retrain_5.pth')
fileNameNew = os.path.join('checkpoint', '20191023_resnet10_fp32_fused_-128_127_224x224_resize', 'checkpoint_fuse.pth')
fileNameDump = os.path.join('checkpoint', '20191022_resnet10_quant8_symmetric_-128_127_224x224_resize', 'quant_info.txt')

model_q = torch.load(fileName)
model_q['trim_state_dict'] = {}
model_q['quant_state_dict'] = {}
model_q['dequant_state_dict'] = {}
_my_dict = {}
_my_dict_tmp = {}
_my_dict_fuse = {}

def trim_prefix():
    for key, data in model_q.items():
        items = str(key).split('.')
        items.remove('module')
        new_key = str('.'.join(items))
        model_q['trim_state_dict'][new_key] = data

def dump_quant_info():
    with open(fileNameDump, "w") as text_file:
        for key, data in _my_dict.items():
            text_file.writelines('{0}\n'.format(key))
            if ('conv' in key or 'fc' in key or 'downsample.0' in key):
                text_file.writelines('w: {0} {1}\n'.format(_my_dict[key]['scale'][0].item(), _my_dict[key]['zero_point'][0].item()))
            else:
                text_file.writelines('o: {0} {1}\n'.format(_my_dict[key]['scale'][0].item(), _my_dict[key]['zero_point'][0].item()))
    text_file.close()

def parse_quant_info():
    for key, data in model_q['net']['state_dict'].items():
        items = str(key).split('.')
        items.remove(items[len(items) - 1])
        new_key = str('.'.join(items))
        # weight quant #
        if(('conv' in key or 'fc' in key or 'downsample.0' in key) and '.weight' in key and '.weight_scale' not in key and '.weight_zero_point' not in key):
            if(new_key not in _my_dict):
                _my_dict[new_key] = { 'input': [],
                                      'scale': [],
                                      'zero_point': [] }
                _my_dict[new_key]['input'].append(data)
        elif(('conv' in key or 'fc' in key or 'downsample.0' in key) and '.weight_scale' in key):
            if(new_key in _my_dict and len(_my_dict[new_key]['scale']) == 0):
                _my_dict[new_key]['scale'].append(data)
        elif (('conv' in key or 'fc' in key or 'downsample.0' in key) and '.weight_zero_point' in key):
            if(new_key in _my_dict and len(_my_dict[new_key]['zero_point']) == 0):
                _my_dict[new_key]['zero_point'].append(data)
        elif('fake_q' in key and '.scale' in key): # activation quant #
            if (new_key not in _my_dict):
                _my_dict[new_key] = {'input': [],
                                     'scale': [],
                                     'zero_point': []}
                _my_dict[new_key]['input'].append(new_key)
            _my_dict[new_key]['scale'].append(data)
        elif('fake_q' in key and '.zero_point' in key):
            _my_dict[new_key]['zero_point'].append(data)
        elif ('bn' in key and '.weight' in key):
            if (new_key not in _my_dict):
                _my_dict[new_key] = {'gamma': [],
                                     'beta': [],
                                     'mean': [],
                                     'var': []}
            _my_dict[new_key]['gamma'].append(data)
        elif ('bn' in key and '.bias' in key):
            _my_dict[new_key]['beta'].append(data)
        elif ('bn' in key and '.running_mean' in key):
            _my_dict[new_key]['mean'].append(data)
        elif ('bn' in key and '.running_var' in key):
            _my_dict[new_key]['var'].append(data)

    # for key, data in _my_dict.items():
    #    print(key, _my_dict[key]['scale'], _my_dict[key]['zero_point'])


def quant_dequant_weight():
    for key, data in _my_dict.items():
        if('conv' in key or 'fc' in key or 'downsample.0' in key):
            model_q['quant_state_dict'][key+'.weight'] = _quantValue(_my_dict[key]['input'][0],\
                                                                     _my_dict[key]['scale'][0],\
                                                                     _my_dict[key]['zero_point'][0])
            model_q['dequant_state_dict'][key + '.weight'] = _dequantValue(model_q['quant_state_dict'][key+'.weight'], \
                                                                           _my_dict[key]['scale'][0],\
                                                                           _my_dict[key]['zero_point'][0])

    # for key, data in model_q['quant_state_dict'].items():
    #    print(key, data)

def replace_with_dequant_value_and_save():
    for key, data in model_q['state_dict'].items():
        if(key not in model_q['dequant_state_dict']):
            _my_dict_tmp[key] = data
        else:
            _my_dict_tmp[key+'_original'] = data
            _my_dict_tmp[key] = model_q['dequant_state_dict'][key]
            _my_dict_tmp[key+'_quant'] = model_q['quant_state_dict'][key]
    torch.save(_my_dict_tmp, fileNameNew)

def _quantValue(input, scale, zero_point):
    return torch.round(scale * input - zero_point)

def _dequantValue(input, scale, zero_point):
    return (input + zero_point) / scale

def fuse_conv_and_bn():
    for key, data in _my_dict.items():
        print(key, data)

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + 1E-7)
    gamma = bn.weight
    beta = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = ((b - mean)/var_sqrt) * gamma + beta
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

if __name__ == '__main__':
    # trim_prefix()
    # torch.save(model_q, fileNameNew)
    parse_quant_info()
    fuse_conv_and_bn()
    # quant_dequant_weight()()
    # replace_with_dequant_value_and_save()


