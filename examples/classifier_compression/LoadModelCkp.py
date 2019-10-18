import os
import torch

fileName = os.path.join('checkpoint', 'checkpoint_ch8_ver2.pth')
fileNameNew = os.path.join('checkpoint', 'checkpoint_ch8_ver2_trim.pth')
model_q = torch.load(fileName)
model_q['new_state_dict'] = {}
model_q['quant_state_dict'] = {}
model_q['dequant_state_dict'] = {}
_my_dict = {}
_my_dict_tmp = {}

def trim_prefix():
    for key, data in model_q.items():
        items = str(key).split('.')
        items.remove('module')
        new_key = str('.'.join(items))
        model_q['new_state_dict'][new_key] = data

def quant_weight():
    for key, data in model_q['state_dict'].items():
        items = str(key).split('.')
        items.remove(items[len(items) - 1])
        new_key = str('.'.join(items))
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

    # for key, data in _my_dict.items():
    #    print(key, _my_dict[key]['scale'], _my_dict[key]['zero_point'])

    for key, data in _my_dict.items():
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
            _my_dict_tmp[key] = model_q['dequant_state_dict'][key]
            _my_dict_tmp[key+'_quant'] = model_q['quant_state_dict'][key]
    state = {
        'net': _my_dict_tmp,
        'acc': 0,
        'epoch': 0
    }
    torch.save(state, fileNameNew)

def _quantValue(input, scale, zero_point):
    return torch.round(scale * input - zero_point)

def _dequantValue(input, scale, zero_point):
    return (input + zero_point) / scale

if __name__ == '__main__':
    trim_prefix()
    # quant_weight()
    # replace_with_dequant_value_and_save()
    torch.save(model_q, fileNameNew)

