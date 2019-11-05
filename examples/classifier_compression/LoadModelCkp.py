import os
import sys
import torch
import struct
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.\\')))
from KeyMap import *

fileName = os.path.join('checkpoint', '20191104_resnet10_quant_ch8_224x', 'checkpoint_224x_fuse_b100.pth')
# fileName = os.path.join('checkpoint', '20191029_resnet10_quant8_fused_symm_-128_127_224x224_test', 'checkpoint_8626.pth')
fileNameNew = os.path.join('checkpoint', '20191024_resnet10_quant8_fused_sym_-128_127_224x224_resize', 'checkpoint_dequant_4.pth')

model_q = torch.load(fileName)

_my_dict = {}
_my_dict_tmp = {}
_my_dict_fuse = {}

def get_clamp_limit(bit_size=8, signed=True):
    signed_limit = 2 ** (bit_size - 1)
    # if(bit_size == 32):
    #     _max = 2147483500
    #     return (-signed_limit, 2147483500) if signed else (0, 2 * signed_limit - 1)
    return (-signed_limit, signed_limit - 1) if signed else (0, 2 * signed_limit - 1)

def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)

def trim_prefix():
    model_q['trim_state_dict'] = {}
    for key, data in model_q.items():
        items = str(key).split('.')
        items.remove('module')
        new_key = str('.'.join(items))
        model_q['trim_state_dict'][new_key] = data

def parse_quant_info():
    for key, data in model_q.items():
        items = str(key).split('.')
        items.remove(items[len(items) - 1])
        new_key = str('.'.join(items))
        # weight quant #
        # if(('conv' in key or 'fc' in key or 'downsample' in key or 'downsample.0' in key or 'fuse' in key) \
        #         and '.weight' in key and '.weight_scale' not in key and '.weight_zero_point' not in key):
        #     if(new_key not in _my_dict):
        #         _my_dict[new_key] = { 'weight': [],
        #                               'w_scale': [],
        #                               'w_zero_point': [],
        #                               'bias': [],
        #                               'b_scale': [],
        #                               'b_zero_point': []}
        #     if (new_key in _my_dict and len(_my_dict[new_key]['weight']) == 0):
        #         _my_dict[new_key]['weight'].append(data)
        if(('conv' in key or 'fc' in key or 'downsample' in key or 'downsample.0' in key or 'fuse' in key) and '.weight_scale' in key):
            if (new_key not in _my_dict):
                _my_dict[new_key] = {'w_scale': [],
                                     'w_zero_point': [],
                                     'b_scale': [],
                                     'b_zero_point': []}
            if(new_key in _my_dict and len(_my_dict[new_key]['w_scale']) == 0):
                _my_dict[new_key]['w_scale'].append(data)
        elif (('conv' in key or 'fc' in key or 'downsample' in key or 'downsample.0' in key or 'fuse' in key) and '.weight_zero_point' in key):
            if(new_key in _my_dict and len(_my_dict[new_key]['w_zero_point']) == 0):
                _my_dict[new_key]['w_zero_point'].append(data)
        elif('fake_q' in key and '.scale' in key): # activation quant #
            if (new_key not in _my_dict):
                _my_dict[new_key] = {'input': [],
                                     'scale': [],
                                     'zero_point': []}
                _my_dict[new_key]['input'].append(new_key)
            _my_dict[new_key]['scale'].append(data)
        elif('fake_q' in key and '.zero_point' in key):
            _my_dict[new_key]['zero_point'].append(data)
        # elif (('conv' in key or 'fc' in key or 'downsample' in key or 'downsample.0' in key or 'fuse' in key) \
        #       and '.bias' in key and '.bias_scale' not in key and '.bias_zero_point' not in key):
        #     if (new_key in _my_dict and len(_my_dict[new_key]['bias']) == 0):
        #         _my_dict[new_key]['bias'].append(data)
        elif (('conv' in key or 'fc' in key or 'downsample' in key or 'downsample.0' in key or 'fuse' in key) and '.bias_scale' in key):
            if (new_key in _my_dict and len(_my_dict[new_key]['b_scale']) == 0):
                _my_dict[new_key]['b_scale'].append(data)
        elif (('conv' in key or 'fc' in key or 'downsample' in key or 'downsample.0' in key or 'fuse' in key) and '.bias_zero_point' in key):
            if(new_key in _my_dict and len(_my_dict[new_key]['b_zero_point']) == 0):
                _my_dict[new_key]['b_zero_point'].append(data)

    # for key, data in _my_dict.items():
    #    print(key, _my_dict[key]['scale'], _my_dict[key]['zero_point'])

def quant_dequant_weight():
    model_q['quant_state_dict'] = {}
    model_q['dequant_state_dict'] = {}
    for key, data in _my_dict.items():
        if('conv' in key or 'fc' in key or 'downsample' in key or 'downsample.0' in key or 'fuse' in key):
            _min, _max = get_clamp_limit(bit_size=8, signed=True)
            model_q['quant_state_dict'][key+'.weight'] = _quantValue(_my_dict[key]['weight'][0],\
                                                                     _my_dict[key]['w_scale'][0],\
                                                                     _my_dict[key]['w_zero_point'][0],\
                                                                     _min, _max)
            model_q['dequant_state_dict'][key + '.weight'] = _dequantValue(model_q['quant_state_dict'][key+'.weight'], \
                                                                           _my_dict[key]['w_scale'][0],\
                                                                           _my_dict[key]['w_zero_point'][0])
            if(len(_my_dict[key]['bias']) > 0):
                _min, _max = get_clamp_limit(bit_size=32, signed=True)
                model_q['quant_state_dict'][key + '.bias'] = _quantValue(_my_dict[key]['bias'][0], \
                                                                         _my_dict[key]['b_scale'][0], \
                                                                         _my_dict[key]['b_zero_point'][0],\
                                                                         _min, _max)
                model_q['dequant_state_dict'][key + '.bias'] = _dequantValue(model_q['quant_state_dict'][key + '.bias'], \
                                                                               _my_dict[key]['b_scale'][0], \
                                                                               _my_dict[key]['b_zero_point'][0])

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

def _quantValue(input, scale, zero_point, clamp_min, clamp_max):
    quant = torch.round(scale * input - zero_point)
    return clamp(quant, clamp_min, clamp_max, False)

def _dequantValue(input, scale, zero_point):
    return (input + zero_point) / scale

def _dump_value_struct_pack(input, outputFolder, name, mode):
    fileName = os.path.join('checkpoint', outputFolder, name+'.txt')
    t = torch.flatten(input)
    with open(fileName, "wb") as text_file:
        for o in range(len(t)):
            text_file.write( struct.pack(mode, t[o]) )
    text_file.close()

def _dump_value_numpy(input, outputFolder, name):
    fileName = os.path.join('checkpoint', outputFolder, name)
    inputToNumpy = input.cpu().numpy()
    np.save(fileName, inputToNumpy)

def dump_weight_and_bias_to_file():
    outputFolder = '20191028_resnet10_quant8_fused_symm_-128_127_224x224_test'
    type = 'weight'
    if(type == 'weight'):
        postfix = '.fp32'
        mode = 'f' ## fp
    elif(type == 'bias'):
        postfix = '.q32'
        mode = 'i' ## int
    else:
        postfix = '.q8'
        mode = 'b' ## char

    for key, data in key_map.items():
        # new_key = str('module.' + str(key) + '.' + type + '_quant')
        new_key = str('module.' + str(key) + '.' + type)
        if new_key in model_q:
            print(new_key)
            _dump_value_struct_pack(model_q[new_key], \
                                    outputFolder, \
                                    data + '.' + type + postfix, \
                                    mode)
            _dump_value_numpy(model_q[new_key], \
                              outputFolder, \
                              data + '.' + type + postfix)

def dump_scale_info(outputFolder, name):
    fileNameDump = os.path.join('checkpoint', outputFolder, name+'.txt')
    with open(fileNameDump, "w") as text_file:
        for key, data in key_ch8_map.items():
            weight_key = str('module.' + str(key) + '.weight_scale')
            bias_key = str('module.' + str(key) + '.bias_scale')
            image_key = str('module.' + str(key) + '.scale')
            if(weight_key in model_q):
                text_file.writelines('config w\n')
                text_file.writelines('{0} {1}\n'.format(data, torch.log2(model_q[weight_key]).item()))
            # if (bias_key in model_q):
            #     text_file.writelines('config b\n')
            #     text_file.writelines('{0} {1}\n'.format(data, torch.log2(model_q[bias_key]).item()))
            if (image_key in model_q):
                text_file.writelines('config o\n')
                text_file.writelines('{0} {1}\n'.format(data, torch.log2(model_q[image_key]).item()))
                text_file.writelines('\n')
    text_file.close()

if __name__ == '__main__':
    torch.set_printoptions(precision=9)

    if ('state_dict' in model_q):
         model_q = model_q['state_dict']

    # trim_prefix()
    # torch.save(model_q, fileNameNew)
    parse_quant_info()
    # quant_dequant_weight()
    # replace_with_dequant_value_and_save()

    outputFolder = '20191104_resnet10_quant_ch8_224x'
    name = 'scale_QAT_shift'
    dump_scale_info(outputFolder, name)

    # name = '2.maxpooling.activation.npy'
    # fileName1 = os.path.join('checkpoint', \
    #                         '20191030_resnet10_fp32_fused_220x220', \
    #                         '20191030_pytorch_batch1_img2_hw_data', \
    #                         name)
    # fileName2 = os.path.join('checkpoint', \
    #                          '20191031_resnet10_fp32_fused_220x220', \
    #                          '20191031_pytorch_input_batch100_0to9', \
    #                          'input.activation.int8.0.npy')
    # try:
    #     tmpNpy1 = np.load(fileName1)
    #     tmpNpy2 = np.load(fileName2)
    #     print(np.array_equal(tmpNpy1, tmpNpy2))
    #     print('max pooling')
    #     print(tmpNpy2)

    #     print('res1_input')
    #     print(tmpNpy1)

    # except IOError as e:
    #     print('[Error] no such file {0}'.format(name))

    # for name in std_names:
    #     fileName = os.path.join('checkpoint', \
    #                             '20191028_resnet10_quant8_fused_symm_-128_127_224x224_test', \
    #                             '2.'+name+'.bias.npy')
    #     try:
    #         tmpNpy = np.load(fileName)
    #         # print(tmpNpy)
    #     except IOError as e:
    #         print('[Error] no such file {0}'.format('2.'+name+'.bias.npy'))

    #     _biasTensor = torch.from_numpy(tmpNpy).float().cuda()

    #     for key, data in key_map.items():
    #         if(data == name and 'module.'+key in _my_dict):
    #             if('b_scale' in _my_dict['module.'+key]):
    #                 _biasScale = _my_dict['module.'+key]['b_scale'][0]
    #                 print(name, _biasScale)
    #                 _min, _max = get_clamp_limit(bit_size=32, signed=True)
    #                 _biasQuant = _quantValue(_biasTensor, \
    #                                          _biasScale, \
    #                                          0, \
    #                                          _min, _max)
    #                 _dump_value_numpy(_biasQuant, outputFolder, '2.'+name+'.bias.int32.npy')



