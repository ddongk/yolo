# def parse_cfg(cfgfile):
#     '''
#     cfgfile : cfgfile path, e.g. ./cfg/yolov3.cfg
#     return : list of blocks. each block describes a block in the neural network to be built.
#     '''
#     blocks = []
#     fp = open(cfgfile, 'r')
#     block = None
#     line = fp.readline()
#     while line != '':
#         line = line.rstrip()
#         if line == '' or line[0] == '#':
#             line = fp.readline()
#             continue
#         elif line[0] == '[':
#             if block:
#                 blocks.append(block)
#             block = dict()
#             block['type'] = line.lstrip('[').rstrip(']')
#             # set default value : batch => 0
#             if block['type'] == 'convolutional':
#                 block['batch_normalize'] = 0
#         else:
#             key, value = line.split('=')
#             key = key.strip()
#             if key == 'type':
#                 key = '_type'
#             value = value.strip()
#             block[key] = value
#         line = fp.readline()

#     if block:
#         blocks.append(block)
#     fp.close()
#     return blocks


def parse_cfg(cfgfile):
    '''
    cfgfile : cfgfile path, e.g. ./cfg/yolov3.cfg
    return : list of blocks. each block describes a block in the neural network to be built.
    '''
    fp = open(cfgfile, 'r')
    lines = fp.readlines()
    fp.close()

    lines = [line.strip() for line in lines if line.strip()]  # 공백 제거
    lines = [line for line in lines if line[0] != '#']  # 주석 제거

    blocks = []
    block = {}
    for line in lines:
        if line[0] == '[':
            if block:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1]
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks[0], blocks[1:]


if __name__ == "__main__":
    pass