import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
# import matplotlib.pyplot as plt
import collections
import os

import models.crnn as crnn


model_path = '/data_2/project_2021/crnn/crnn.pytorch-master/data/netCRNN_28500_1.pth'
dir_img = "/data_1/everyday/xian/20210308_crnn/IIIT5K/test/"


alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

nclass = len(alphabet) + 1

model = crnn.CRNN(32, 1, nclass, 256)#model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()

#
# for m in model.state_dict().keys():
#      print("==:: ", m)

load_model_ = torch.load(model_path)
# for k, v in load_model_.items():
#     print(k,"  ::shape",v.shape)

state_dict_rename = collections.OrderedDict()
for k, v in load_model_.items():
    name = k[7:] # remove `module.`
    state_dict_rename[name] = v


print('loading pretrained model from %s' % model_path)
model.load_state_dict(state_dict_rename)



converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))



list_img = os.listdir(dir_img)
for cnt,img_name in enumerate(list_img):
    print(cnt,img_name)
    path_img = dir_img + img_name

    image = Image.open(path_img).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('%-20s => %-20s' % (raw_pred, sim_pred))
    print("\n"*2)

    # image_show = Image.open(path_img)
    # plt.figure("show")
    # plt.imshow(image_show)
    # plt.show()





# list_img = os.listdir(dir_img)
# for cnt,img_name in enumerate(list_img):
#     print(cnt,img_name)
#     path_img = dir_img + img_name
#
#     image_show = Image.open(path_img)
#     # image_show.show()
#
#     plt.figure("dog")
#     plt.imshow(image_show)
#     plt.show()





