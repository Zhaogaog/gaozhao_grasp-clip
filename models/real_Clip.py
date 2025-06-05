import torch
# import clip
import os
import json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image
# from PIL import standard_transforms
import torchvision.transforms as transforms
from _clip import build_model, load_clip, tokenize
from unet import Up
import fusion
from _resnet import IdentityBlock, ConvBlock
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# from .model import build_model
# from .simple_tokenizer import SimpleTokenizer as _Tokenizer
# from clip import build_model

# print(clip.available_models())

class my_clip(nn.Module):
    def __init__(self, text=None, img_encoder_frozen = True, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), split='train'):
        super(my_clip, self).__init__()
        t1 = time.time()
        # self.text = torch.tensor([[22],[79]], dtype=torch.int32) if text is None else text
        # self.text = torch.tensor([[22], [79], [82], [51], [1], [24]], dtype=torch.int32) if text is None else text
        self.device =device
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        # self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        # self.model, self.preprocess = clip.load('RN50', self.device)
        # self.clip_rn50, self.preprocess = clip.load("RN50", device=self.device)
        # print('load_time:', time.time() - t1)
        t1 = time.time()
        self.img_encoder_frozen = img_encoder_frozen
        self.lang_fusion_type = 'mult'
        self.input_dim = 2048
        self.bilinear = True
        self.batchnorm = False
        self.up_factor = 2 if self.bilinear else 1
        self.output_dim = 16
        #text 的维度是batch_size*1 对应obj_id
        with open('/media/ama/data0/gz/graspnet/graspnet_real/obj_lists/all_category_' + split +'_dict.json','r') as f:
            obj_classes_dict = json.loads(f.read())
        self.obj_classes_dict = {v:k for k,v in obj_classes_dict.items()}
        # self.obj_classes = list(self.obj_classes_dict.values())
        # self.obj_classes = ['cracker box', 'sugar box', 'tomato soup can', 'mustard bottle',
        #                     'potted meat can', 'banana', 'bowl', 'mug',
        #                     'power drill', 'scissors', 'chips can', 'strawberry',
        #                     'apple', 'lemon', 'peach', 'pear',
        #                     'orange', 'plum', 'knife', 'phillips screwdriver',
        #                     'flat screwdriver', 'racquetball', 'b_cups', 'd_cups',
        #                     'a_toy_airplane', 'c_toy_airplane', 'd_toy_airplane', 'f_toy_airplane',
        #                     'h_toy_airplane', 'i_toy_airplane', 'j_toy_airplane', 'k_toy_airplane',
        #                     'padlock', 'dragon', 'secret repair', 'jvr cleansing foam',
        #                     'dabao wash soup', 'nzskincare mouth rinse', 'dabao sod', 'soap box',
        #                     'kispa cleanser', 'darlie toothpaste', 'nivea men oil control', 'baoke marker',
        #                     'hosjam', 'pitcher cap', 'dish', 'white mouse',
        #                     'camel', 'deer', 'zebra', 'large elephant',
        #                     'rhinocero', 'small elephant', 'monkey', 'giraffe',
        #                     'gorilla', 'weiquan', 'darlie box', 'soap',
        #                     'black mouse', 'dabao facewash', 'pantene', 'head shoulders supreme',
        #                     'thera med', 'dove', 'head shoulders care', 'lion',
        #                     'coconut juice box', 'hippo', 'tape', 'rubiks cube',
        #                     'peeler cover', 'peeler', 'ice cube mould', 'bar clamp',
        #                     'climbing hold', 'endstop holder', 'gearbox', 'mount1',
        #                     'mount2', 'nozzle', 'part1', 'part3',
        #                     'pawn', 'pipe connector', 'turbine housing', 'vase']

        self._load_clip()
        self._build_decoder()
        #输入的文本该如何处理
        # self.get_classes()
        # print('get_class_time:', time.time() - t1)
    def _load_clip(self):
        model, preprocess = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device).float()
        self.preprocess = preprocess
        del model

    def get_classes(self):
        sentence_list = []
        for i in range(self.text.size()[0]):
            index = int(self.text[i,0].detach().cpu().numpy())
            sentence_list.append('pick up the ' + self.obj_classes_dict[index].replace('_', ' '))
            #
            # if 'toy_airplane' in self.obj_classes[self.text[i,0]]:
            #     sentence_list.append('pick up the toy airplane')
            # elif 'cup' in self.obj_classes[self.text[i,0]]:
            #     sentence_list.append('pick up the cup')
            # elif 'mount' in self.obj_classes[self.text[i,0]]:
            #     sentence_list.append('pick up the mount')
            # elif 'part' in self.obj_classes[self.text[i,0]]:
            #     sentence_list.append('pick up the part')
            # elif 'elephant' in self.obj_classes[self.text[i,0]]:
            #     sentence_list.append('pick up the elephant')
            # else:
            #     sentence_list.append(f'pick up the {self.obj_classes[self.text[i,0]]}')
        self.sentences_list = sentence_list
    def _encode_text(self, text):
        self.text = text
        self.get_classes()
        with torch.no_grad():
            t1 = time.time()
            # print('load_time:', time.time() - t1)
            t1 = time.time()
            tokens = tokenize(self.sentences_list).to(self.device)
            # print('tokenize_time:', time.time() - t1)
            t1 = time.time()
            # text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)
            text_feat, text_emb= self.clip_rn50.encode_text_with_embeddings(tokens)
            # text_emb = text_feat.clone()
            # print('encode_text_time:', time.time() - t1)
        text_mask = torch.where(tokens == 0, tokens, torch.ones_like(tokens))  # [1, max_token_len]
        return text_feat, text_emb, text_mask
    def _encode_image(self, img):
        if self.img_encoder_frozen:
            with torch.no_grad():
                t1 = time.time()
                img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
                # print('encode_image_time:', time.time() - t1)
            return img_encoding, img_im
        else:
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
            # print('encode_image_time:', time.time() - t1)
            return img_encoding, img_im

    def _build_decoder(self):
        # language
        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=torch.div(self.input_dim, 2, rounding_mode='trunc'))
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=torch.div(self.input_dim, 4, rounding_mode='trunc'))
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=torch.div(self.input_dim, 8, rounding_mode='trunc'))

        self.proj_input_dim =  1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

        # vision
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.up1 = Up(2048, torch.div(1024, self.up_factor, rounding_mode='trunc'), self.bilinear)
        # self.lat_fusion1 = FusionConvLat(input_dim=1024 + 512, output_dim=512)

        self.up2 = Up(1024, torch.div(512, self.up_factor, rounding_mode='trunc'), self.bilinear)
        # self.lat_fusion2 = FusionConvLat(input_dim=512 + 256, output_dim=256)

        self.up3 = Up(512, torch.div(256, self.up_factor, rounding_mode='trunc'), self.bilinear)
        # self.lat_fusion3 = FusionConvLat(input_dim=256 + 128, output_dim=128)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        # self.lat_fusion4 = FusionConvLat(input_dim=128 + 64, output_dim=64)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        # self.lat_fusion5 = FusionConvLat(input_dim=64 + 32, output_dim=32)

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        # self.lat_fusion6 = FusionConvLat(input_dim=32 + 16, output_dim=16)

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, self.output_dim, kernel_size=1)
        # )

    def forward(self, img, text):
        # transform = transforms.ToPILImage()
        # img_ = transform(img)
        # numpy = img.cpu().numpy()
        # iamge1 = Image.fromarray(numpy)
        # image2 = transform(numpy)

        # img_ =
        # img = self.preprocess(img).to(self.device)

        in_type = img.dtype
        in_shape = img.shape
        # x = x[:, :3]  # select RGB
        img, img_im = self._encode_image(img)
        img = img.to(in_type)

        l_enc, l_emb, l_mask = self._encode_text(text)
        l_input = l_enc
        l_input = l_input.to(dtype=img.dtype)
        t1 = time.time()
        assert img.shape[1] == self.input_dim
        img = self.conv1(img)

        img = self.lang_fuser1(img, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        img = self.up1(img, img_im[-2])
        # x = self.lat_fusion1(x, lat[-6])

        img = self.lang_fuser2(img, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        img = self.up2(img, img_im[-3])

        img = self.lang_fuser3(img, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        img = self.up3(img, img_im[-4])

        img = self.layer1(img)
        # x = self.lat_fusion4(x, lat[-3])

        img = self.layer2(img)
        # x = self.lat_fusion5(x, lat[-2])

        img = self.layer3(img)
        # x = self.lat_fusion6(x, lat[-1])

        # img = self.conv2(img)

        # img = F.interpolate(img, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        img = F.interpolate(img, size=(720, 1280), mode='bilinear')
        # print('decode_image_time:', time.time() - t1)
        return img, l_input

            # self.clip_rn50 = clip.build_model(model.state_dict()).to(self.device)
            # del model
        # text_inputs = torch.cat([clip.tokenize(f"pick up the {obj}") for obj in classes]).to(device)  # 生成文字描述
# a = _clip_text_encoder()
# a.encode_sentence()

# print(a.encode_sentence())
# from torchvision.datasets import CIFAR100
# from PIL import Image

# img_pah = 'cup3.jpg'
# classes = ['cup', 'not_cup']

#加载模型

#
# #准备输入集
# # image = Image.open(img_pah)
# # image_input = preprocess(image).unsqueeze(0).to(device)
# text_inputs = torch.cat([clip.tokenize(f"pick up the {obj}") for obj in classes]).to(device) #生成文字描述
#
# #特征编码
# with torch.no_grad():
#     # image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_inputs)
#



