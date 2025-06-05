import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from models.resnet import ResNetBase, ResNetBase_with_text
# from models.Clip import _clip_text_encoder
import torch.nn as nn
import torch

#在backbone阶段加入了文本的融合。
class MinkUNetBase_with_text(ResNetBase_with_text):

    BLOCK = None
    PLANES = None
    # DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    #8层
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # # PLANES = (32, 64, 128, 256, 192, 192, 192, 192)
    # BLOCK = BasicBlock
    # LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)
    #继承了权重初始化函数、重写了网络初始化函数以及forward函数。
    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        #将网络结构可视化 断点调试。
        #self.inplanes维度为32.
        self.inplanes = self.INIT_DIM
        self.text_enc_dim = 1024
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])
        #解码器阶段
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.lang_proj1 = nn.Linear( self.text_enc_dim, self.PLANES[3])
        self.bn5 = ME.MinkowskiBatchNorm(self.PLANES[3])
        self.linear1 = ME.MinkowskiLinear(self.PLANES[3], self.PLANES[3]//2)
        self.linear2 = ME.MinkowskiLinear(self.PLANES[3]//2, self.PLANES[3])

        self.lang_proj2 = nn.Linear(self.text_enc_dim, self.PLANES[4])
        self.bn6 = ME.MinkowskiBatchNorm(self.PLANES[4])
        self.linear3 = ME.MinkowskiLinear(self.PLANES[4], self.PLANES[4] // 2)
        self.linear4 = ME.MinkowskiLinear(self.PLANES[4] // 2, self.PLANES[4])

        self.lang_proj3 = nn.Linear(self.text_enc_dim, self.PLANES[5])
        self.bn7 = ME.MinkowskiBatchNorm(self.PLANES[5])
        self.linear5 = ME.MinkowskiLinear(self.PLANES[5], self.PLANES[5] // 2)
        self.linear6 = ME.MinkowskiLinear(self.PLANES[5] // 2, self.PLANES[5])

        self.lang_proj4 = nn.Linear(self.text_enc_dim, self.PLANES[6])
        self.bn8 = ME.MinkowskiBatchNorm(self.PLANES[6])
        self.linear7 = ME.MinkowskiLinear(self.PLANES[6], self.PLANES[6] // 2)
        self.linear8 = ME.MinkowskiLinear(self.PLANES[6] // 2, self.PLANES[6])

        # self.lang_proj5 = nn.Linear(self.text_enc_dim, self.PLANES[7])
        # self.bn9 = ME.MinkowskiBatchNorm(self.PLANES[7])
        # self.linear7 = ME.MinkowskiLinear(self.PLANES[7], self.PLANES[7] // 2)
        # self.linear8 = ME.MinkowskiLinear(self.PLANES[7] // 2, self.PLANES[7])

        # self.lang_proj4 = nn.Linear(self.text_enc_dim, self.PLANES[5])
        # self.bn7 = ME.MinkowskiBatchNorm(self.PLANES[5])
        # self.linear5 = ME.MinkowskiLinear(self.PLANES[5], self.PLANES[5] // 2)
        # self.linear6 = ME.MinkowskiLinear(self.PLANES[5] // 2, self.PLANES[5])


    def lang_fuser(self, x1, x2):
        # x1为点云特征，x2为文本特征
        coords, feat = x1.coordinates, x1.features
        index = coords[:, 0]
        x3 = torch.randn_like(feat)
        num, vision_dim = feat.size()
        for i in range(num):
            key = index[i]
            x3[i] = x2[key]
        # 完成了相乘
        text_sparse = ME.SparseTensor(x3, coordinate_map_key=x1.coordinate_map_key,
                                      coordinate_manager=x1.coordinate_manager)
        out = text_sparse * x1
        # out = ME.SparseTensor(x3*feat, coordinate_map_key=x1.coordinate_map_key, coordinate_manager=x1.coordinate_manager)
        # text_sparse = ME.SparseTensor(x3, coordinate_map_key=x1.coordiante_map_key, coordinate_manager=x1.coordinate_manager)
        return out
    def forward(self, x, l):
        l = l.float()
        out = self.conv0p1s1(x)
        # x:(32401,4) (32401,3)
        # out:(32401,4) (32401,32)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        # (32401,32)
        out = self.conv1p1s2(out_p1)
        # (13742,32)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        # (13742,32)
        out = self.conv2p2s2(out_b1p2)
        # (4370, 32)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        # (4370, 64)
        out = self.conv3p4s2(out_b2p4)
        # (1232, 64)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        # (1232, 128)
        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        # (330, 128)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        # (330, 256)
        # 解码器阶段
        # tensor_stride=8
        l1 = self.lang_proj1(l)
        out_multi = self.lang_fuser(out, l1)
        out = self.bn5(out_multi) + self.bn5(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.convtr4p16s2(out)
        # (1232, 192)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        # (1232, 192+128)
        out = self.block5(out)
        # (1232, 192)

        # tensor_stride=4
        l2 = self.lang_proj2(l)
        out_multi = self.lang_fuser(out, l2)
        out = self.bn6(out_multi) + self.bn6(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.convtr5p8s2(out)
        # (4370, 192)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        # (4370，192+64)
        out = self.block6(out)
        # (4370，192)

        # tensor_stride=2
        l3 = self.lang_proj3(l)
        out_multi = self.lang_fuser(out, l3)
        out = self.bn7(out_multi) + self.bn7(out)
        out = self.linear5(out)
        out = self.relu(out)
        out = self.linear6(out)
        out = self.convtr6p4s2(out)
        # (13742，192）
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        # (13742，192+32)
        out = self.block7(out)
        # (13742，192)
        # tensor_stride=1
        l4 = self.lang_proj4(l)
        out_multi = self.lang_fuser(out, l4)
        out = self.bn8(out_multi) + self.bn8(out)
        out = self.linear7(out)
        out = self.relu(out)
        out = self.linear8(out)
        out = self.convtr7p2s2(out)
        # (32401，192)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        # (32401，192+32)
        out = self.block8(out)
        # (32401，192)
        out = self.final(out)
        # (32401，512)

        return out

class MinkUNetBase_with_text_frozen(ResNetBase_with_text):

    BLOCK = None
    PLANES = None
    # DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    #8层
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # # PLANES = (32, 64, 128, 256, 192, 192, 192, 192)
    # BLOCK = BasicBlock
    # LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)
    #继承了权重初始化函数、重写了网络初始化函数以及forward函数。
    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        #将网络结构可视化 断点调试。
        #self.inplanes维度为32.
        self.inplanes = self.INIT_DIM
        self.text_enc_dim = 1024
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])
        #解码器阶段
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

        # self.lang_proj1 = nn.Linear( self.text_enc_dim, self.PLANES[3])
        # self.bn5 = ME.MinkowskiBatchNorm(self.PLANES[3])
        # self.linear1 = ME.MinkowskiLinear(self.PLANES[3], self.PLANES[3]//2)
        # self.linear2 = ME.MinkowskiLinear(self.PLANES[3]//2, self.PLANES[3])
        #
        # self.lang_proj2 = nn.Linear(self.text_enc_dim, self.PLANES[4])
        # self.bn6 = ME.MinkowskiBatchNorm(self.PLANES[4])
        # self.linear3 = ME.MinkowskiLinear(self.PLANES[4], self.PLANES[4] // 2)
        # self.linear4 = ME.MinkowskiLinear(self.PLANES[4] // 2, self.PLANES[4])
        #
        # self.lang_proj3 = nn.Linear(self.text_enc_dim, self.PLANES[5])
        # self.bn7 = ME.MinkowskiBatchNorm(self.PLANES[5])
        # self.linear5 = ME.MinkowskiLinear(self.PLANES[5], self.PLANES[5] // 2)
        # self.linear6 = ME.MinkowskiLinear(self.PLANES[5] // 2, self.PLANES[5])
        #
        # self.lang_proj4 = nn.Linear(self.text_enc_dim, self.PLANES[6])
        # self.bn8 = ME.MinkowskiBatchNorm(self.PLANES[6])
        # self.linear7 = ME.MinkowskiLinear(self.PLANES[6], self.PLANES[6] // 2)
        # self.linear8 = ME.MinkowskiLinear(self.PLANES[6] // 2, self.PLANES[6])
        #
        # # self.lang_proj5 = nn.Linear(self.text_enc_dim, self.PLANES[7])
        # # self.bn9 = ME.MinkowskiBatchNorm(self.PLANES[7])
        # # self.linear7 = ME.MinkowskiLinear(self.PLANES[7], self.PLANES[7] // 2)
        # # self.linear8 = ME.MinkowskiLinear(self.PLANES[7] // 2, self.PLANES[7])
        #
        # # self.lang_proj4 = nn.Linear(self.text_enc_dim, self.PLANES[5])
        # # self.bn7 = ME.MinkowskiBatchNorm(self.PLANES[5])
        # # self.linear5 = ME.MinkowskiLinear(self.PLANES[5], self.PLANES[5] // 2)
        # # self.linear6 = ME.MinkowskiLinear(self.PLANES[5] // 2, self.PLANES[5])


    def lang_fuser(self, x1, x2):
        # x1为点云特征，x2为文本特征
        coords, feat = x1.coordinates, x1.features
        index = coords[:, 0]
        x3 = torch.randn_like(feat)
        num, vision_dim = feat.size()
        for i in range(num):
            key = index[i]
            x3[i] = x2[key]
        # 完成了相乘
        text_sparse = ME.SparseTensor(x3, coordinate_map_key=x1.coordinate_map_key,
                                      coordinate_manager=x1.coordinate_manager)
        out = text_sparse * x1
        # out = ME.SparseTensor(x3*feat, coordinate_map_key=x1.coordinate_map_key, coordinate_manager=x1.coordinate_manager)
        # text_sparse = ME.SparseTensor(x3, coordinate_map_key=x1.coordiante_map_key, coordinate_manager=x1.coordinate_manager)
        return out
    def forward(self, x, l):
        l = l.float()
        out = self.conv0p1s1(x)
        # x:(32401,4) (32401,3)
        # out:(32401,4) (32401,32)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        # (32401,32)
        out = self.conv1p1s2(out_p1)
        # (13742,32)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        # (13742,32)
        out = self.conv2p2s2(out_b1p2)
        # (4370, 32)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        # (4370, 64)
        out = self.conv3p4s2(out_b2p4)
        # (1232, 64)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        # (1232, 128)
        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        # (330, 128)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        # (330, 256)
        # 解码器阶段
        # tensor_stride=8
        l1 = self.lang_proj1(l)
        out_multi = self.lang_fuser(out, l1)
        out = self.bn5(out_multi) + self.bn5(out)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.convtr4p16s2(out)
        # (1232, 192)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        # (1232, 192+128)
        out = self.block5(out)
        # (1232, 192)

        # tensor_stride=4
        l2 = self.lang_proj2(l)
        out_multi = self.lang_fuser(out, l2)
        out = self.bn6(out_multi) + self.bn6(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.convtr5p8s2(out)
        # (4370, 192)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        # (4370，192+64)
        out = self.block6(out)
        # (4370，192)

        # tensor_stride=2
        l3 = self.lang_proj3(l)
        out_multi = self.lang_fuser(out, l3)
        out = self.bn7(out_multi) + self.bn7(out)
        out = self.linear5(out)
        out = self.relu(out)
        out = self.linear6(out)
        out = self.convtr6p4s2(out)
        # (13742，192）
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        # (13742，192+32)
        out = self.block7(out)
        # (13742，192)
        # tensor_stride=1
        l4 = self.lang_proj4(l)
        out_multi = self.lang_fuser(out, l4)
        out = self.bn8(out_multi) + self.bn8(out)
        out = self.linear7(out)
        out = self.relu(out)
        out = self.linear8(out)
        out = self.convtr7p2s2(out)
        # (32401，192)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        # (32401，192+32)
        out = self.block8(out)
        # (32401，192)
        out = self.final(out)
        # (32401，512)

        return out

class MinkUNetBase(ResNetBase):
    # # PLANES = (32, 64, 128, 256, 192, 192, 192, 192)
    # BLOCK = BasicBlock
    # LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    BLOCK = None
    PLANES = None
    # DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)
    #继承了权重初始化函数、重写了网络初始化函数以及forward函数。
    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        #x:(32401,4) (32401,3)
        #out:(32401,4) (32401,32)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        #(32401,32)
        out = self.conv1p1s2(out_p1)
        #(13742,32)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        # (13742,32)
        out = self.conv2p2s2(out_b1p2)
        #(4370, 32)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)
        # (4370, 64)
        out = self.conv3p4s2(out_b2p4)
        #(1232, 64)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)
        # (1232, 128)
        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        # (330, 128)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)
        # (330, 256)
        #解码器阶段
        # tensor_stride=8
        out = self.convtr4p16s2(out)
        # (1232, 192)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        # (1232, 192+128)
        out = self.block5(out)
        # (1232, 192)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        # (4370, 192)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        # (4370，192+64)
        out = self.block6(out)
        # (4370，192)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        # (13742，192）
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        # (13742，192+32)
        out = self.block7(out)
        # (13742，192)
        # tensor_stride=1
        out = self.convtr7p2s2(out)
        # (32401，192)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        # (32401，192+32)
        out = self.block8(out)
        # (32401，192)
        out = self.final(out)
        # (32401，512)
        return out


class MinkUNet14(MinkUNetBase):
    #PLANES = (32, 64, 128, 256, 192, 192, 192, 192)
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet14_with_text(MinkUNetBase_with_text):
    #PLANES = (32, 64, 128, 256, 192, 192, 192, 192)
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet14_with_text_frozen(MinkUNetBase_with_text_frozen):
    #PLANES = (32, 64, 128, 256, 192, 192, 192, 192)
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14Dori(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet14E(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 192, 192)

class MinkUNet14D_with_text_frozen(MinkUNet14_with_text_frozen):
    PLANES = (32, 64, 128, 256, 192, 192, 192, 192)

class MinkUNet14D_with_text(MinkUNet14_with_text):
    PLANES = (32, 64, 128, 256, 192, 192, 192, 192)
class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
