import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from models.Clip import _clip_text_encoder


class ExampleNetwork(ME.MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D=3):
        super(ExampleNetwork, self).__init__(D)
        self.conv =  ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D)
        self.text_enc_dim = 1024
        self.lang_proj1 = nn.Linear( self.text_enc_dim, 64)
        self.proj2 = ME.MinkowskiLinear(64, 32)
        self.proj3 = ME.MinkowskiLinear(32, 64)
        # self.proj2 = MEF.linear(64,32)
        # self.proj3 = MEF.linear(32, 64)
        self.bn = ME.MinkowskiBatchNorm(64)
        self.conv_tr = ME.MinkowskiConvolutionTranspose(
                in_channels=64,
                out_channels=4,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D)

    def forward(self, x):
        # ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)
        text_feat, text_emb, text_mask = _clip_text_encoder().encode_sentence()
        print('input: ', x.coordinates.size(), x.features.size())
        out = self.conv(x)
        text_feat = text_feat.float().to(torch.device('cpu'))
        #half（16位）转化为float
        text_feat = self.lang_proj1(text_feat)

        def lang_fuser(x1,x2):
            #x1为点云特征，x2为文本特征
            coords, feat = x1.coordinates, x1.features
            index = coords[:,0]
            x3 = torch.randn_like(feat)
            num , vision_dim = feat.size()
            for i in range(num):
                key = index[i]
                x3[i] = x2[key]

            #完成了相乘
            text_sparse = ME.SparseTensor(x3, coordinate_map_key=x1.coordinate_map_key, coordinate_manager=x1.coordinate_manager)
            out = text_sparse * x1
            # out = ME.SparseTensor(x3*feat, coordinate_map_key=x1.coordinate_map_key, coordinate_manager=x1.coordinate_manager)
            # text_sparse = ME.SparseTensor(x3, coordinate_map_key=x1.coordiante_map_key, coordinate_manager=x1.coordinate_manager)
            return out
        # out_ = text_feat * out[:2]

        print('conv: ', out.coordinates.size(), out.features.size())
        out = self.bn(out)
        out_multi = lang_fuser(out, text_feat)
        out_multi = self.bn(out_multi)
        out = out + out_multi
        # out = MEF.linear(out,64, 32)
        out = self.proj2(out)
        print('bn: ', out.coordinates.size(), out.features.size())
        out = MEF.relu(out)
        out = self.proj3(out)
        # out = MEF.linear(out,32, 64)

        # out = out * out
        print('relu: ', out.coordinates.size(), out.features.size())
        out = self.conv_tr(out)
        print('conv_tr', out.coordinates.size(), out.features.size())
        # ME.clear_global_coordinate_manager()
        return out


if __name__ == '__main__':
    origin_pc1 = 5 * np.random.uniform(0, 1, (300, 3))
    # origin_pc1 = np.ones((300, 3), dtype=np.float32)
    feat1 = np.ones((300, 3), dtype=np.float32)
    origin_pc2 = 100 * np.random.uniform(0, 1, (6, 3))
    feat2 = np.ones((6, 3), dtype=np.float32)

    coords, feats = ME.utils.sparse_collate([origin_pc1, origin_pc2], [feat1, feat2])
    #会滤除重复的
    input = ME.SparseTensor(feats, coordinates=coords)
    # print(input.coordinates.size(), input.features.size())
    net = ExampleNetwork(in_feat=3, out_feat=32)
    output = net(input)

    print(torch.equal(input.coordinates, output.coordinates))
    print(torch.equal(input.features, output.features))
