# import os
# from torch.utils.data import DataLoader
# import clip
# import torch
# import torchvision
# import time
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# def model_load(model_name):
#     # 加载模型
#     model, preprocess = clip.load(model_name, device) #ViT-B/32 RN50x16
#     return model, preprocess
#
# def data_load(data_path):
#     #加载数据集和文字描述
#     celeba = torchvision.datasets.CelebA(root='CELEBA', split='test', download=True)
#     text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in celeba.attr_names]).to(device)
#     return celeba, text_inputs
#
#
# def test_model(start, end, celeba, text_inputs, model, preprocess):
#     #测试模型
#     length = end - start + 1
#     face_accuracy = 0
#     face_score = 0
#
#     for i, data in enumerate(celeba):
#         face_result = 0
#         if i < start:
#             continue
#         image, target = data
#         image_input = preprocess(image).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             image_features = model.encode_image(image_input)
#             text_features = model.encode_text(text_inputs)
#
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#
#         text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#         top_score, top_label = text_probs.topk(6, dim=-1)
#         for k, score in zip(top_label[0], top_score[0]):
#             if k.item() < 40 and target[k.item()] == 1:
#                 face_result = 1
#                 face_score += score.item()
#                 print('Predict right! The predicted is {}'.format(celeba.attr_names[k.item()]))
#             else:
#                 print('Predict flase! The predicted is {}'.format(celeba.attr_names[k.item()]))
#         face_accuracy += face_result
#
#         if i == end:
#             break
#     face_score = face_score / length
#     face_accuracy = face_accuracy / length
#
#     return face_score, face_accuracy
#
# def main():
#     start = 0
#     end = 1000
#     model_name = 'ViT-B/32' #ViT-B/32 RN50x16
#     data_path = 'CELEBA'
#
#     time_start = time.time()
#     model, preprocess = model_load(model_name)
#     celeba, text_inputs = data_load(data_path)
#     face_score, face_accuracy = test_model(start, end, celeba, text_inputs, model, preprocess)
#     time_end = time.time()
#
#     print('The prediction:')
#     print('face_accuracy: {:.2f} face_score: {}%'.format(face_accuracy, face_score*100))
#     print('running time: %.4f'%(time_end - time_start))
#
# if __name__ == '__main__':
#     main()
import torch
import clip
from PIL import Image
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

image_original = np.array(Image.open("CLIP.png"))
from _clip import build_model, load_clip, tokenize

class my_clip(nn.Module):
    def __init__(self):
        super(my_clip, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_clip()

    def _load_clip(self):
        model, preprocess = load_clip("RN50", device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)
        self.preprocess = preprocess
        del model


    def encode_image(self, img):
        with torch.no_grad():
            img = self.preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im


    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens == 0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask
    def forward(self,img, text):
        return img, text



if __name__ == '__main__':
    net = my_clip()
    state_dict = net.state_dict()
    net.encode_image(image_original)
    net.encode_text(["a diagram", "a dog", "a cat"])
    # net.encode_image(image_.to(device))

    model, preprocess = clip.load("RN50", device=device)
    image_ = preprocess(Image.fromarray(image_original))
    image = image_.unsqueeze(0).to(device)
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)#输出的是1*1024维度特征
        text_features = model.encode_text(text)#输出的是3*1024维度特征
        # image_features_ = model.visual.prepool_im(image)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]


