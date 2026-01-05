import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from DAGNet_model.PAR import PAR

from .GSN import GroupedSpatialNormalization as pin
# from .CASAtt import SpatialOperation as CASAtt
# from .FAM import FirstOctaveConv as FAM
# from omegaconf import OmegaConf
# from .GBC import GBC
# from .MWSAttention import MWSAttention
#from .FCM import FCM
from .your_conv_module import Conv
from .DAAF import DAAF

def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)

#这是一个结合 CLIP 编码器和语义分割解码器的神经网络模型，通常用于零样本语义分割任务，或者利用文本引导图像理解的场景。
class DAGNet(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, dataset_root_path=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        #使用 OpenAI 的 CLIP 模型加载器。加载视觉-文本模型（如 ViT-B/16）及其权重到指定设备上。
        self.encoder, _ = clip.load(clip_model, device=device)
        #冻结部分参数（Fine-tuning 第12层）
        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False
        #打印所有参数的训练状态，用于调试确认哪些部分被冻结了。
        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)
        #存储传入的图像特征通道数，供解码器使用。
        self.in_channels = in_channels
        ################################################
        self.convpin = pin(3)
        #self.fcm = FCM(dim=self.embedding_dim, dim_out=self.embedding_dim)
        self.fcm = DAAF(dim=self.embedding_dim)
        ############################################
        #SegFormerHead 是轻量化 Transformer 分割头，用于将 in_channels 融合成 embedding_dim，最后预测 num_classes 分类图。其中index=11 可能是表示和 encoder 的第 12 层对齐。
        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        #用于细化分割特征的 Transformer 解码器，有 3 层、8 个注意头，输出维度等于分类数。
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)
        #文本特征（前景与背景）初始化
        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], self.encoder)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], self.encoder)
        #Grad-CAM 是一种可视化方法，用于定位图像中与分类相关的区域。此处选择 encoder 最后一层的 LayerNorm 层做目标层。
        #reshape_transform 用于将 CLIP 的 token 特征 reshape 成空间特征图格式。
        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        #设置路径，用于读取图像标签文件（增强版）。
        self.root_path = os.path.join(dataset_root_path, 'SegmentationClassAug')
        #CAM 背景阈值，用于过滤背景区域
        self.cam_bg_thres = 1
        #将 encoder 设为 eval 模式，防止 Dropout 或 BatchNorm 被更新。
        self.encoder.eval()
        #初始化PAR模块 PAR 是自定义的图像细化模块，用于伪标签精修。
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True



    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups
    


    def forward(self, img, img_names='2007_000032', mode='train'):
        ############################
        img = self.convpin(img)#####      cams  miou = 0.7457  segs  miou= 0.0020
        #############################
        cam_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1
        #生成 CLIP 特征与注意力图
        #使用 generate_clip_fts 获取所有层的特征（fts_all）和注意力权重（attn_weight_list）
        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        #将特征和注意力列表堆叠为张量，方便后续处理
        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        #选择 CAM 特征来源层，默认选用最后一层
        if self.require_all_fts==True:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)
        #图像令牌处理与特征融合
        #取每层除 [CLS] token 外的图像 token  调整维度为 [层数*b, c, h, w]，为后续卷积处理做准备
        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)

        #使用 decoder_fts_fuse 做卷积特征融合；复制一份给注意力部分
        fts = self.decoder_fts_fuse(all_img_tokens)
        ########################################
        # attn_fts = self.fcm(fts.clone())
        attn_fts = fts.clone()
        fts, attn_fts = self.fcm(fts, attn_fts)
        ########################################
        _, _, fts_h, fts_w = fts.shape
        #解码生成语义分割图 & 注意力预测  通过 DecoderTransformer 输出分割预测图和注意力图
        seg, seg_attn_weight_list = self.decoder(fts)
        #将特征展平为 [B, C, H*W]，计算注意力自相关图 B × (H*W) × (H*W)
        #使用 sigmoid 得到二值化注意力预测图
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)
        #单图像逐个生成 refined CAMs 和标签
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.png')
            img_i = img[i]
            #获取当前图像的特征、注意力、分割注意力预测
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]
            #训练早期不用 segmentation transformer refine CAM，后期或验证阶段启用
            if self.iter_num > 15000 or mode=='val': #15000
                require_seg_trans = True
            else:
                require_seg_trans = False
            #CAM生成与精炼  调用函数融合 CLIP Attention 和 segmentation attention 生成 refined CAM
            cam_refined_list, keys, w, h = perform_single_voc_cam(img_path, img_i, cam_fts, cam_attn, seg_attn,
                                                                   self.bg_text_features, self.fg_text_features,
                                                                   self.grad_cam,
                                                                   mode=mode,
                                                                   require_seg_trans=require_seg_trans)

            #从 refined CAM 中生成 CAM 标签  添加背景类别得分
            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()
            #PAR精化CAM为像素标签
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()
            
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            #使用 PAR 算法进一步精化伪标签
            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)

        return seg, all_cam_labels, attn_pred

        
