import argparse
import datetime
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from DAGNet_model.model_attn_aff_voc import DAGNet


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='../configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate(model=None, data_loader=None, cfg=None):

    preds, gts, cams, aff_gts = [], [], [], []
    num = 1
    seg_hist = np.zeros((21, 21))
    cam_hist = np.zeros((21, 21))
    for _, data in tqdm(enumerate(data_loader),
                        total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        segs, cam, attn_loss = model(inputs, name, 'val')

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
        cams += list(cam.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))

        num+=1

        if num % 1000 ==0:
            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
            cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
        # if num % 1000 == 0:
        #     seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist, cfg.dataset.num_classes)
        #     cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist, cfg.dataset.num_classes)
            preds, gts, cams, aff_gts = [], [], [], []

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
    # seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist, cfg.dataset.num_classes)
    # cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist, cfg.dataset.num_classes)

    model.train()
    return seg_score, cam_score

#这段代码定义了一个分割损失函数 get_seg_loss，它将前景和背景分别计算交叉熵损失，然后取平均，以便更好地平衡前景与背景的监督信号。
def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()#拷贝一份标签，准备用于提取背景标签。
    bg_label[label!=0] = ignore_index#将所有不等于 0 的像素（即前景）标记为 ignore_index。
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)#使用 PyTorch 的 cross_entropy 计算背景类的损失。
    fg_label = label.clone()#同样拷贝一份标签，准备提取前景标签。
    fg_label[label==0] = ignore_index#将所有等于 0 的像素（即背景）标记为忽略。
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)#对前景像素计算交叉熵损失。
    #返回前景和背景损失的平均值。  这样可以防止模型偏向背景类别（因为背景通常占多数）。
    return (bg_loss + fg_loss) * 0.5

#是用于生成一个二维亲和掩码矩阵的函数。这个掩码用于指定在特征图中哪些像素对之间需要计算亲和关系（Affinity），用于训练亲和损失等任务
def get_mask_by_radius(h=20, w=20, radius=8):#定义函数，生成一个基于二维位置的邻域掩码。
    hw = h * w #计算总像素点数，即展平后特征图的长度。
    mask  = np.zeros((hw, hw))#初始化一个 [hw, hw] 的全零矩阵 mask，表示所有像素对之间的关系（是否在有效邻域内）。
    for i in range(hw): #遍历展平后的每个像素点编号 i。将一维索引 i 转换回二维坐标：(_h, _w)，即当前像素在二维图上的位置。
        _h = i // w
        _w = i % w
        #  计算当前像素点 (_h, _w) 周围的邻域范围，即 (_h0, _h1) 和 (_w0, _w1)。
        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):#遍历当前像素的邻域内所有的像素坐标 (_i1, _i2)，并将其转换成展平后的一维索引 _i2。
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1#设置掩码矩阵中对应的位置为 1，表示这两个像素点是邻域内的亲和对。
                mask[_i2, i] = 1#因为亲和是对称的，所以同时设置 mask[i, _i2] 和 mask[_i2, i]。

    return mask



def train(cfg):

    num_workers = 10
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='train',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    DAGNet_model = DAGNet(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device='cuda'
    )
    #计算参数量
    # 计算模型总参数量（只统计可训练参数）
    total_params = sum(p.numel() for p in DAGNet_model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params:,}")

    logging.info('\nNetwork config: \n%s'%(DAGNet_model))
    param_groups = DAGNet_model.get_param_groups()
    DAGNet_model.cuda()

    #计算用于注意力掩码（attn_mask）的尺寸。
    mask_size = int(cfg.dataset.crop_size // 16)
    #调用函数 get_mask_by_radius 生成一个 注意力掩码（attention mask）
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    #初始化一个 TensorBoard 可视化工具 的写入器（writer）实例。
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
    #涉及优化器（PolyWarmupAdamW）设置、训练数据加载器初始化，以及训练性能统计器的构造
    optimizer = PolyWarmupAdamW(
        params=[
            {#包含网络中最主要的部分，例如 backbone；
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {#学习率和 weight decay 为 0；该部分通常用于冻结层（如 CLIP 编码器的早期层），确保这些参数不参与训练。
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {#对应模型中需快速收敛的模块（如分类器、decoder 等）；
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {#使用 基础学习率的 10 倍，以加速这些新参数的学习。
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    #这段代码实现了一个基于 DAGNet_model 的语义分割模型训练过程，包含数据读取、前向传播、损失计算、反向传播、日志记录、验证和模型保存等功能。
    for n_iter in range(cfg.train.max_iters):
        #从训练数据迭代器 train_loader_iter 中获取一批数据。
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        #模型前向传播 segs: 分割结果（即 mask 输出）；cam: 来自视觉注意力机制或 Grad-CAM 的类激活图（Class Activation Map）；attn_pred: 用于注意力一致性监督的预测结果。
        segs, cam, attn_pred = DAGNet_model(inputs.cuda(), img_name)
        #将 CAM（类激活图）作为伪标签，监督分割网络学习。
        pseudo_label = cam
        #将 segs 的尺寸调整为和 pseudo_label 一致，用于对齐计算损失。
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        #复制一份 CAM 图用于后续计算。
        fts_cam = cam.clone()

        #利用 CAM 和邻接关系（mask）构造注意力监督标签 aff_label，即哪些像素属于同一目标。
        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        #计算注意力损失，比较 attn_pred 与 aff_label 的一致性。
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)
        #计算伪标签监督下的分割损失。
        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        #最终总损失为：分割损失 + 0.1 倍的注意力损失。
        loss = 1 * seg_loss + 0.1*attn_loss

        #更新滑动平均记录器中的分割损失与注意力损失值。
        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #每隔一定迭代数（如 100 次）打印日志：
        if (n_iter + 1) % cfg.train.log_iters == 0:
            #计算耗时与剩余时间 ETA，记录当前学习率。
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']
            #计算预测的分割精度（mAcc）
            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size

            #记录到日志与 TensorBoard 中。
            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'), seg_mAcc))

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item()}, global_step=n_iter)

        #每隔 eval_iters 次迭代进行一次验证：
        if (n_iter + 1) % cfg.train.eval_iters == 0:
            #设置模型权重保存路径，并记录日志。
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "DAGNet_model_iter_%d.pth"%(n_iter+1))
            logging.info('Validating...')
            #大于某个迭代次数（例如 26000）才保存模型。
            if (n_iter + 1) > 26000:
                torch.save(DAGNet_model.state_dict(), ckpt_name)
            #调用 validate 函数对当前模型进行验证，并记录分割与 CAM 得分。
            seg_score, cam_score = validate(model=DAGNet_model, data_loader=val_loader, cfg=cfg)
            logging.info("cams score:")
            logging.info(cam_score)
            logging.info("segs score:")
            logging.info(seg_score)

    return True


if __name__ == "__main__":
    # 解析命令行参数，并加载配置文件。
    args = parser.parse_args()
    #使用 OmegaConf 加载 YAML 格式的配置文件（通常定义训练超参数、路径、网络结构等）。
    cfg = OmegaConf.load(args.config)
    #用命令行参数覆盖配置文件中的图像裁剪大小（常用于实验中动态调整）。
    cfg.dataset.crop_size = args.crop_size
    #如果命令行指定了 --work_dir，则覆盖配置文件中定义的默认工作目录。
    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir
    #生成当前时间戳字符串，用于保存不同时间点的训练结果文件
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    #构建输出目录路径
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)
    #创建目录
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)
    #设置日志系统  初始化日志系统，将日志写入时间戳命名的 .log 文件。
    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)
    #设置随机数种子，确保模型训练过程可复现。
    setup_seed(1)
    train(cfg=cfg)
    # # 训练完成后，执行 CRF 后处理
    # from test_msc_flip_voc import crf_proc  # 假设你放在 utils 目录下
    # crf_proc(config=cfg)
