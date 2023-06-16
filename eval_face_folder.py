import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from matplotlib import pyplot as plt

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

# try:
#     rank = int(os.environ["RANK"])
#     local_rank = int(os.environ["LOCAL_RANK"])
#     world_size = int(os.environ["WORLD_SIZE"])
#     distributed.init_process_group("nccl")
# except KeyError:
#     rank = 0
#     local_rank = 0
#     world_size = 1
#     distributed.init_process_group(
#         backend="nccl",
#         init_method="tcp://127.0.0.1:12584",
#         rank=rank,
#         world_size=world_size,
#     )


def sim_cos(emb1, emb2):
    emb1 = emb1.detach().cpu().numpy()
    emb2 = emb2.detach().cpu().numpy()
    emb1 = emb1/np.linalg.norm(emb1)
    emb2 = emb2/np.linalg.norm(emb2)
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    sim = np.dot(emb1, emb2)
    return sim


if  __name__ == "__main__":
    # eval_folder = "test_ms1mv2"
    eval_folder = "ev_test"
    eval_batch = 1
    # torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    args = parser.parse_args()

    # get config
    print("the value of config is {}".format(args.config))
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    # torch.cuda.set_device(local_rank)

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    # backbone = torch.nn.parallel.DistributedDataParallel(
    #     module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
    #     find_unused_parameters=True)
    # backbone.register_comm_hook(None, fp16_compress_hook)

    if cfg.resume:
        if "rank" not in vars():
            rank=0
        dict_checkpoint = torch.load(os.path.join("work_dirs/ms1mv2_r50_train","model.pt"))# f"checkpoint_gpu_{rank}.pt"))#os.path.join(cfg.output, "model.pt"))#
        # start_epoch = dict_checkpoint["epoch"]
        # global_step = dict_checkpoint["global_step"]
        backbone.load_state_dict(dict_checkpoint)
        # backbone.module.load_state_dict(dict_checkpoint)
        # module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        # opt.load_state_dict(dict_checkpoint["state_optimizer"])
        # lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    backbone.eval()
    # backbone._set_static_graph()

    ### Evaluation on folder
    ids = os.listdir(eval_folder)
    images_list = []
    image2id_dict = {}
    id2image_dict = {}
    for id_ in ids:
        id_folder = os.path.join(eval_folder, id_)
        imgs = os.listdir(id_folder)
        img_paths = [os.path.join(id_folder, i) for i in imgs]
        images_list += img_paths

        for img_path in img_paths:
            image2id_dict[img_path] = id_
        id2image_dict[id_] = img_paths

    if "random" not in vars():
        import random
    if "cv2" not in vars():
        import cv2
    # random.shuffle(images_list)

    img_tensor_list = []
    sims_pos_ls = []
    sims_neg_ls = []
    def preprocess(img):
        if len(img.shape)==2:
            img_new = np.zeros((*img.shape, 3),dtype = img.dtype)
            img_new[...,0] = img
            img_new[...,1] = img
            img_new[...,2] = img
            img = img_new
        h,w,c = img.shape
        hw = [h,w]
        if h != w:
            max_side_site = np.argmax(hw)
            max_side_length = max(hw)
            other_side_site = 1 - max_side_site
            padding_num = max_side_length - hw[other_side_site]
            padding_first = padding_num // 2
            padding_second = padding_num - padding_first
            img_new = np.zeros((max_side_length, max_side_length, c))
            if other_side_site == 0:
                img_new[padding_first:padding_first + h,...] = img
            else :
                img_new[:, padding_first:padding_first+w,:] = img
            img = img_new

        img = cv2.resize(img, (112,112))
        img = torchvision.transforms.ToTensor()(img)
        img = img*2-1
        return img

    embeddings_dict = {}
    processed = []
    sim_pos_all = []
    sim_neg_all = []
    with torch.no_grad():
        if True:# with torch.cuda.amp.autocast():
            for img_path in images_list:
                id_ = image2id_dict[img_path]
                if id_ in processed:
                    continue
                # else:

                img_ = cv2.imread(img_path)
                img_ = img_[...,:3]
                img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                img1_raw = img_.copy()
                img_ = preprocess(img_)
                img_tensor_list.append(img_)
                if len(img_tensor_list) % eval_batch == 0:
                    # if len(img_tensor_list)>1:
                    #     img_tensor = torch.cat([i.unsqueeze(dim=0) for i in img_tensor_list],axis = 0)
                    # elif len(img_tensor_list)==1:
                    #     img_tensor = torch.cat([img_tensor_list.unsqueeze(dim=0), img_tensor_list[0].unsqueeze(dim=0)],axis = 0)
                    img_tensor = torch.cat([img_tensor_list[0].unsqueeze(dim=0), img_tensor_list[0].unsqueeze(dim=0)], axis=0)
                    embeddings = backbone(img_tensor.cuda())
                    img_tensor_list = []


                processed.append(id_)

                # generate the positive and negative pairs of the specified id
                ids_without_id = ids.copy()
                ids_without_id.remove(id_)
                ids_different = random.sample(ids_without_id, 5)
                imgs_negative = []
                for id_different_ in ids_different :
                    imgs_negative.append(random.choice(id2image_dict[id_different_]))
                candidate_positive = id2image_dict[id_].copy()
                candidate_positive.remove(img_path)
                imgs_positive = random.sample(candidate_positive, min(len(candidate_positive), 5))
                # if len(imgs_positive) >= 5 and img_path in imgs_positive:
                #     for temp_ in candidate_positive:
                #         if temp_ not in imgs_positive:
                #             imgs_positive.append(temp_)
                #             break
                # if img_path in imgs_positive:
                #     imgs_positive.remove(img_path)

                # pairs in image form to pairs in embeddings, both the positive and the negative
                embs_pos = []
                embs_neg = []
                for img_positive in imgs_positive:
                    img_ = cv2.imread(img_positive)
                    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                    img_ = preprocess(img_)
                    img_tensor = torch.unsqueeze(img_, dim=0)
                    feature = backbone(img_tensor.cuda())
                    embs_pos.append(feature)
                for img_negative in imgs_negative:
                    img_ = cv2.imread(img_negative)
                    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                    img_ = preprocess(img_)
                    img_tensor = torch.unsqueeze(img_, dim=0)
                    feature = backbone(img_tensor.cuda())
                    embs_neg.append(feature)

                # the record of similarity for the specified id
                sim_pos = []
                sim_neg = []

                for j, emb_ in enumerate(embs_pos):
                    sim = sim_cos(embeddings[0],emb_)
                    sim_pos.append(sim)

                    plt.subplot(121)
                    plt.imshow(img1_raw)
                    id1 = os.path.split(os.path.split(img_path)[0])[-1]
                    plt.title(id1)
                    plt.subplot(122)
                    img_path2 = imgs_positive[j]
                    id2 = os.path.split(os.path.split(img_path2)[0])[-1]
                    img_2 = cv2.imread(img_path2)[...,:3]
                    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
                    plt.imshow(img_2)
                    plt.title(id2)
                    plt.suptitle(sim)
                    plt.savefig(f"{sim}_{id1}+{os.path.basename(img_path)}_{id2}+{os.path.basename(img_path2)}.png",dpi=200)
                    plt.close()


                for emb_ in embs_neg:
                    sim = sim_cos(embeddings[0], emb_)
                    sim_neg.append(sim)

                    plt.subplot(121)
                    plt.imshow(img1_raw)
                    id1 = os.path.split(os.path.split(img_path)[0])[-1]
                    plt.title(id1)
                    plt.subplot(122)
                    img_path2 = imgs_negative[j]
                    id2 = os.path.split(os.path.split(img_path2)[0])[-1]
                    img_2 = cv2.imread(img_path2)[..., :3]
                    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
                    plt.imshow(img_2)
                    plt.title(id2)
                    plt.suptitle(sim)
                    plt.savefig(f"{sim}_{id1}+{os.path.basename(img_path)}_{id2}+{os.path.basename(img_path2)}.png",
                                dpi=200)
                    plt.close()

                sim_pos_all += sim_pos
                sim_neg_all += sim_neg

    plt.hist(sim_pos_all, bins = 50, color =  'g',label = "pos",alpha=0.5)
    plt.hist(sim_neg_all, bins=50,  color = 'r',label = "neg",alpha=0.5)
    plt.legend()
    plt.savefig("hist.png",dpi=200)
    plt.show()












