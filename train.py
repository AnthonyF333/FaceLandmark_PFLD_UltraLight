#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import warnings
import random
import os
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset.datasets import WLFWDatasets
from pfld.utils import init_weights, save_checkpoint, set_logger, write_cfg
from pfld.loss import LandmarkLoss
from test import compute_nme

from models.PFLD import PFLD
from models.PFLD_Ultralight import PFLD_Ultralight
from models.PFLD_Ultralight_Slim import PFLD_Ultralight_Slim


def train(model, train_dataloader, loss_fn, optimizer, cfg):
    losses = []
    model.train()
    with tqdm(total=len(train_dataloader)) as t:
        for img, landmark_gt in train_dataloader:
            img = img.to(cfg.DEVICE)
            landmark_gt = landmark_gt.to(cfg.DEVICE)
            landmark_pred = model(img)
            loss = loss_fn(landmark_gt, landmark_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())
            t.update()

    return np.mean(losses)


def validate(model, val_dataloader, loss_fn, cfg):
    model.eval()
    losses = []
    nme_list = []
    with torch.no_grad():
        for img, landmark_gt in val_dataloader:
            img = img.to(cfg.DEVICE)
            landmark_gt = landmark_gt.to(cfg.DEVICE)
            landmark_pred = model(img)
            loss = loss_fn(landmark_gt, landmark_pred)
            losses.append(loss.cpu().numpy())

            landmark_pred = landmark_pred.reshape(landmark_pred.shape[0], -1, 2).cpu().numpy()
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()
            nme_temp = compute_nme(landmark_pred, landmark_gt)
            for item in nme_temp:
                nme_list.append(item)

    return np.mean(losses), np.mean(nme_list)


def main():
    cfg = get_config()

    SEED = cfg.SEED
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    warnings.filterwarnings("ignore")
    set_logger(cfg.LOGGER_PATH)
    write_cfg(logging, cfg)

    main_worker(cfg)


def main_worker(cfg):
    # ======= LOADING DATA ======= #
    logging.warning('=======>>>>>>> Loading Training and Validation Data')
    TRAIN_DATA_PATH = cfg.TRAIN_DATA_PATH
    VAL_DATA_PATH = cfg.VAL_DATA_PATH
    TRANSFORM = cfg.TRANSFORM

    train_dataset = WLFWDatasets(TRAIN_DATA_PATH, TRANSFORM)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)

    val_dataset = WLFWDatasets(VAL_DATA_PATH, TRANSFORM)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    # ======= MODEL ======= #
    MODEL_DICT = {'PFLD': PFLD,
                  'PFLD_Ultralight': PFLD_Ultralight,
                  'PFLD_Ultralight_Slim': PFLD_Ultralight_Slim,
                  }
    MODEL_TYPE = cfg.MODEL_TYPE
    WIDTH_FACTOR = cfg.WIDTH_FACTOR
    INPUT_SIZE = cfg.INPUT_SIZE
    LANDMARK_NUMBER = cfg.LANDMARK_NUMBER
    model = MODEL_DICT[MODEL_TYPE](WIDTH_FACTOR, INPUT_SIZE[0], LANDMARK_NUMBER).to(cfg.DEVICE)
    # model.apply(init_weights)
    if cfg.RESUME:
        if os.path.isfile(cfg.RESUME_MODEL_PATH):
            model.load_state_dict(torch.load(cfg.RESUME_MODEL_PATH))
        else:
            logging.warning("MODEL: No Checkpoint Found at '{}".format(cfg.RESUME_MODEL_PATH))
    logging.warning('=======>>>>>>> {} Model Generated'.format(MODEL_TYPE))

    # ======= LOSS ======= #
    loss_fn = LandmarkLoss(LANDMARK_NUMBER)
    logging.warning('=======>>>>>>> Loss Function Generated')

    # ======= OPTIMIZER ======= #
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}],
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY)
    logging.warning('=======>>>>>>> Optimizer Generated')

    # ======= SCHEDULER ======= #
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=0.1)
    logging.warning('=======>>>>>>> Scheduler Generated' + '\n')

    # ======= TENSORBOARDX WRITER ======= #
    writer = SummaryWriter(cfg.LOG_PATH)

    dummy_input = torch.rand(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(cfg.DEVICE)
    writer.add_graph(model, (dummy_input,))

    best_nme = float('inf')
    for epoch in range(1, cfg.EPOCHES + 1):
        logging.warning('Epoch {} Start'.format(epoch))
        train_loss = train(model, train_dataloader, loss_fn, optimizer, cfg)
        val_loss, val_nme = validate(model, val_dataloader, loss_fn, cfg)
        scheduler.step()

        if val_nme < best_nme:
            best_nme = val_nme
            save_checkpoint(cfg, model, extra='best')
            logging.info('Save best model')
        save_checkpoint(cfg, model, epoch)

        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Val_Loss', val_loss, epoch)
        writer.add_scalar('Val_NME', val_nme, epoch)

        logging.info('Train_Loss: {}'.format(train_loss))
        logging.info('Val_Loss: {}'.format(val_loss))
        logging.info('Val_NME: {}'.format(val_nme) + '\n')

    save_checkpoint(cfg, model, extra='final')


if __name__ == "__main__":
    main()
