from utils import Logger, adjust_learning_rate, CreateLogger, create_cosine_learing_schdule, encode_test_label, set_seed
from TDSF.model import TDSFNet
from dependency import *
from torch import optim
from dataloader import generate_dataloader
import torch
import Config as config
import wandb
import numpy as np

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def build_metrics(num_classes, device):
    """
    Returns a MetricCollection on the given device.
    Call .reset() before each epoch, .update() each batch, .compute() at end of epoch.
    """
    metrics = MetricCollection({
        'acc':           MulticlassAccuracy(num_classes=num_classes, average='macro'),
        'f1_macro':      MulticlassF1Score(num_classes=num_classes, average='macro'),
        'f1_weighted':   MulticlassF1Score(num_classes=num_classes, average='weighted'),
        'precision':     MulticlassPrecision(num_classes=num_classes, average='macro'),
        'recall':        MulticlassRecall(num_classes=num_classes, average='macro'),
        # 'auc':           MulticlassAUROC(num_classes=num_classes, average='macro'),
    })
    return metrics.to(device)


def log_metrics(metric_collection, prefix):

    computed = metric_collection.compute()
    metric_collection.reset()
    return {f'{prefix}/{k}': v.item() for k, v in computed.items()}


def train(net, train_dataloader, model_name, metrics):
    net.set_mode('train')
    train_loss = 0
    train_acc  = 0
    metrics.reset()

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        opt.zero_grad()

        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
        meta_data    = meta_data.cuda()

        diagnosis_label = label[0].long().cuda()
        pn_label  = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label  = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label  = label[7].long().cuda()

        [(logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,
          logit_bwv_fusion, logit_vs_fusion)] = net((clinic_image, derm_image))

        loss_fusion = torch.true_divide(
            net.criterion(logit_fusion,     diagnosis_label)
            + net.criterion(logit_pn_fusion,  pn_label)
            + net.criterion(logit_str_fusion, str_label)
            + net.criterion(logit_pig_fusion, pig_label)
            + net.criterion(logit_rs_fusion,  rs_label)
            + net.criterion(logit_dag_fusion, dag_label)
            + net.criterion(logit_bwv_fusion, bwv_label)
            + net.criterion(logit_vs_fusion,  vs_label), 8)

        acc_fusion = torch.true_divide(
            net.metric(logit_fusion,     diagnosis_label)
            + net.metric(logit_pn_fusion,  pn_label)
            + net.metric(logit_str_fusion, str_label)
            + net.metric(logit_pig_fusion, pig_label)
            + net.metric(logit_rs_fusion,  rs_label)
            + net.metric(logit_dag_fusion, dag_label)
            + net.metric(logit_bwv_fusion, bwv_label)
            + net.metric(logit_vs_fusion,  vs_label), 8 * clinic_image.size(0))

        loss_fusion.backward()
        opt.step()

        train_loss += loss_fusion.item()
        train_acc  += acc_fusion.item()

        # probs = torch.softmax(logit_fusion, dim=1).detach()
        probs = logit_fusion.argmax(dim=1)
        metrics.update(probs, diagnosis_label)

    train_loss /= (index + 1)
    train_acc  /= (index + 1)

    extra_metrics = log_metrics(metrics, prefix='train')
    return train_loss, train_acc, extra_metrics


def validation(net, val_dataloader, model_name, metrics):
    net.set_mode('valid')
    val_loss = 0
    val_acc  = 0
    metrics.reset()

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):

        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()

        diagnosis_label = label[0].long().cuda()
        pn_label  = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label  = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label  = label[7].long().cuda()

        with torch.no_grad():
            [(logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,
              logit_bwv_fusion, logit_vs_fusion)] = net((clinic_image, derm_image))

            loss_fusion = torch.true_divide(
                net.criterion(logit_fusion,     diagnosis_label)
                + net.criterion(logit_pn_fusion,  pn_label)
                + net.criterion(logit_str_fusion, str_label)
                + net.criterion(logit_pig_fusion, pig_label)
                + net.criterion(logit_rs_fusion,  rs_label)
                + net.criterion(logit_dag_fusion, dag_label)
                + net.criterion(logit_bwv_fusion, bwv_label)
                + net.criterion(logit_vs_fusion,  vs_label), 8)

            acc_fusion = torch.true_divide(
                net.metric(logit_fusion,     diagnosis_label)
                + net.metric(logit_pn_fusion,  pn_label)
                + net.metric(logit_str_fusion, str_label)
                + net.metric(logit_pig_fusion, pig_label)
                + net.metric(logit_rs_fusion,  rs_label)
                + net.metric(logit_dag_fusion, dag_label)
                + net.metric(logit_bwv_fusion, bwv_label)
                + net.metric(logit_vs_fusion,  vs_label), 8 * clinic_image.size(0))

            probs = torch.softmax(logit_fusion, dim=1)
            metrics.update(probs, diagnosis_label)

        val_loss += loss_fusion.item()
        val_acc  += acc_fusion.item()

    val_loss /= (index + 1)
    val_acc  /= (index + 1)

    extra_metrics = log_metrics(metrics, prefix='val')
    return val_loss, val_acc, extra_metrics


def test(net, test_dataloader, model_name, metrics):
    net.set_mode('valid')
    test_loss = 0
    test_acc  = 0
    metrics.reset()

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(test_dataloader):

        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()

        diagnosis_label = label[0].long().cuda()
        pn_label  = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label  = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label  = label[7].long().cuda()

        with torch.no_grad():
            [(logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,
              logit_bwv_fusion, logit_vs_fusion)] = net((clinic_image, derm_image))

            loss_fusion = torch.true_divide(
                net.criterion(logit_fusion,     diagnosis_label)
                + net.criterion(logit_pn_fusion,  pn_label)
                + net.criterion(logit_str_fusion, str_label)
                + net.criterion(logit_pig_fusion, pig_label)
                + net.criterion(logit_rs_fusion,  rs_label)
                + net.criterion(logit_dag_fusion, dag_label)
                + net.criterion(logit_bwv_fusion, bwv_label)
                + net.criterion(logit_vs_fusion,  vs_label), 8)

            acc_fusion = torch.true_divide(
                net.metric(logit_fusion,     diagnosis_label)
                + net.metric(logit_pn_fusion,  pn_label)
                + net.metric(logit_str_fusion, str_label)
                + net.metric(logit_pig_fusion, pig_label)
                + net.metric(logit_rs_fusion,  rs_label)
                + net.metric(logit_dag_fusion, dag_label)
                + net.metric(logit_bwv_fusion, bwv_label)
                + net.metric(logit_vs_fusion,  vs_label), 8 * clinic_image.size(0))

            probs = torch.softmax(logit_fusion, dim=1)
            metrics.update(probs, diagnosis_label)

        test_loss += loss_fusion.item()
        test_acc  += acc_fusion.item()

    test_loss /= (index + 1)
    test_acc  /= (index + 1)

    extra_metrics = log_metrics(metrics, prefix='test')
    return test_loss, test_acc, extra_metrics


def run_train(model_name, mode, i, num_classes):
    log.write('** start training here! **\n')
    best_mean_acc = 0

    device = torch.device('cuda')

    # One shared MetricCollection — reset() is called at the top of each function
    metrics = build_metrics(num_classes=num_classes, device=device)

    for epoch in range(epochs):
        swa_lr = cosine_learning_schule[epoch]
        adjust_learning_rate(opt, swa_lr)

        # ── Train ────────────────────────────────────────────────────────────
        train_loss, train_acc, train_metrics = train(net, train_dataloader, model_name, metrics)
        log.write('Round: {}, epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, '
                  'Train F1: {:.4f}\n'.format(
                      i, epoch, train_loss, train_acc,
                      train_metrics['train/f1_macro'],
                    #   train_metrics.get('train/auc', 0.0)))
                  ))

        # ── Validation ───────────────────────────────────────────────────────
        val_loss, val_acc, val_metrics = validation(net, val_dataloader, model_name, metrics)
        log.write('Round: {}, epoch: {}, Valid Loss: {:.4f}, Valid Acc: {:.4f}, '
                  'Valid F1: {:.4f}\n'.format(
                      i, epoch, val_loss, val_acc,
                      val_metrics['val/f1_macro'],
                  ))
                    #   val_metrics.get('val/auc', 0.0)))

        # ── W&B logging ──────────────────────────────────────────────────────
        wandb.log({
            'epoch':         epoch,
            'round':         i,
            'learning_rate': swa_lr,
            'train/loss':    train_loss,
            'train/acc':     train_acc,
            'val/loss':      val_loss,
            'val/acc':       val_acc,
            **train_metrics,
            **val_metrics,
        })

        # ── Save best checkpoint ─────────────────────────────────────────────
        if val_acc > best_mean_acc:
            best_mean_acc = val_acc
            torch.save(net.state_dict(), out_dir + '/checkpoint/{:.4f}val_model.pth'.format(best_mean_acc))
            torch.save(net.state_dict(), out_dir + '/best_modal.pth')
            log.write('Current Best Mean Acc is {}\n'.format(best_mean_acc))

            wandb.run.summary['best_val_acc']  = best_mean_acc
            wandb.run.summary['best_val_f1']   = val_metrics['val/f1_macro']
            # wandb.run.summary['best_val_auc']  = val_metrics.get('val/auc', None)
            wandb.run.summary['best_epoch']    = epoch

    # ── Test on best checkpoint ──────────────────────────────────────────────
    log.write('\n** Loading best model for test evaluation **\n')
    net.load_state_dict(torch.load(out_dir + '/best_modal.pth'))

    test_loss, test_acc, test_metrics = test(net, test_dataloader, model_name, metrics)
    log.write('Round: {}, Test Loss: {:.4f}, Test Acc: {:.4f}, '
              'Test F1: {:.4f}\n'.format(
                  i, test_loss, test_acc,
                  test_metrics['test/f1_macro'],
              ))
                #   test_metrics.get('test/auc', 0.0)))

    wandb.log({
        'test/loss': test_loss,
        'test/acc':  test_acc,
        **test_metrics,
    })
    wandb.run.summary['test_acc']      = test_acc
    wandb.run.summary['test_loss']     = test_loss
    wandb.run.summary['test_f1_macro'] = test_metrics['test/f1_macro']
    # wandb.run.summary['test_auc']      = test_metrics.get('test/auc', None)


if __name__ == '__main__':
    mode        = 'multimodal'
    model_name  = 'TDSFNet'
    shape       = (224, 224)
    batch_size  = 16
    num_workers = 10
    data_mode   = 'Normal'
    num_classes = 5
    deterministic = True

    if deterministic:
        if data_mode == 'Normal':
            random_seeds = 100
        elif data_mode == 'self_evaluated':
            random_seeds = 183

    rounds    = 1
    lr        = 5e-5
    epochs    = 350
    swa_epoch = 50

    train_dataloader, val_dataloader, test_dataloader = generate_dataloader(shape, batch_size, num_workers, data_mode)

    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)

        print(random_seeds + i)
        log, out_dir = CreateLogger(mode, model_name, i, data_mode)

        # wandb.init(
        #     project='TDSFNet on SPC',
        #     name=f'{model_name}_round{i}',
        #     config={
        #         'model':       model_name,
        #         'mode':        mode,
        #         'shape':       shape,
        #         'batch_size':  batch_size,
        #         'lr':          lr,
        #         'epochs':      epochs,
        #         'swa_epoch':   swa_epoch,
        #         'data_mode':   data_mode,
        #         'num_classes': num_classes,
        #         'random_seed': random_seeds + i,
        #     },
        #     reinit=True,
        #     save_code=True
        # )

        net = TDSFNet(class_list=class_list, config=config.get_model_config()).cuda()
        # from torchinfo import summary

        # dummy_input = (
        # torch.randn(1, 3, 224, 224),
        # torch.randn(1, 3, 224, 224),
        # )

        # print(summary(net, input_data=(dummy_input, ), device='cpu', depth=2))
        # raise

        batches_per_epoch = len(train_dataloader)
        wandb.watch(net, log='gradients', log_freq=batches_per_epoch)

        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = optimizer

        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(model_name, mode, i, num_classes)

        wandb.finish()