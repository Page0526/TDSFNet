from utils import Logger, adjust_learning_rate, CreateLogger, create_cosine_learing_schdule, encode_test_label, set_seed
from TDSF.model import TDSFNet
from dependency import *
from torch import optim
from dataloader import generate_dataloader
import torch
import Config as config
import wandb

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def train(net, train_dataloader, model_name):
    net.set_mode('train')
    train_loss = 0
    train_acc = 0

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        opt.zero_grad()
        
        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()
        meta_data = meta_data.cuda()
        
        diagnosis_label = label[0].long().cuda()
        pn_label = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label = label[7].long().cuda()

        [(logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,
          logit_bwv_fusion, logit_vs_fusion)
         ] = net((clinic_image, derm_image))
        
        loss_fusion = torch.true_divide(
            net.criterion(logit_fusion, diagnosis_label)
            + net.criterion(logit_pn_fusion, pn_label)
            + net.criterion(logit_str_fusion, str_label)
            + net.criterion(logit_pig_fusion, pig_label)
            + net.criterion(logit_rs_fusion, rs_label)
            + net.criterion(logit_dag_fusion, dag_label)
            + net.criterion(logit_bwv_fusion, bwv_label)
            + net.criterion(logit_vs_fusion, vs_label), 8)

        loss = loss_fusion

        acc_fusion = torch.true_divide(net.metric(logit_fusion, diagnosis_label)
                                     + net.metric(logit_pn_fusion, pn_label)
                                     + net.metric(logit_str_fusion, str_label)
                                     + net.metric(logit_pig_fusion, pig_label)
                                     + net.metric(logit_rs_fusion, rs_label)
                                     + net.metric(logit_dag_fusion, dag_label)
                                     + net.metric(logit_bwv_fusion, bwv_label)
                                     + net.metric(logit_vs_fusion, vs_label), 8 * clinic_image.size(0))

        acc = acc_fusion

        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_acc += acc.item()

    train_loss = train_loss / (index + 1)
    train_acc = train_acc / (index + 1)

    return train_loss, train_acc


def validation(net, val_dataloader, model_name):
    net.set_mode('valid')
    val_loss = 0
    val_acc = 0

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):

        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()

        diagnosis_label = label[0].long().cuda()
        pn_label = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label = label[7].long().cuda()

        with torch.no_grad():
            [(logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,
              logit_bwv_fusion, logit_vs_fusion)
             ] = net((clinic_image, derm_image))

            loss_fusion = torch.true_divide(
                net.criterion(logit_fusion, diagnosis_label)
                + net.criterion(logit_pn_fusion, pn_label)
                + net.criterion(logit_str_fusion, str_label)
                + net.criterion(logit_pig_fusion, pig_label)
                + net.criterion(logit_rs_fusion, rs_label)
                + net.criterion(logit_dag_fusion, dag_label)
                + net.criterion(logit_bwv_fusion, bwv_label)
                + net.criterion(logit_vs_fusion, vs_label), 8)

            loss = loss_fusion

            acc_fusion = torch.true_divide(net.metric(logit_fusion, diagnosis_label)
                                       + net.metric(logit_pn_fusion, pn_label)
                                       + net.metric(logit_str_fusion, str_label)
                                       + net.metric(logit_pig_fusion, pig_label)
                                       + net.metric(logit_rs_fusion, rs_label)
                                       + net.metric(logit_dag_fusion, dag_label)
                                       + net.metric(logit_bwv_fusion, bwv_label)
                                       + net.metric(logit_vs_fusion, vs_label), 8 * clinic_image.size(0))
            acc = acc_fusion

        val_loss += loss.item()
        val_acc += acc.item()

    val_loss = val_loss / (index + 1)
    val_acc = val_acc / (index + 1)

    return val_loss, val_acc


def test(net, test_dataloader, model_name):
    """Run evaluation on the test set using the best saved model."""
    net.set_mode('valid')  # use eval/no-dropout mode
    test_loss = 0
    test_acc = 0

    for index, (clinic_image, derm_image, meta_data, label) in enumerate(test_dataloader):

        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()

        diagnosis_label = label[0].long().cuda()
        pn_label = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label = label[7].long().cuda()

        with torch.no_grad():
            [(logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,
              logit_bwv_fusion, logit_vs_fusion)
             ] = net((clinic_image, derm_image))

            loss_fusion = torch.true_divide(
                net.criterion(logit_fusion, diagnosis_label)
                + net.criterion(logit_pn_fusion, pn_label)
                + net.criterion(logit_str_fusion, str_label)
                + net.criterion(logit_pig_fusion, pig_label)
                + net.criterion(logit_rs_fusion, rs_label)
                + net.criterion(logit_dag_fusion, dag_label)
                + net.criterion(logit_bwv_fusion, bwv_label)
                + net.criterion(logit_vs_fusion, vs_label), 8)

            acc_fusion = torch.true_divide(net.metric(logit_fusion, diagnosis_label)
                                       + net.metric(logit_pn_fusion, pn_label)
                                       + net.metric(logit_str_fusion, str_label)
                                       + net.metric(logit_pig_fusion, pig_label)
                                       + net.metric(logit_rs_fusion, rs_label)
                                       + net.metric(logit_dag_fusion, dag_label)
                                       + net.metric(logit_bwv_fusion, bwv_label)
                                       + net.metric(logit_vs_fusion, vs_label), 8 * clinic_image.size(0))

        test_loss += loss_fusion.item()
        test_acc += acc_fusion.item()

    test_loss = test_loss / (index + 1)
    test_acc = test_acc / (index + 1)

    return test_loss, test_acc


def run_train(model_name, mode, i):
    log.write('** start training here! **\n')
    best_mean_acc = 0

    for epoch in range(epochs):
        swa_lr = cosine_learning_schule[epoch]
        adjust_learning_rate(opt, swa_lr)

        # Train
        train_loss, train_acc = train(net, train_dataloader, model_name)
        log.write('Round: {}, epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}\n'.format(i, epoch, train_loss, train_acc))

        # Validation
        val_loss, val_acc = validation(net, val_dataloader, model_name)
        log.write('Round: {}, epoch: {}, Valid Loss: {:.4f}, Valid_Acc: {:.4f}\n'.format(i, epoch, val_loss, val_acc))

        # Log metrics to W&B
        wandb.log({
            'epoch': epoch,
            'round': i,
            'learning_rate': swa_lr,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
        })

        # Save best model and run test
        if val_acc > best_mean_acc:
            best_mean_acc = val_acc
            torch.save(net.state_dict(), out_dir + '/checkpoint/{:.4f}val_model.pth'.format(best_mean_acc))
            torch.save(net.state_dict(), out_dir + '/best_modal.pth')
            log.write('Current Best Mean Acc is {}\n'.format(best_mean_acc))

            # Log best val acc milestone to W&B
            wandb.run.summary['best_val_acc'] = best_mean_acc
            wandb.run.summary['best_epoch'] = epoch

    # ── Test on best checkpoint after training is done ──────────────────────
    log.write('\n** Loading best model for test evaluation **\n')
    net.load_state_dict(torch.load(out_dir + '/best_modal.pth'))

    test_loss, test_acc = test(net, test_dataloader, model_name)
    log.write('Round: {}, Test Loss: {:.4f}, Test Acc: {:.4f}\n'.format(i, test_loss, test_acc))

    wandb.log({
        'test/loss': test_loss,
        'test/acc': test_acc,
    })
    wandb.run.summary['test_acc'] = test_acc
    wandb.run.summary['test_loss'] = test_loss


if __name__ == '__main__':
    mode = 'multimodal'
    model_name = 'TDSFNet'
    shape = (224, 224)
    batch_size = 16
    num_workers = 10
    data_mode = 'Normal'
    deterministic = True
    if deterministic:
        if data_mode == 'Normal':
            random_seeds = 100
        elif data_mode == 'self_evaluated':
            random_seeds = 183
    rounds = 1
    lr = 5e-5
    epochs = 350
    swa_epoch = 50

    train_dataloader, val_dataloader, test_dataloader = generate_dataloader(shape, batch_size, num_workers, data_mode)

    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)

        print(random_seeds + i)
        log, out_dir = CreateLogger(mode, model_name, i, data_mode)

        # ── Init W&B run for this round ──────────────────────────────────────
        wandb.init(
            project='TDSFNet on SPC',           # ← change to your W&B project name
            name=f'{model_name}_round{i}',
            config={
                'model': model_name,
                'mode': mode,
                'shape': shape,
                'batch_size': batch_size,
                'lr': lr,
                'epochs': epochs,
                'swa_epoch': swa_epoch,
                'data_mode': data_mode,
                'random_seed': random_seeds + i,
            },
            reinit=True,  # allows multiple runs in the same process (multi-round)
        )

        net = TDSFNet(class_list=class_list, config=config.get_model_config()).cuda()
        wandb.watch(net, log='gradients', log_freq=100)  # optional: track gradients

        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = optimizer

        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(model_name, mode, i)

        wandb.finish()  # close the run before starting the next round