import os
from mandelli.utils import architectures
from cl_utils.data import GanDataset, SDDataset
import albumentations as A
import albumentations.pytorch as Ap
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from collections import OrderedDict
from cl_utils.ewc import EWC, test, ewc_train
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse

torch.manual_seed(21)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmbda', default=1000, type=float)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--ewc_correction', action='store_true')
    parser.add_argument('--finetuning', action='store_true')
    parser.add_argument('--importance_method', type=str,
                        choices=['fisher', 'fisher_complete', 'mas'], default='fisher')
    parser.add_argument('--model', type=str, required=False)

    args = parser.parse_args()

    lmbda = args.lmbda
    optimizer_name = args.optimizer
    lr = args.lr
    ewc_correction = args.ewc_correction
    finetuning = args.finetuning
    importance_method = args.importance_method
    model_name = args.model

    model_letter = model_name if model_name is not None else 'A'
    checkpoint_dir = 'weigths'
    checkpoint_path = os.path.join(checkpoint_dir, f'method_{model_letter}.pth')
    device = 'cuda'
    num_workers = 10  # cpu_count()
    epochs = 500
    patience = 50
    batch_size = 128

    tensorboard_logdir = 'tb_data'
    suffix = f'{optimizer_name}_{lr}_{lmbda}'
    if ewc_correction:
        suffix += '_correct'
    if finetuning:
        suffix += '_finetuning'
    if importance_method:
        suffix += f'_{importance_method}'
    if model_name:
        suffix += f'_model-{model_letter}'

    save_path = os.path.join(checkpoint_dir, suffix, 'bestval.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tb = SummaryWriter(log_dir=os.path.join(tensorboard_logdir, suffix))

    network_class = getattr(architectures, 'EfficientNetB4')
    net = network_class(n_classes=2, pretrained=False).eval().to(device)
    print(f'Loading model {model_letter}...')
    state_tmp = torch.load(checkpoint_path, map_location='cpu')

    if 'net' not in state_tmp.keys():
        state = OrderedDict({'net': OrderedDict()})
        [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
    else:
        state = state_tmp
    incomp_keys = net.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    print('Model loaded!\n')

    if finetuning:
        for param in net.efficientnet.parameters():
            param.requires_grad = False

    net_normalizer = net.get_normalizer()  # pick normalizer from last network
    transform = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std),
        Ap.transforms.ToTensorV2()
    ]

    cropper = A.RandomCrop(width=128, height=128, always_apply=True, p=1.)
    train_trans = A.Compose([cropper] + transform)
    if optimizer_name == 'adam':
        optimizer = Adam(net.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(net.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    criterion = torch.nn.CrossEntropyLoss()

    test_gan_dataset = GanDataset(transform=train_trans, phase='test')
    val_gan_dataset = GanDataset(transform=train_trans, phase='val')
    test_gan_dataloader = DataLoader(test_gan_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                     num_workers=num_workers)
    val_gan_dataloader = DataLoader(val_gan_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                    num_workers=num_workers)

    sd_train_dataset = SDDataset(transform=train_trans, phase='train')
    sd_val_dataset = SDDataset(transform=train_trans, phase='val')
    sd_test_dataset = SDDataset(transform=train_trans, phase='test')
    sd_train_dataloader = DataLoader(sd_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                                     num_workers=num_workers)
    sd_val_dataloader = DataLoader(sd_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                   num_workers=num_workers)
    sd_test_dataloader = DataLoader(sd_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers)

    # initialize lmbda with GAN
    ewc = EWC(model=net, dataloader=test_gan_dataloader, lmbda=lmbda, importance_method=importance_method)

    # initial testing
    # acc_gan, _ = test(model=net,
    #                   data_loader=test_gan_dataloader,
    #                   dataset_name='GAN',
    #                   criterion=criterion,
    #                   ewc=ewc,
    #                   phase='test',
    #                   lr=optimizer.param_groups[0]['lr'])
    # print(f'Accuracy on GAN: {acc_gan: .3f}')
    # acc_sd, _ = test(model=net,
    #                  data_loader=sd_test_dataloader,
    #                  dataset_name='SD',
    #                  criterion=criterion,
    #                  ewc=ewc,
    #                  phase='test',
    #                  lr=optimizer.param_groups[0]['lr'])
    # print(f'Accuracy on SD: {acc_sd: .3f}')

    # train
    best_val_loss = 10000
    global_step = 0
    early_stopping_counter = 0
    for epoch in range(epochs):
        training_loss, cross_entropy, penalty, global_step = ewc_train(model=net,
                                                                       optimizer=optimizer,
                                                                       criterion=criterion,
                                                                       data_loader=sd_train_dataloader,
                                                                       ewc=ewc,
                                                                       epoch=epoch + 1,
                                                                       tb=tb,
                                                                       global_step=global_step,
                                                                       ewc_correction=ewc_correction)
        print(f'\nTraining loss epoch {epoch + 1}: {training_loss: .5f}'
              f'\nCross Entropy epoch {epoch + 1}: {cross_entropy: .5f}',
              f'\nPenalty epoch {epoch + 1}: {penalty: .5f}')
        acc_val, loss_val = test(model=net,
                                 data_loader=sd_val_dataloader,
                                 dataset_name='SD',
                                 phase='validation',
                                 criterion=criterion,
                                 ewc=ewc,
                                 tb=tb,
                                 global_step=global_step,
                                 lr=optimizer.param_groups[0]['lr'])
        acc_val_gan, _ = test(model=net,
                              data_loader=val_gan_dataloader,
                              dataset_name='GAN',
                              phase='validation',
                              criterion=criterion,
                              ewc=ewc,
                              tb=tb,
                              global_step=global_step,
                              lr=optimizer.param_groups[0]['lr'])
        print(f'Validation Accuracy on SD: {acc_val: .3f}')
        print(f'Validation Accuracy on GAN: {acc_val_gan: .3f}')

        if loss_val < best_val_loss:
            print(f'Saving best model epoch {epoch + 1}')
            torch.save(net.state_dict(), save_path)
            best_val_loss = loss_val
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f'Early stopping: {early_stopping_counter}/{patience}')

        if early_stopping_counter == patience:
            print(f'Terminating training on epoch {epoch + 1} due to early stopping')
            break

        scheduler.step(loss_val)

    # load best model
    net.load_state_dict(torch.load(save_path))

    # repeat testing
    print('\nTesting:')
    acc_gan, _ = test(model=net,
                      data_loader=test_gan_dataloader,
                      dataset_name='GAN',
                      criterion=criterion,
                      ewc=ewc,
                      phase='test',
                      lr=optimizer.param_groups[0]['lr'])
    print(f'Accuracy on GAN: {acc_gan: .3f}')
    acc_sd, _ = test(model=net,
                     data_loader=sd_test_dataloader,
                     dataset_name='SD',
                     criterion=criterion,
                     ewc=ewc,
                     phase='test',
                     lr=optimizer.param_groups[0]['lr'])
    print(f'Accuracy on SD: {acc_sd: .3f}')


if __name__ == '__main__':
    main()
