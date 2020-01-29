import torch


def get_optimizer_config(model, name, lr, weight_decay):
    if name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def get_lr_scheduler(name, optimizer, lr_decay, last_epoch=-1):
    if name == 'sgd':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay, gamma=0.1, last_epoch=last_epoch)
    elif name == 'adam':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay, gamma=0.1, last_epoch=last_epoch)
    return lr_scheduler