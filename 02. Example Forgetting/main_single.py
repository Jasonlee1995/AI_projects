import data, utils
import argparse, os, time

import torch, torchvision

import numpy as np
import torch.nn as nn
import torch.optim as optim


torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--gpu',           default=0, type=int)

parser.add_argument('--data-dir',      default='/mnt/ssd1/ImageNet', type=str)
parser.add_argument('--batch-size',    default=256, type=int)
parser.add_argument('--num-workers',   default=8, type=int)

parser.add_argument('--epochs',        default=180, type=int)
parser.add_argument('--warmup-epochs', default=5, type=int)
parser.add_argument('--min-lr',        default=1e-8, type=float)

parser.add_argument('--optim',         default='SGD', type=str)
parser.add_argument('--lr',            default=0.1, type=float)
parser.add_argument('--momentum',      default=0.9, type=float)
parser.add_argument('--weight-decay',  default=0.0001, type=float)

parser.add_argument('--save-name',     default='ResNet50', type=str)
parser.add_argument('--print-freq',    default=500, type=int)
parser.add_argument('--log',           default='./logs/', type=str)
parser.add_argument('--npy',           default='./npys/', type=str)
args = parser.parse_args()



def main(args):
    gpu = args.gpu
    
    # dataloader
    train_dataset = data.ImageNetDB(os.path.join(args.data_dir, 'train'), transform=data.train_transform())
    train_loader  = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    
    val_dataset = data.ImageNetDB(os.path.join(args.data_dir, 'val'), transform=data.val_transform())
    val_loader  = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=False)

    # model
    net = torchvision.models.resnet50().cuda(gpu)

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # critertion
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # logger
    logger = utils.Logger(args)
    logger.initialize()
    
    # save event logs
    train_events = torch.zeros(len(train_dataset), args.epochs).to(int).cuda(gpu)
        
    # scaler
    scaler = torch.cuda.amp.GradScaler()

    # epoch start
    for epoch in range(args.epochs):
        
        # train
        net.train()
        print('Epoch {} Train Started...'.format(epoch))

        train_loss = []
        train_start = time.time()
        for i, (idxs, imgs, labels) in enumerate(train_loader):
            lr = utils.cosine_scheduler(optimizer, epoch + i/len(train_loader), args)
            idxs, imgs, labels = idxs.cuda(gpu), imgs.cuda(gpu), labels.cuda(gpu)
            
            with torch.no_grad():
                net.eval()
                with torch.cuda.amp.autocast():
                    output = net(imgs)
                    predict = torch.argmax(output, 1)
                    train_events[idxs, epoch] = (predict == labels).to(int)
                net.train()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = net(imgs)
                loss = criterion(output, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())

            if i % args.print_freq == 0:
                print('Iteration : {:0>5}   LR : {:.6f}   Train Loss : {:.6f}'.format(i, lr, train_loss[-1]))

        train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start))
        
        
        # val
        net.eval()
        print('Epoch {} Val Started...'.format(epoch))
        
        val_start = time.time()
        with torch.no_grad():
            val_loss, correct = [], 0
            for idxs, imgs, labels in val_loader:
                idxs, imgs, labels = idxs.cuda(gpu), imgs.cuda(gpu), labels.cuda(gpu)
                
                with torch.cuda.amp.autocast():
                    output = net(imgs)
                    loss = criterion(output, labels)

                predict = torch.argmax(output, 1)
                c = (predict == labels).sum()
                
                correct += c.item()
                val_loss.append(loss.item())
                    
        val_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - val_start))
        

        # print results
        train_loss = sum(train_loss) / len(train_loss)
        val_loss = sum(val_loss) / len(val_loss)
        acc = 100 * correct / len(val_dataset)
        
        print(); print('-' * 50)
        print('Epoch : {}'.format(epoch))
        print('Acc : {:.4f}'.format(acc))
        print('Train Time : {}   Val Time : {}'.format(train_time, val_time))
        print('Train Loss : {:.8f}   Val Loss : {:.8f}'.format(train_loss, val_loss))
        print('-' * 50); print()

        # update log
        logger.update({'epoch' : epoch,
                       'lr' : lr,
                       'acc' : acc,
                       'train_time' : train_time,
                       'train_loss' : train_loss,
                       'val_time' : val_time,
                       'val_loss' : val_loss,})
            
    
    # save event logs
    train_events = train_events.cpu().numpy()
    train_log = os.path.join(args.npy, '{}_train.npy'.format(args.save_name))
    with open(train_log, 'wb') as l:
        np.save(l, train_events)



if __name__ == '__main__':
    print('Using GPU {}'.format(args.gpu))
    main(args)