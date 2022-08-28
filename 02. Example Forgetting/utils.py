import math, os



''' scheduler '''

def cosine_scheduler(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr



def step_scheduler(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        if epoch >= milestone:
            lr *= 0.1
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr



''' logger '''

class Logger():
    def __init__(self, args):
        self.args = args
        self.log_file = os.path.join(args.log, '{}.txt'.format(args.save_name))
        self.check_first = True

    def initialize(self):
        with open(self.log_file, 'a') as l:
            l.write('-' * 50 + '\n')
            for arg in vars(self.args):
                l.write('{} : {}\n'.format(arg, vars(self.args)[arg]))
            l.write('-' * 50 + '\n' * 2)

    def update(self, dic):
        with open(self.log_file, 'a') as l:
            if self.check_first:
                l.write('epoch,' + ','.join(sorted(set(dic) - {'epoch'})) + '\n')
                self.check_first = False
            row = '{:0>3},'.format(dic['epoch'])
            for key in sorted(dic):
                if key == 'acc': row += '{:.4f},'.format(dic[key])
                elif key in ['train_time', 'val_time']: row += '{},'.format(dic[key])
                elif key in ['train_loss', 'val_loss', 'lr']: row += '{:.8f},'.format(dic[key])
            row += '\n'
            l.write(row)