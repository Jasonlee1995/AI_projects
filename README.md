# AI Projects

## ResNet
- ResNet 50 automatic mixed precision training on ImageNet

### Base Config
|config|value|
|:-:|:-:|
|optimizer|SGD|
|weight decay|1e-4|
|optimizer momentum|0.9|
|learning rate schedule|cosine decay|
|warmup epochs|5|
|augmentation|RandomResizedCrop|

- base learning rate and batch size if different per experiment
- not used nn.SyncBatchNorm on DDP but want to use, change code like below

```
net = torchvision.models.resnet50().cuda(gpu)
net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = DistributedDataParallel(net, device_ids=[gpu])
```


### Experiment Results
|logs|acc|batch size|learning rate|GPU usage|
|:-:|:-:|:-:|:-:|:-:|
|ResNet 50 - 128 - DDP|76.2580|128|0.1|use 2 GPU|
|ResNet 50 - 128 - DDP - half|76.6780|128|0.05|use 2 GPU|
|ResNet 50 - 128 - single|76.7940|128|0.1|use 1 GPU|
|ResNet 50 - 128 - single - half|77.1320|128|0.05|use 1 GPU|
|ResNet 50 - 256 - DDP|76.6380|256|0.1|use 2 GPU|
|ResNet 50 - 256 - single|77.0360|256|0.1|use 1 GPU|
