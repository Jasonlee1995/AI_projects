# AI Projects

## ResNet
- ResNet 50 mixed precision training on ImageNet
- All the experiments are done with Titan Xp GPU

### Base Config
|config|value|
|:-:|:-:|
|optimizer|SGD|
|weight decay|1e-4|
|optimizer momentum|0.9|
|learning rate|0.1 * batch_size / 256|
|learning rate schedule|cosine decay|
|warmup epochs|5|
|augmentation|RandomResizedCrop|


### Experiment Results
|batch size|epochs|loss|acc|
|:-:|:-:|:-:|:-:|
|128|120|CE||
