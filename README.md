# AI Projects

## 01. ResNet
- Goal : check train logs and convergence of ResNet 50 with BCE, CE loss
- Details
  - ResNet 50 mixed precision training with DDP on ImageNet-1K
  - All the experiments are done with 4 Titan Xp GPU

### Base Configuration
|config|value|
|:-:|:-:|
|optimizer|SGD|
|base learning rate|0.1 * batch_size / 256|
|weight decay|1e-4|
|optimizer momentum|0.9|
|learning rate schedule|cosine decay|
|epochs|120|
|warmup epochs|5|
|augmentation|RandomResizedCrop|

### Experiment Results
|batch size|loss|acc|
|:-:|:-:|:-:|
|128|BCE|77.2440|
|256|BCE|77.4340|
|512|BCE|77.3160|
|128|CE|76.8040|
|256|CE|76.4800|
|512|CE|76.5680|


## 02. Example Forgetting
- Goal : check data with no learning events among 8 randon seeds
- Details
  - ResNet 50 mixed precision training on ImageNet-1K
  - 8 random trials per every epochs
  - All the experiments are done with 1 RTX A5000 GPU

### Base Configuration
|config|value|
|:-:|:-:|
|optimizer|SGD|
|base learning rate|0.1|
|weight decay|1e-4|
|optimizer momentum|0.9|
|batch size|256|
|learning rate schedule|cosine decay|
|warmup epochs|5|
|augmentation|RandomResizedCrop|
|loss|CrossEntropy|

### Train Results
|epoch|average|1|2|3|4|5|6|7|8|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 40|75.2067|75.2860|75.1040|75.1980|75.2240|75.2220|75.2360|75.1360|75.2480|
| 60|76.1833|76.1120|76.2920|76.1860|76.1340|76.1340|76.1880|76.2720|76.1480|
| 90|76.7815|76.7920|76.6600|76.9400|76.8160|76.6480|76.8200|76.7980|76.7780|
|120|77.0607|76.9940|77.1360|76.9580|77.0180|77.0720|77.0460|77.0500|77.2120|
|180|77.1765|77.2320|77.0760|77.0200|77.1120|77.1860|77.1960|77.3560|77.2340|
|270|77.0595|77.1440|77.0620|76.9260|77.0460|77.1980|77.1040|77.0180|76.9780|
|450|76.7848|76.9220|76.6280|76.8820|76.7140|76.8800|76.7600|76.8520|76.6400|