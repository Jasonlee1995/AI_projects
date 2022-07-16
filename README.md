# AI Projects

## ResNet
- ResNet 50 experiment on ImageNet
- Train epoch : 120
- Warmup epoch : 5
- SGD optimizer
- Momentum : 0.9
- Weight decay : 1e-4
- Cosine decay

|Logs|Acc|Batch Size|Learning Rate|GPU usage|
|:-:|:-:|:-:|:-:|:-:|
|ResNet 50 - 128 - DDP|76.2580|128|0.1|use 2 GPU|
|ResNet 50 - 128 - DDP - half|76.6780|128|0.05|use 2 GPU|
|ResNet 50 - 128 - single|76.7940|128|0.1|use 1 GPU|
|ResNet 50 - 128 - single - half|77.1320|128|0.05|use 1 GPU|
|ResNet 50 - 256 - DDP|76.6380|256|0.1|use 2 GPU|
|ResNet 50 - 256 - single|77.0360|256|0.1|use 1 GPU|