#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --port-num='8888' --batch-size=1024 --epochs=120 --lr=0.4 --loss-type='CE' --save --save-name='ResNet50_128_120_CE' &
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --port-num='8889' --batch-size=1024 --epochs=120 --lr=0.4 --loss-type='BCE' --save --save-name='ResNet50_128_120_BCE'