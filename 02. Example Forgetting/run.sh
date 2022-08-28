#!/bin/sh
python main_single.py --gpu=0 --epochs=270 --save-name='ResNet50_270_00' &
python main_single.py --gpu=1 --epochs=270 --save-name='ResNet50_270_01' &
python main_single.py --gpu=2 --epochs=270 --save-name='ResNet50_270_02' &
python main_single.py --gpu=3 --epochs=270 --save-name='ResNet50_270_03' &
python main_single.py --gpu=4 --epochs=270 --save-name='ResNet50_270_04' &
python main_single.py --gpu=5 --epochs=270 --save-name='ResNet50_270_05' &
python main_single.py --gpu=6 --epochs=270 --save-name='ResNet50_270_06' &
python main_single.py --gpu=7 --epochs=270 --save-name='ResNet50_270_07'
