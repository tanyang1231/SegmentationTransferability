#!/bin/bash


transfer_setting="BDD100K"
metric="OTCE" #OTCE, LEEP, LogME

python -u pixelwise_transferability.py \
--metric $metric \
--stride 4 \
--transfer_setting $transfer_setting 

