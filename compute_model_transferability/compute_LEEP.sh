#!/bin/bash


src_task_list=("segnet_bdd100k_2021-08-27")
tar_task_list=("aachen")

for tar_task in ${tar_task_list[@]};do
for src_task in ${src_task_list[@]};do

echo "-----$src_task, $tar_task -------"

python -u LEEP.py \
--tar_predicted_map_dir "../feature/target_feature/""$src_task/$tar_task""/final" \
--tar_gt_map_dir "../feature/target_feature/""$src_task/$tar_task""/label" \

done
done