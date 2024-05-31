#!/bin/bash

src_task_list=("segnet_bdd100k_2021-08-27")
tar_task_list=("aachen")
num_repeat=1
num_pixel=10000
OT_solver="emd"
num_src_samples=20

for src_task in ${src_task_list[@]};do
for tar_task in ${tar_task_list[@]};do

echo "-----$src_task, $tar_task, src:$num_src_samples, pixel:$num_pixel, repeat:$num_repeat-------"

python -u OTCE.py \
--src_predicted_map_dir "../feature/source_feature/""$src_task""/final" \
--src_gt_map_dir "../feature/source_feature/""$src_task""/label" \
--tar_predicted_map_dir "../feature/target_feature/""$src_task/$tar_task""/final" \
--tar_gt_map_dir "../feature/target_feature/""$src_task/$tar_task""/label" \
--OT_solver $OT_solver \
--num_repeat $num_repeat \
--num_pixel $num_pixel \
--num_src_samples $num_src_samples

done
done

