#

# run time series experiment
# discriminator : 

python main.py --checkpoint_dir time_teacher_4000_z_dim_50_c_1e-4/ --teachers_batch 40 --batch_teachers 100 --dataset time --train --sigma_thresh 3000 --sigma 1000 --step_size 1e-4 --max_eps 10 --nopretrain --z_dim 50 --batch_size 64 --data_dir ./data

