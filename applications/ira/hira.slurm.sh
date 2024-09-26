#!/bin/bash -l

for order in 11
do
  for n_points in 40 45 50
  do
    for seed in {0..15}
    do
	job_file=".job/order-${order}-n_points-${n_points}-seed-${seed}.job"
	echo "#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
module purge
module load gcc/11.3.0 intel/2021.6.0 python/3.10.4

source ../../venv/bin/activate
python HIRA.py --dt 6.4e-12 --down_sample_time 1 \
           --verbose_every 100 --plot False \
            --filename ../../../git_ignore/ti-hira/ti_hira_v3.txt --center_x -.5 \
            --x1 .8 .9 -.4 .5 --x2 0 0 0 .6 --x3 0 0 .5 0 --find_center True --coeff_derivative 0 \
            --before .6 --phase_correction 0 0 0 0 --t_max 2.8 \
            --n_points $n_points --n_tail $n_points \
            --order ${order} --scale 1e5 --shift 2 --seed $seed --save_path data \
            > data/v8-ti-hira-order-${order}-n_points-$n_points-seed-${seed}.txt" > $job_file
	sbatch $job_file
    done
  done
done
