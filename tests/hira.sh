for order in 0 1 2 3 4 5
do
  python HIRA.py --dt 1e-10 --down_sample_time 4 --r_obs_min 1 --r_obs_max 1.5 --n_r_obs 2 --n_theta_obs 5 \
    --n_phi_obs 8 --phi_obs_min 3.24 --phi_obs_max 6.18 --theta_obs_min 0.1 --theta_obs_max 3.04 --tol 1e-7 \
    --n_points 15 --n_tail 10 --verbose_every 10 --plot False --scale 1e5 --find_center True --max_global_tries 1 \
    --filename ../../../git_ignore/GLOBALEM/hira_v12.txt --order $order > data/globalem_order_$order.txt &
done
