for order in 1 2 3 4 5
do
  python HIRA.py --dt 8e-12 --down_sample_time 1 \
    --n_points 70  --n_tail 35 --verbose_every 50 --plot False  \
    --filename ../../../git_ignore/ti-hira/ti_hira_v1.txt --center_x -.5 \
    --x1 .4 .5 .5 .5 --x2 0 0 .2 0 --x3 0 0 0 .2 --find_center True --scale 1e-12 --coeff_derivative 0 --before .6 \
      --order $order > data/order-$order-ti-hira-70-pts.txt &
done
