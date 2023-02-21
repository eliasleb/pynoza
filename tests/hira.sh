for order in
do
  python HIRA.py --dt 8e-12 --down_sample_time 1 \
    --n_points 100  --n_tail 50 --verbose_every 10 --plot False  \
    --filename ../../../git_ignore/ti-hira/ti_hira_v1.txt --center_x -.5 \
    --x1 .4 .5 .45 .45 --x2 0 0 .2 0 --x3 0 0 0 .2 --find_center True --scale 1e-12  --order $order --before .6 \
    > data/order-$order-ti-hira-v5.txt &
done
