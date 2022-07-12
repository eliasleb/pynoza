for order in 0 1 2 3 4 5 6 7 8 9 10
do
 python mikheev.py --down_sample_time 1 --tol 1e-8 \
    --n_points 10 --n_tail 10 --verbose_every 10 --plot False --scale 1e4 --find_center False \
    --order $order --norm 2 \
    > data/order-$order-mikheev_v5.txt &
done