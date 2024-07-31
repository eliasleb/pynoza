#!/bin/zsh
for order in {0..16..2}
do
  python lightning.py --max_order $order > ../../../git_ignore/lightning_inverse/opt_results/max_order_$order.txt &
done
