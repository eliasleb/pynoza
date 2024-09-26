#!/bin/zsh
for order in {21..29..2}
do
  python epfl_logo.py --order "$order" --pause_plot --plot_max 1e-6 &
done
