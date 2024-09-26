#!/bin/zsh
COMMANDS=()
scale=1e6
for n_points in 100; do
  n_tail=10
  for order in {0..20..2}; do
    for seed in {0..9}; do
      COMMANDS+=("python lightning.py --max_order $order --n_points $n_points --n_tail $n_tail --order_scale $scale --seed $seed > ../../../git_ignore/lightning_inverse/opt_results/max_order_${order}_scale_${scale}_n_points_${n_points}_n_tail_${n_tail}_seed_${seed}.txt &")
    done
  done
done

batch_size=6
i=1
total_commands=${#COMMANDS}
PIDS=()

while (( i < total_commands || ${#PIDS[@]} > 0 )); do
    # Launch new processes until we reach the batch size
    while (( ${#PIDS[@]} < batch_size && i < total_commands )); do
        date +%d.%m.%y-%H:%M:%S
        echo "Executing: ${COMMANDS[i]}"
        eval "${COMMANDS[i]}"
        PIDS+=($!)
        ((i++))
    done

    for pid in $PIDS; do
        if ! kill -0 $pid 2>/dev/null; then
            # Process has finished, remove it from the list
            PIDS=("${PIDS[@]:#$pid}")
        fi
    done

    # Sleep for a short time to avoid busy-waiting
    sleep 1
done
