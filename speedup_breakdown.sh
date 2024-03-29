#!/bin/bash

run_single() {
    echo "[Restricted DAM] Running naive attention for $LEN"
    taskset -c 1 cargo test --release --package stream-attn-dam --lib -- $LEN test::naive_attn::tests::qkt_red_div_matvec_test --exact --nocapture > rDAM_run_$LEN.txt
    
    VAR="$(egrep "finished in" rDAM_run_$LEN.txt)"
    value1="${VAR#*'d in'}" # take everything after the 'd in'
    echo "Real Time: $value1"
    echo ""
    rm rDAM_run_$LEN.txt
}

cd /home/dam/dam-experiments/stream-attn-r-dam/

# The total sweep will be 22 min
LEN=512     # 3.7 sec
run_single

LEN=1024    # 15 sec
run_single

LEN=2048    # 1 min
run_single

LEN=4096    # 4 min
run_single

LEN=8192    # 16 min
run_single