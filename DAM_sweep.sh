#!/bin/bash

run_single() {
    echo "Running naive-attention for $LEN"
    cargo test --release --package stream-attn-dam --lib -- $LEN test::naive_attn::tests::qkt_red_div_matvec_test --exact --nocapture > DAM_run_$LEN.txt
    
    VAR="$(egrep "finished in" DAM_run_$LEN.txt)"
    value1="${VAR#*'d in'}" # take everything after the 'd in'
    # valueA2="${valueA1%(*}" # take everything before the '('
    echo "Real Time: $value1"

    grep Simulated DAM_run_$LEN.txt
    echo ""
    rm DAM_run_$LEN.txt
}

cd /home/dam/dam-experiments/stream-attn-dam/

# The total sweep will be 8 min
LEN=512     # 0.3 sec
run_single

LEN=1024    # 1.4 sec
run_single

LEN=2048    # 5 sec
run_single

LEN=4096    # 21 sec
run_single

LEN=8192    # 84 sec
run_single

LEN=16384   # 5.5 min
run_single

LEN=32768   # 45 min
run_single