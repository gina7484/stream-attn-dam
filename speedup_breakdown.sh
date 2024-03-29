#!/bin/bash

run_single() {
    cd /home/dam/dam-experiments/stream-attn-r-dam/
    echo "[Restricted DAM] Running naive attention for $LEN"
    taskset -c 1 cargo test --release --package stream-attn-dam --lib -- $LEN test::naive_attn::tests::qkt_red_div_matvec_test --exact --nocapture > rDAM_run_$LEN.txt

    cd /home/dam/dam-experiments/stream-attn-dam/
    echo "[DAM] Running naive-attention for $LEN"
    cargo test --release --package stream-attn-dam --lib -- $LEN test::naive_attn::tests::qkt_red_div_matvec_test --exact --nocapture > DAM_run_$LEN.txt
    
    cd /home/dam/dam-experiments/stream-attn-r-dam/
    VAR1="$(egrep "finished in" rDAM_run_$LEN.txt)"
    value1="${VAR1#*'d in'}" # take everything after the 'd in'
    echo "[Restricted DAM] Real Time: $value1"
    value_a="${value1%'s'*}" # take everything after the 's'
    rm rDAM_run_$LEN.txt

    cd /home/dam/dam-experiments/stream-attn-dam/
    VAR2="$(egrep "finished in" DAM_run_$LEN.txt)"
    value2="${VAR2#*'d in'}" # take everything after the 'd in'
    value_b="${value2%'s'*}" # take everything after the 's'
    echo "           [DAM] Real Time: $value2"
    rm DAM_run_$LEN.txt

    cd /home/dam/dam-experiments/stream-attn-r-dam/
    python speedup_calc.py $value_a $value_b
    echo ""
    echo ""
}


# The total sweep will be 24 min
LEN=512     # 4 sec
run_single

LEN=1024    # 16 sec
run_single

LEN=2048    # 64 sec
run_single

LEN=4096    # 4.3 min
run_single

LEN=8192    # 18 min
run_single