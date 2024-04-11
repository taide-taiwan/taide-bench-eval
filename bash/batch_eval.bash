#!/bin/bash

RESP_DIR=$1
REQ_METHOD=${2:-"async"}


for resp in `ls $RESP_DIR/resp_*.jsonl`; do
    # echo $resp
    output=${resp//resp_/eval_}
    output=${output//jsonl/json}
    
    # if output exists, skip
    if [[ -f $output ]]; then
        continue
    fi

    # run evaluation
    echo $output
    python evaluation/run_geval_ground.py \
    --gen_result_path $resp \
    --output_path $output \
    --req_method $REQ_METHOD
done
