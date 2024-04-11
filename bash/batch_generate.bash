#!/bin/bash

CKPTS_PATH=$1
OUTPUT_PATH=$2
TASKS=${3:-"['en2zh','zh2en','summary','essay','letter']"}
MAX_NEW_TOKENS=${4:-"2048"}
NUM_GPUS=${5:-"8"}
PROMPT_PATH=./template_prompt/llama2_zh_no_sys.json

echo Tasks: $TASKS
echo Max new tokens: $MAX_NEW_TOKENS

# if ckpt_path is a single ckpt, only generate once.
# if [[ $CKPTS_PATH == *"epoch="*".ckpt"* ]]; then
echo $CKPTS_PATH
# read name <<< "${CKPTS_PATH##*/}"
echo $OUTPUT_PATH
mkdir -p $OUTPUT_PATH

# if ckpt_path is a dir, generate all ckpts in the dir.
c=0

ls -d $CKPTS_PATH/*.ckpt

for ckpt_path in `ls -d $CKPTS_PATH/*.ckpt`; do
	# if .yaml in ckpt_path, skip
	if [[ $ckpt_path == *.yaml ]]; then
		continue
	fi
	echo $ckpt_path
	read name <<< "${ckpt_path##*/}"
	mkdir -p $OUTPUT_PATH/${name}
	echo $c,$((c+1))
	CUDA_VISIBLE_DEVICES=$c python generation/generate_with_large_lm.py $ckpt_path  $OUTPUT_PATH/${name} --tasks $TASKS --max_new_tokens $MAX_NEW_TOKENS --template_path $PROMPT_PATH &
	c=$((c+1))
	if (($c >= $NUM_GPUS)); then
	    wait
	    c=0
	fi
done

wait
