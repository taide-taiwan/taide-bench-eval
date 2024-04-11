# taide-bench-eval

Welcome to the taide-bench-eval repository! This tool enables you to evaluate the natural language generation (NLG) quality of models using GPT-4 with improved human alignment. For more details, please refer to the paper: [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/).

## Environment

Before you get started, ensure that you have Python 3.10 installed. You can create a virtual environment with the required dependencies using the following command:

```bash
conda env create -f environment.yml
```

## How to Use

### Generating Response

Generate text responses from your models using the provided script. Customize the parameters according to your needs:

```bash
CKPTS_PATH=<HF_CKPT_PATH>
PROMPT_PATH=./template_prompt/llama2_zh_no_sys.json
OUTPUT_PATH=<OUTPUT_JSONL_PATH>
TASKS="['en2zh','zh2en','summary']" # You can select a subset from ['en2zh','zh2en','summary','essay','letter']

MAX_NEW_TOKENS=1024

python generation/generate_with_large_lm.py \
$ckpt_path \
$OUTPUT_PATH/${name} \
--tasks $TASKS \
--max_new_tokens $MAX_NEW_TOKENS \
<--other generation config>
```

Alternatively, you can use the provided batch generation script:

```bash
bash bash/batch_generate.bash \
  <CKPT_PATH> \
  <OUTPUT_PATH>
```

or

```bash
bash bash/batch_generate.bash \
  <FOLDER_OF_CKPTS_PATH> \
  <OUTPUT_PATH>
```

### Using GPT-4 as Ground Truth

Evaluate your generated results against GPT-4 using the following command:

```bash
python evaluation/run_geval_ground.py
  --gen_result_path <generated jsonl path> \
  --output_path <output judge json file> \
  --req_method async # or req
```

* This script will automatically judge the task you need to evaluate according to the generated jsonl file (`--gen_result_path`).
* Alternatively, you can specify the task you want to evaluate using the `--task` argument.

```bash
python evaluation/run_geval_ground.py
  --gen_result_path <generated jsonl path> \
  --output_path <output judge json file> \
  --task <task name> # such as en2zh
```

- `req_method`: Choose between 'async' (using asyncio for sending requests) or 'req' (using requests with multiprocess support).

Or you can use the provided batch evaluation script:

```bash
bash bash/batch_evaluate.bash \
  <GENERATED_JSONL_PATH>
```
The script will automatically output judge json file in the same folder as the generated jsonl file.

### Comparing Two Models

Compare the quality of two models by running GEval. Evaluate the generated responses using the GEval script and specify the template and paths to the generated results:

```bash
python evaluation/run_geval.py \
--judge_model gpt-4 \
--template_path ./prompt_template/geval.json \
--generated_result_paths "['./result/llama-7b_sft_wudao-chunk-9_lima/essay_prompt.jsonl','./result/ft_lima_zh/essay_prompt.jsonl']"
--output_path test.json
```

To evaluate three models, adjust the template and paths as shown below:

```bash
python evaluation/run_geval.py \
--judge_model gpt-4 \
--template_path ./prompt_template/geval_3.json \
--generated_result_paths "['result/ft_lima_zh','result/llama-7b_sft_wudao-chunk-9_lima','result/llama-7b_sft_wudao-chunk-19_lima']" \
--output_path test.json
```

### Count Translation Language Errors

You can count translation language errors using the following command:

```bash
python evaluation/run_cnt_translation_mixed_lang.py $model_generated_jsonl $output_json
```

This script helps identify mixed Chinese and English occurrences in the responses, depending on the prompt language.


### Use Local Model to evaluate

use following command to evaluate the generated response by local model.

```bash
python evaluation/run_local_eval.py \
    --judge_model $eval_model_path \
    --template_path ./template_judge/local_tw.json \
    --gen_result_path $generated_responses_path \
    --task $task \
    --output_path $output_jsonl_path

```


## Performance

### Speed

- Without asyncio: 11 seconds per iteration
- With asyncio: 1.94 seconds per iteration
- 3 models evaluated in 6755 seconds
