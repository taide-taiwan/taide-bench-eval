# Welcome to taide-bench-eval

This project aims to evaluate the performance of large language models (LLMs) on office tasks leveraging GPT-4, including Chinese-to-English translation, English-to-Chinese translation, summarization, essay writing, and letter writing. For more analysis on the evaluation using GPT-4, please refer to this paper: [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

[English Version of README.md](README_en.md)

## Environment and Installation

You can create a virtual environment using conda and install the required dependencies.

Install the dependencies using the following commands:

```bash
conda env create -f environment.yml --solver libmamba
conda activate taide-bench
```

## How to Use

### Generate Responses

You can use the provided script to generate text responses from the models. Here are some customizable parameters:

```bash
CKPTS_PATH=<HF_CKPT_PATH>
PROMPT_PATH=./template_prompt/llama2_zh_no_sys.json
OUTPUT_PATH=<OUTPUT_JSONL_PATH>
TASKS="['en2zh','zh2en','summary','essay','letter']"  # You can select a subset from ['en2zh','zh2en','summary','essay','letter']
MAX_NEW_TOKENS=2048

python generation/generate_with_large_lm.py \
$ckpt_path \
$OUTPUT_PATH/${name} \
--tasks $TASKS \
--max_new_tokens $MAX_NEW_TOKENS \
<--other generation config>
```

### Evaluate Using GPT-4 as the Ground Truth

You can use the following command to evaluate the generated results based on GPT-4:

```bash
python evaluation/run_geval_ground.py --gen_result_path <generated jsonl path> \
                                     --output_path <output judge json file> \
                                     --req_method async  # or req
```

- This script will automatically determine the tasks to be evaluated based on the generated jsonl file (--gen_result_path).
- Alternatively, you can use the `--task` parameter to specify the task you want to evaluate.

```bash
python evaluation/run_geval_ground.py --gen_result_path <generated jsonl path> \
                                     --output_path <output judge json file> \
                                     --task <task name>  # such as en2zh
```

- `req_method`: Choose 'async' (using asyncio to send requests) or 'req' (using multithreaded requests).

Alternatively, you can use the following batch script to evaluate multiple results:

```bash
bash bash/batch_eval.bash \
<GENERATED_JSONL_PATH>
```

This script will automatically output the evaluation JSON files in the same folder as the corresponding jsonl files.

### Compare Between Two Models

You can compare the output quality of two models:

```bash
python evaluation/run_geval.py \
--judge_model gpt-4 \
--template_path ./prompt_template/geval.json \
--generated_result_paths "['$result1','$result2']" --output_path test.json
```