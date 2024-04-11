# taide-bench-eval

歡迎來到 taide-bench-eval 這個專案！這工具可以讓你評估測試自然語言產生品質的模型，透過 GPT-4 並與人類偏好對齊。想了解更多細節可以參考以下論文 [Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality](https://lmsys.org/blog/2023-03-30-vicuna/).

[English Version of README.md](README_en.md)

## 環境與安裝

在開始之前，請確定你的環境已經安裝 Python 3.10。你可以透過建立虛擬環境 (Virtual Environment)並安裝必要的依賴函式庫。
依賴的函式庫可以透過以下指令安裝：

```bash
conda env create -f environment.yml
```

## 如何使用

### 生成回應

你可以透過提供的腳本模型的生成文字回應。以下是一些客製化參數：

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

或者你可以使用以下腳本批次生成結果：

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

### 使用 GPT-4 作為基準真相

你也可以根據以下指令根據 GPT-4 評估生成結果：

```bash
python evaluation/run_geval_ground.py
  --gen_result_path <generated jsonl path> \
  --output_path <output judge json file> \
  --req_method async # or req
```

* 這個腳本將根據生成的 jsonl 檔（--gen_result_path）自動判斷您需要評估的任務。
* 或者，您可以使用 `--task` 參數指定您想要評估的任務。

```bash
python evaluation/run_geval_ground.py
  --gen_result_path <generated jsonl path> \
  --output_path <output judge json file> \
  --task <task name> # such as en2zh
```

- `req_method`：選擇使用 'async' (使用asyncio發送請求) 或 'req'（使用支援多線程的 requests）。

或者您可以使用以下腳本批次評估：

```bash
bash bash/batch_evaluate.bash \
  <GENERATED_JSONL_PATH>
```
The script will automatically output judge json file in the same folder as the generated jsonl file.
這個腳本會自動在同一個資料夾中透過對應的 jsonl 檔輸出評估的 json 檔

### 兩種模型間比較

你可以透過執行 GEval 來比較兩個模型的輸出品質。他會透過 GEval 的腳本評估生成的結果，這也支援使用特定的模板、路徑來生成結果：

```bash
python evaluation/run_geval.py \
--judge_model gpt-4 \
--template_path ./prompt_template/geval.json \
--generated_result_paths "['./result/llama-7b_sft_wudao-chunk-9_lima/essay_prompt.jsonl','./result/ft_lima_zh/essay_prompt.jsonl']"
--output_path test.json
```

如果你想一次評估三顆模型，則可以透過下方腳本進行評估：

```bash
python evaluation/run_geval.py \
--judge_model gpt-4 \
--template_path ./prompt_template/geval_3.json \
--generated_result_paths "['result/ft_lima_zh','result/llama-7b_sft_wudao-chunk-9_lima','result/llama-7b_sft_wudao-chunk-19_lima']" \
--output_path test.json
```

### 計算翻譯語言錯誤率

你可以透過以下的指令計算語言錯誤率：

```bash
python evaluation/run_cnt_translation_mixed_lang.py $model_generated_jsonl $output_json
```

透過以下的腳本可以幫助你根據你的 Prompt 指出中英混在你模型回應的情況。
This script helps identify mixed Chinese and English occurrences in the responses, depending on the prompt language.


### 使用自訓練評估模型進行評估

你可以透過以下的的指令用自訓練評估模型來評估生成的結果

```bash
python evaluation/run_local_eval.py \
    --judge_model $eval_model_path \
    --template_path ./template_judge/local_tw.json \
    --gen_result_path $generated_responses_path \
    --task $task \
    --output_path $output_jsonl_path

```


## 表現

### 速度

- 不使用 asyncio: 約 11 秒/迭代
- 使用 asyncio: 1.94 秒/迭代
- 三個模型評估約 6755 秒
