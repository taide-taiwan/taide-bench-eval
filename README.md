# taide-bench-eval

歡迎來到 taide-bench-eval，本 project 以 GPT-4 評估 LLM 的辦公室任務，例如: 中翻英、英翻中、摘要、寫文章、寫信等。
關於以 GPT-4 評估的更多分析，請參考這篇論文: [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
[English Version of README.md](README_en.md)

## 環境與安裝

請確定你的環境已經安裝 Python 3.10。你可以透過建立虛擬環境 (Virtual Environment)並安裝必要的依賴函式庫。
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
TASKS="['en2zh','zh2en','summary','essay','letter']" # You can select a subset from ['en2zh','zh2en','summary','essay','letter']

MAX_NEW_TOKENS=2048

python generation/generate_with_large_lm.py \
$ckpt_path \
$OUTPUT_PATH/${name} \
--tasks $TASKS \
--max_new_tokens $MAX_NEW_TOKENS \
<--other generation config>
```

### 使用 GPT-4 作為基準真相

你可以使用以下指令根據 GPT-4 評估生成結果：

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
bash bash/batch_eval.bash \
  <GENERATED_JSONL_PATH>
```

這個腳本會自動在同一個資料夾中透過對應的 jsonl 檔輸出評估的 json 檔

### 兩種模型間比較

你可以比較兩個模型的輸出品質：

```bash
python evaluation/run_geval.py \
--judge_model gpt-4 \
--template_path ./prompt_template/geval.json \
--generated_result_paths "['$result1','$result2']"
--output_path test.json
```

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
