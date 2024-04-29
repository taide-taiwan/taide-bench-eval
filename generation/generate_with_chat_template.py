import json
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
from tqdm import tqdm

def auto_load_dataset(dataset_path: str, dataset_split: str = 'train'):
    if dataset_path.endswith('.csv'):
        return load_dataset('csv', data_files=dataset_path)[dataset_split]
    return load_dataset(dataset_path)[dataset_split]


def add_template_to_instruction(inst: str, tokenizer: PreTrainedTokenizerBase):
    insts = [#{ 'role': 'system', 'content': '你是一個只會說台灣繁體中文的AI助理。'}, # you can add system prompt here
                                            {'role': 'user', 'content': inst}
                                            ]
    result = tokenizer.apply_chat_template(insts, 
                                           tokenize=False,
                                           add_generation_prompt=True)
    return {'prompt': result}


def get_pipeline(path, tokenizer, eos_token=None):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16,
        device_map='auto',
        attn_implementation='sdpa',
        trust_remote_code=True)
    
    print(type(model))
    
    tokenizer.padding_side = 'left'
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token) if eos_token else tokenizer.eos_token_id
    print('Model loaded')
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=eos_token_id,
                         )

    return generator


@torch.inference_mode()
def main(model_path: str,
         output_dir: str,
         tokenizer_path: str = None,
         dataset_path: str = 'TLLM/eval-geval',
         tasks: list[str] = ['vicuna_tw',
                             'summary',
                             'en2zh',
                             'zh2en',
                             'letter',
                             'essay',
                             'vicuna_en',
                             ],
         batch_size: int = 4,
         use_fast: bool = True,
         **kwargs):
    print('model_path', model_path)
    print('output_dir', output_dir)
    print('tokenizer_path', tokenizer_path)
    print('tasks', tasks)

    os.makedirs(output_dir, exist_ok=True)

    # clear the file
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=use_fast
    )
    print(tokenizer.pad_token, tokenizer.eos_token)
    tokenizer.pad_token = tokenizer.eos_token
    pipe = get_pipeline(model_path, tokenizer)

    for task in tqdm(tasks):
        if os.path.exists(f'{output_dir}/resp_{task}.jsonl'):
            continue
        print(f'Generating for {task}')
        dataset = load_dataset(dataset_path, task)['train']
        dataset = dataset.map(
            lambda x: add_template_to_instruction(
                x['prompt'], tokenizer),
            desc='Adding template to prompt')
        result = pipe(dataset['prompt'],
                      return_full_text=False,
                      max_new_tokens=kwargs.pop('max_new_tokens', 4096),
                      batch_size=batch_size,
                      ** kwargs)
        print(result[:4])
        assert len(result) == len(dataset)

        output_path = f'{output_dir}/resp_{task}.jsonl'
        with open(output_path, 'w') as f:
            for r, x in zip(result, dataset):
                dct = {
                    'qid': x['qid'],
                    'model': model_path,
                    'prompt': x['prompt'],
                    'resp': r[0]['generated_text'],
                }
                x = {k: v for k, v in x.items() if isinstance(v, str)}
                x.update(dct)

                f.write(
                    json.dumps(x, ensure_ascii=False) + '\n'
                )


if __name__ == '__main__':
    import fire

    fire.Fire(main)
