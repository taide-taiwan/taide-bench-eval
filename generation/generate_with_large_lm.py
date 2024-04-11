import json
import os

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextGenerationPipeline, pipeline, LlamaForCausalLM
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import torch
from vllm import LLM, SamplingParams
from optimum.bettertransformer import BetterTransformer


def auto_load_dataset(dataset_path: str, dataset_split: str = 'train'):
    if dataset_path.endswith('.csv'):
        return load_dataset('csv', data_files=dataset_path)[dataset_split]
    return load_dataset(dataset_path)[dataset_split]


def add_template_to_instruction(inst: str, template: dict, tokenizer: AutoTokenizer):
    result = ''
    if 'system' in template:
        result += template['system']

    if 'user' in template:
        result += template['user'].format(BOS=tokenizer.bos_token,
                                          EOS=tokenizer.eos_token,
                                          prompt=inst.strip())
    return {'prompt': result}


def get_pipeline(path, tokenizer, eos_token=None):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    print(type(model))
    model = torch.compile(model)
    eos_token_id = tokenizer.convert_tokens_to_ids(
        eos_token) if eos_token else tokenizer.eos_token_id
    print('Model loaded')
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=eos_token_id,
                         )

    return generator


def get_vllm(
    path: str,
    tokenizer: AutoTokenizer,
    **kwargs
):
    num_dev = torch.cuda.device_count()
    model = LLM(
        path,
        tensor_parallel_size=num_dev,
    )
    model.set_tokenizer(tokenizer)

    return model


@torch.inference_mode()
def main(model_path: str,
         output_dir: str,
         tokenizer_path: str = None,
         template_path: str = 'template_prompt/llama2_zh_no_sys.json',
         dataset_path: str = 'taide/taide-bench',
         tasks: List[str] = ['summary',
                             'en2zh',
                             'zh2en',
                             'letter',
                             'essay',
                             ],
         use_fast: bool = False,
         **kwargs):
    print('model_path', model_path)
    print('output_dir', output_dir)
    print('tokenizer_path', tokenizer_path)
    print('template_path', template_path)
    print('dataset_path', dataset_path)

    os.makedirs(output_dir, exist_ok=True)

    with open(template_path) as f:
        inst_template = json.load(f)

    # clear the file
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=use_fast)
    print(tokenizer.pad_token, tokenizer.eos_token)
    tokenizer.pad_token = tokenizer.eos_token
    pipe = get_pipeline(model_path, tokenizer, inst_template.get('eos', None))

    print('tasks', tasks)
    for task in tqdm(tasks):
        if os.path.exists(f'{output_dir}/resp_{task}.jsonl'):
            continue
        print(f'Generating for {task}')
        dataset = load_dataset(dataset_path, task)['train']
        dataset = dataset.map(lambda x: add_template_to_instruction(
            x['prompt'], inst_template, tokenizer
        ), desc='Adding template to prompt')

        result = pipe(dataset['prompt'],
                      return_full_text=False,
                      max_new_tokens=kwargs.pop('max_new_tokens', 2048),
                      ** kwargs)
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
