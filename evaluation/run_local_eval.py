from typing import Dict, List
from abc import abstractmethod
from string import Template
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, pipeline, LlamaTokenizer, LlamaForCausalLM
import re

import json
from tqdm import tqdm
import torch
import json
import argparse
import yaml
from pathlib import Path
import os
from datasets import load_dataset

torch.cuda.manual_seed(42)
torch.manual_seed(42)


def get_score(response):
    if '[score result]' in response: #score
        try:
            return get_score(re.findall("\[score result\].*", response)[0][14:])
        except:
            print('response: {}'.format(response))
            return -1.0
    elif '[score]' in response:
        try:
            return get_score(re.findall("\[score\].*", response)[0][7:])
        except:
            print('response: {}'.format(response))
            return -1.0
    else:
        try:
            try:
                return float(re.findall("\d+\.\d+", response)[0])
            except:
                return float(re.findall("\d+", response)[0])
        except:
            print('response: {}'.format(response))
            return -1.0


def extract_response(response):
    if '[/INST]' in response: # llama 2 template
        rindex = response.find('[/INST]')
        return response[rindex + 7:]
    elif 'ASSISTANT:' in response: # vicuna template
        rindex = response.find('ASSISTANT:')
        return response[rindex + 10:]
    else:
        raise NotImplementedError('Template not seen')


class ResponseModel:
    @abstractmethod
    def get_response(input_text, **kwargs) -> Dict[str, any]:
        return NotImplementedError

class AutoHFResponseModel(ResponseModel):
    def __init__(
            self,
            hf_model_path: str=None,
            device: str='cuda',
            device_map: str='auto',
            load_from_config: bool=False,
            hf_model_config: dict={},
            hf_generation_config: dict={},
            hf_pipeline_config: dict={},
            **kwargs
        ) -> None:
        self.device = device
        self.device_map = device_map
        self.geneartion_cfg = hf_generation_config
        if load_from_config:
            self.config = AutoConfig.from_pretrained(hf_model_config)
            #self.model = AutoModelForCausalLM.from_pretrained(self.config, torch_dtype=torch.float16) #quantization
            #self.config['max_memory'] = {2: "32GB", 3: "32GB"}
            self.model = AutoModelForCausalLM.from_pretrained(self.config) #quantization
        else:
            # hf_model_config['pretrained_model_name_or_path'] = hf_model_config
            hf_model_config['device_map'] = device_map
            #hf_model_config['max_memory'] = {2: "32GB", 3: "32GB"}
            #hf_model_config['torch_dtype'] = torch.float16 #quantization
            self.config = hf_model_config
            self.model = AutoModelForCausalLM.from_pretrained(hf_model_path, **self.config)
        # self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        self.pipeline = pipeline("text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **hf_pipeline_config,
        )
        # Create a text generation pipeline
        # text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

        # # Generate text
        # prompt = "Once upon a time"
        # generated_text = text_generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']

    def get_response(self, input_text: str, **kwargs) -> Dict[str, any]:
        # encoded_inputs = self.tokenizer(input_text, return_tensor="pt").to(self.device)
        # Setting generation config; greedy decoding by default
        generation_cfg = {'do_sample': kwargs.get('do_sample', False),
                          'temperature': kwargs.get('temperature', 0),
                          'max_new_tokens': kwargs.get('max_new_tokens', 128),
                          'use_cache': True,}
        with torch.no_grad():
            output = self.pipeline(input_text, **generation_cfg)
        #torch.cuda.empty_cache()
        # sequences = output.sequences
        # all_decoded_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        completions = []
        for decoded_text in output:
            completions.append(decoded_text['generated_text'])

        return {"completions": completions}

    def get_batch_response(self, key_dataset: List[str], **kwargs) -> Dict[str, any]:
        # encoded_inputs = self.tokenizer(input_text, return_tensor="pt").to(self.device)
        # Setting generation config; greedy decoding by default
        generation_cfg = {'do_sample': kwargs.get('do_sample', False),
                          'temperature': kwargs.get('temperature', 0),
                          'max_new_tokens': kwargs.get('max_new_tokens', 128),
                          'use_cache': True,}
        with torch.no_grad():
            output = self.pipeline(key_dataset, batch_size=1, **generation_cfg)
        # sequences = output.sequences
        # all_decoded_text = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        completions = []
        for decoded_text in tqdm(output, desc='inference pipeline ...'):
            completions += list(map(lambda x: x['generated_text'], decoded_text))
            torch.cuda.empty_cache()

        return {"completions": completions}

def create_instruction_local(template: Dict[str, str],
                       question: str,
                       responses: List[str],
                       ground_truth: List[str] = None):
    example = template['example']
    fexample = example.format(
        question=question
        , response=responses[1]
    )

    prompt = template["prompt"].format(
        instruction = template['instruction']
        , task_detail = template['task_detail'][template['category']]
        , example = fexample
    )
    return prompt


score_start = "Score:"


TASK2TEMPLATE = {
    "en2zh": "template_judge/geval_translation.json",
    "zh2en": "template_judge/geval_translation.json",
    "summary": "template_judge/geval_summarization.json",
}


def local_eval(
    judge_model: str = 'ev.1.1.0',
    template_path: str = None,
    gen_result_path: str = 'result/model_ouput.jsonl',
    ground_dataset_path: str = 'taide/taide-bench',
    task: str = None,  # essay, letter, summary, en2zh, zh2en
    output_path: str = './result/tmp_score.json',
    **kwargs
):
    # assert task is not None, "task is None"
    #task = os.path.basename(gen_result_path).split('_')[1].split('.')[0]

    # load template
    if template_path is None:
        template_path = TASK2TEMPLATE.get(
            task, 'template_judge/geval_tw_gpt4.json')
        print('use default template', template_path)

    with open(template_path) as f:
        template = json.load(f)
    # load_data #pass
    ground_ds = load_dataset(
        ground_dataset_path, task)['train']
    gen_ds = load_dataset('json', data_files=gen_result_path)['train']
    ground_ds, gen_ds = ground_ds.sort('qid'), gen_ds.sort('qid')

    resp_dct = {
        ground_ds['model'][0]: ground_ds,
        gen_ds['model'][0]: gen_ds
    }

    # get prompt
    result = []
    resp_names = list(resp_dct.keys())
    num_samples = len(list(resp_dct.values())[0])
    for i in range(num_samples):
        qid = resp_dct[resp_names[0]][i]['qid']
        # check same qid
        for name in resp_names:
            assert resp_dct[name][i][
                'qid'] == qid, f"qid not match, {qid} vs {resp_dct[name][i]['qid']}, {name}, {resp_names[0]} {i}"

        question = resp_dct[resp_names[0]][i]['prompt']
        model_resps = [resp_dct[name][i]['resp'] for name in resp_names]

        result.append({"qid": qid,
                       "question": question,
                       "model_responses": {name: resp_dct[name][i]['resp'] for name in resp_names},
                       "eval_instruction": create_instruction_local(template, question, model_resps),
                       })

    eval_instruction = [r.pop('eval_instruction') for r in result]

    #load evaluation model
    rm = AutoHFResponseModel(hf_model_path=judge_model, device = 'auto')

    # score
    for r, input_prompt in tqdm(zip(result, eval_instruction)):
        output = rm.get_response(input_prompt)['completions'][0]
        response = extract_response(output)

        score = get_score(response)

        r['score'] = score
        r['judge_response'] = input_prompt

    # overall score
    overall_scores = 0
    result = [r for r in result if r['score'] != -1]
    for r in result:
        overall_scores += r['score']

    # save result
    with open(output_path, 'w') as f:
        json.dump({"overall": {"score": overall_scores,
                               "avg_score": overall_scores / len(result)},
                  "result": result},
                  f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import fire
    fire.Fire(local_eval)
