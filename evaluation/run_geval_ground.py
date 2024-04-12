import asyncio
import json
import os
from typing import Any, Dict, List
from tqdm.asyncio import tqdm as tqdm_async
from datasets import load_dataset, Dataset
from tqdm import tqdm, trange
from concurrent.futures import ProcessPoolExecutor
from openai import AsyncOpenAI, OpenAI


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


async def get_completion(client, model, content):
    for _ in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=content,
                temperature=0.2,
                timeout=180,
                seed=42
            )
            return parse_score(resp.choices[0].message.content)
        except Exception as e:
            # time.sleep(1)
            print(e)
            await asyncio.sleep(10)
    print('ignore', content)
    return {"score": -1, "judge_response": "error"}


def get_completion_req(client, model, content):
    for _ in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=content,
                temperature=0.2
            )
            return  parse_score(resp.choices[0].message.content)
        except Exception as e:
            print(e)

    print('ignore', content)
    return {"score": -1, "judge_response": "error"}


async def get_completion_list(model, content_list, batch_size=20):
    client = AsyncOpenAI()
    result = await tqdm_async.gather(*[get_completion(client, model, content) for content in content_list], )

    return result


def get_completion_list_req(model, content_list):
    client = OpenAI()

    with ProcessPoolExecutor(max_workers=4, ) as executor:
        iters = tqdm(executor.map(get_completion_req, client, [
                     model]*len(content_list), content_list),
                     total=len(content_list))
        return list(iters)


def create_instruction(template: Dict[str, str],
                       question: str,
                       responses: List[str],
                       ground_truth: List[str] = None):
    user_dct = {'question': question.strip()}
    for i, resp in enumerate(responses):
        user_dct[f'answer_{i+1}'] = resp.replace('\u200b', '')[:2000]

    # ground_truth
    user_dct['ground_truth'] = ground_truth
    user_context = template['user'].format(**user_dct)

    return [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": user_context}
    ]


score_start = "Score:"


def parse_score(review):
    try:
        score = review.split("\n")[-1].split(":")[-1]
        if '/10' in score:
            score = score.replace('/10', '')
        return {
            "score": eval(score),
            "judge_response": review,
        }
    except Exception as e:
        print(score)
        return {
            "score": 0,
            "judge_response": review,
        }


TASK2TEMPLATE = {
    "en2zh": "template_judge/geval_translation.json",
    "zh2en": "template_judge/geval_translation.json",
    "summary": "template_judge/geval_summarization.json",
}


def geval(
    judge_model: str = 'gpt-4-0613',
    template_path: str = None,
    gen_result_path: str = 'result/model_ouput.jsonl',
    ground_dataset_path: str = 'taide/taide-bench',
    task: str = None,  # essay, letter, summary, en2zh, zh2en
    output_path: str = './result/tmp_score.json',
    req_method: str = 'async',  # req or async
    **kwargs
):
    # assert task is not None, "task is None"
    task = os.path.basename(gen_result_path).split('_')[1].split('.')[0]

    # load template
    if template_path is None:
        template_path = TASK2TEMPLATE.get(
            task, 'template_judge/geval_tw_gpt4.json')
        print('use default template', template_path)

    with open(template_path) as f:
        template = json.load(f)
    # load_data

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
                       "eval_instruction": create_instruction(template, question, model_resps),
                       })

    eval_instruction = [r.pop('eval_instruction') for r in result]

    if req_method == 'async':
        judge_resps = asyncio.run(get_completion_list(
            judge_model, eval_instruction, **kwargs), debug=True)
    else:
        judge_resps = get_completion_list_req(
            judge_model, eval_instruction, **kwargs)

    for prompt, resp in zip(result, judge_resps):
        prompt['judge_response'] = resp['judge_response']

        prompt['score'] = resp['score']

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
    fire.Fire(geval)
