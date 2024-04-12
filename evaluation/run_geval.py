import asyncio
import json
import time
import os
from collections import defaultdict
from typing import Any, Dict, List

from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm
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
    return {"score": [-1], "judge_response": "error"}


async def get_completion_list(model, content_list):
    client = AsyncOpenAI()
    return await tqdm_async.gather(*[get_completion(client, model, content) for content in content_list],)


def get_completion_list_req(model, content_list):
    result = []
    for content in tqdm(content_list):
        while 1:
            try:
                result.append(asyncio.run(get_completion(model, content)))
                break
            except Exception as e:
                print(e)
                time.sleep(5)
                print("retry")
    return result


def create_instruction(template: Dict[str, str], question: str, responses: List[str], ground_truth: List[str] = None):
    user_dct = {'question': question}
    for i, resp in enumerate(responses):
        user_dct[f'answer_{i+1}'] = resp

    # ground_truth
    user_dct['ground_truth'] = ground_truth
    user_context = template['user'].format(**user_dct)

    return [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": user_context}
    ]


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        sp = [s for s in sp if s != ""]
        sp = [float(s) for s in sp]
        return {
            "score": sp,
            "judge_response": review,
        }
    except Exception as e:
        return {
            "score": [0],
            "judge_response": review,
        }


def geval(
    judge_model: str = 'gpt-4-0613',
    template_path: str = './prompt_template/geval.json',
    generated_result_paths: List[str] = [
        'result/tmp.jsonl', 'result/tmp2.jsonl', 'result/tmp3.jsonl'],
    output_path: str = './result/tmp_geval.json',
    req_method: str = 'req',  # req or async
    **kwargs
):
    assert isinstance(generated_result_paths, list) and len(
        generated_result_paths) > 1

    # load template
    with open(template_path) as f:
        template = json.load(f)

    # load_data
    resp_dct = {}
    for path in generated_result_paths:
        # name = os.path.basename(path).split('.')[0]
        with open(path) as f:
            data = sorted([json.loads(line) for line in f],
                          key=lambda x: x['qid'])
            # data = sorted([json.loads(line) for line in f],
            # key=lambda x: x['qid'])
            resp_dct[path] = data
    print(resp_dct.keys(), generated_result_paths)
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

        if 'ground_truth' in resp_dct[resp_names[0]][i]:
            ground_truth = resp_dct[resp_names[0]][i]['ground_truth']
        else:
            ground_truth = None

        result.append({"qid": qid,
                       "question": question,
                       "model_responses": {name: resp_dct[name][i]['resp'] for name in resp_names},
                       "eval_instruction": create_instruction(template, question, model_resps, ground_truth),
                       "ground_truth": ground_truth,
                       })

    eval_instruction = [r.pop('eval_instruction') for r in result]

    if req_method == 'async':
        judge_resps = asyncio.run(get_completion_list(
            judge_model, eval_instruction, **kwargs))
    else:
        judge_resps = get_completion_list_req(
            judge_model, eval_instruction, **kwargs)

    for prompt, resp in zip(result, judge_resps):
        scores = resp['score']
        prompt['judge_response'] = resp['judge_response']
        if len(scores) != len(resp_names):
            scores = [0] * len(resp_names)

        prompt['scores'] = {resp_names[i]: scores[i]
                            for i in range(len(scores))}
        best_index = scores.index(max(scores))
        if scores.count(max(scores)) > 1:
            prompt['best'] = 'tie'
        else:
            prompt['best'] = resp_names[best_index]

    # overall score
    overall_scores, overall_best = defaultdict(float), defaultdict(int)
    for r in result:
        for k, v in r['scores'].items():
            overall_scores[k] += v
        overall_best[r['best']] += 1

    # save result
    with open(output_path, 'w') as f:
        json.dump({"overall": {"score": overall_scores,
                               "avg_score": {k: v / num_samples for k, v in overall_scores.items()},
                               "best": overall_best},
                  "result": result},
                  f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    import fire
    fire.Fire(geval)
