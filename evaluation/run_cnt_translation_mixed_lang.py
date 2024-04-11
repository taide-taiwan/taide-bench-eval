import re
import json
from collections import Counter


EN_PAT = re.compile(r'[a-zA-Z]+', re.MULTILINE)
ZH_PAT = re.compile(r'[\u4e00-\u9fff]+', re.MULTILINE)


def cnt_english(text):
    en_words = EN_PAT.findall(text)
    en_words = [w.strip().split() for w in en_words]
    # flatten
    en_words = [w for ws in en_words for w in ws]

    return len(en_words)


def cnt_chinese(text):
    return sum(map(len, ZH_PAT.findall(text)))


def cnt_different_lang(prompt: str, resp: str, skip_prompt_char: int = 10):
    num_prompt_en = cnt_english(prompt[skip_prompt_char:])
    num_prompt_zh = cnt_chinese(prompt[skip_prompt_char:])

    num_resp_en = cnt_english(resp)
    num_resp_zh = cnt_chinese(resp)

    if num_prompt_en > num_prompt_zh:
        return num_resp_en
    else:
        return num_resp_zh


def main(
    data_path: str,
    output_path: str,
):
    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    tot_cnt = 0
    for d in data:
        d['wrong_lang_cnt'] = cnt_different_lang(d['prompt'], d['resp'])
        tot_cnt += d['wrong_lang_cnt'] >= 1

    with open(output_path, 'w') as f:
        json.dump({'overview': {'tot_wrong_lang_cnt': tot_cnt},
                   'result': data
                   },
                  f,
                  ensure_ascii=False,
                  indent=2
                  )


if __name__ == '__main__':
    from fire import Fire

    Fire(main)
