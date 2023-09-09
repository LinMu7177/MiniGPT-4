import os
import json

import time

import random

import openai
import pandas as pd

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)


def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')

# GRES Data
GRES_Data = '/mnt/local/wwx/LLM_Data/refcoco/'
GRES_refs_path = GRES_Data + 'grefcoco/grefs(unc).json'
GRES_ann_path = GRES_Data + 'grefcoco/instances.json'

# GRES Refs
gres_refs = json.load(open(GRES_refs_path, 'rb'))
gres_refs_df = pd.DataFrame(gres_refs)
gres_refs_df = gres_refs_df[gres_refs_df['split'] == 'val'].reset_index(drop=True)


df_list = []
for i in range(3):
    down = 1000*i
    up = 1000*(i+1)
    globals()['gres_refs_df_'+str(down)+'_'+str(up)] = gres_refs_df[down:up]
    df_list.append(globals()['gres_refs_df_'+str(down)+'_'+str(up)])


def get_query_from_sent(phrase):
    response = chat_completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You're an assistant that helps me construct data and be able to construct appropriate questions for specific phrases. Pay attention to the questions you construct that need to be strictly around a given phrase, do not introduce people and objects casually, and especially do not include the word 'you' in your question."},
            {"role": "user",
             "content": "I will give you a phrase and ask you to construct a short question around it. The first phrase I'll give you is 'blue shirt'."},
            {"role": "assistant", "content": "Output: 'Who is wearing the blue shirt?'"},
            {"role": "user", "content": "The next phrase is:'woman sitting on the right'"},
            {"role": "assistant", "content": "Output: 'What the woman sitting on the right is holding?'"},
            {"role": "user", "content": "The next phrase is:'left vase'"},
            {"role": "assistant", "content": "Output: 'How many flowers are in the vase on the left?'"},
            {"role": "user", "content": "The next phrase is:'top left corner of pic'"},
            {"role": "assistant", "content": "Output: 'What is in the top left corner of the pic?'"},
            {"role": "user", "content": "The next phrase is:'" + phrase + "'"}
        ],
    )
    try:
        content = response["choices"][0]['message']['content']
        if len(content) > 0 and 'Output:' in content and 'you' not in content:
            res = content.split('Output:', 1)[1].replace("'", "").replace('"', '').strip()
        else:
            res = ''
    except:
        res = ''

    return res


counter = 0
transfer_count = 0


def sent_to_query(sentences):
    global counter
    global transfer_count
    global start_time
    counter += 1
    if counter % 200 == 0:
        print(time.time() / 60 - start_time)
        start_time = time.time() / 60
        print("=====" * 5)
        print(counter)
        print("=====" * 5)
    time.sleep(0.1)
    for sent in sentences:
        if random.random() >= 0.5:
            if len(sent['tokens']) >= 2:
                # print("+++++" * 10)
                # print(sent['sent'])
                # print("+++++" * 10)
                query = get_query_from_sent(sent['sent'])
                if len(query) > 0:
                    # print("=====" * 5)
                    # print(query)
                    # print("=====" * 5)
                    sent['sent'] = query
                    sent['tokens'] = query.split(" ")
                    sent['raw'] = query
                    transfer_count += 1
    return sentences

for i in range(len(df_list)):
    print("+++++"*10)
    print(i)
    print("+++++"*10)
    start_time = (time.time()/60)
    df_list[i]['sentences'] = df_list[i]['sentences'].map(sent_to_query)


tmp_df = df_list[0]

for i in range(1,len(df_list)):
     tmp_df = pd.concat([tmp_df,df_list[i]])

tmp_df.to_json('/home/tsingqguo/wwx/workspace/DataReader/grefs(unc)_val_turbo.json',orient='records')