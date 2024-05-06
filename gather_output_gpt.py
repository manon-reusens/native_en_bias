import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Gather output from chatbots')
parser.add_argument(
                    "--input_file",
                    action="store",
                    type=str,
                    help="Give the full path to the input file",
)
parser.add_argument(
                    '--key',
                    action='store',
                    type=str,
                    help='give the API key')

parser.add_argument(
                    "--model",
                    action="store",
                    type=str,
                    default='gpt3.5',
                    choices=['gpt3.5', 'gpt4'],
                    help="add the score you want to calculate"
)

parser.add_argument(
                    "--set",
                    action="store",
                    type=str,
                    default='all',
                    choices=['all', '10 & 30'],
                    help="add the score you want to calculate"
)

parser.add_argument(
                    "--output_file",
                    action="store",
                    type=str,
                    help="Give the full path of where you want to save the output file",
)

import openai
import pandas as pd
from openai import OpenAI
import os
import numpy as np
import time

def gather_answers(index,df,model='gpt-3.5-turbo'):
    temperature=temp_dict[df.loc[index]['dataset_id']]
    response = client.chat.completions.create(
        model=model,
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": df.loc[index]['task_def']},
        {"role": "user", "content": df.loc[index]['final_prompt_en']}
        ],
        temperature=temperature,
        logprobs=True,
        top_logprobs=5,
        seed=42
    )
    return response

if __name__ == "__main__":
    args = parser.parse_args()
    df=pd.read_parquet(args.input_file)
    df_final=df.loc[df['validated']==1]

    df_final['final_prompt_en']=df_final.apply(lambda row: row['instruction'].replace('<markprompt>[Your Prompt]</markprompt>.', row['prompt_en']).replace('<markprompt>[Your Prompt]</markprompt>?', row['prompt_en']), axis=1)
    df_final['final_prompt_en']=df_final.apply(lambda row: row['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>', row['prompt_en']), axis=1)

    if args.set=='10 & 30':
        df_final_set=df_final.loc[(df_final['set_id']==10) | (df_final['set_id']==30)]
    else:
        df_final_set=df_final

    os.environ['OPENAI_API_KEY']=args.key

    client = OpenAI()
    df_final_set[args.model+' replies']=np.NaN
    df_final_set[args.model+' logprobs']=np.NaN

    temp_dict={0:0,1:0.7,2:0,3:0,4:0,5:0,6:0.7,7:0.7,8:0.7,9:0}

    if args.model=='gpt3.5':
        model='gpt-3.5-turbo-0125'
    if args.model=='gpt4':
        model='gpt-4-0125-preview'
    
    results_full=[]
    cleaned_reasults_gpt3=[]
    for i in df_final_set.index:
        if i%200==0:
            df_final_set.to_parquet(args.output_file)
        try:
            result=gather_answers(i,df_final_set, model=model)
            df_final_set.at[i,args.model+' replies']=result.choices[0].message.content
            df_final_set.at[i,args.model+' logprobs']=str(result.choices[0].logprobs.content)
            results_full.append(result)
            cleaned_reasults_gpt3.append(result.choices[0].message.content)
        except:
            print('we are going to sleep for a while')
            time.sleep(5)

    df_final_set.to_parquet(args.output_file)





    