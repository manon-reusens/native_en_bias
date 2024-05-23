import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
import pandas as pd
import argparse
import os
import numpy as np
import time

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
                    default='haiku',
                    choices=['haiku', 'sonnet','opus'],
                    help="add the score you want to calculate"
)

parser.add_argument(
                    "--output_file",
                    action="store",
                    type=str,
                    help="Give the full path of where you want to save the output file",
)


def gather_answers(index,df,model='gpt-3.5-turbo'):
    temperature=temp_dict[df.loc[index]['dataset_id']]
    response = client.messages.create(
        model=model,
        system="You are a helpful assistant.",
        messages=[
        {"role": "user", "content": df.loc[index]['task_def']},
        {"role": "user", "content": df.loc[index]['final_prompt_en']}
        ],
        temperature=temperature,
        max_tokens=4096
    )
    return response

if __name__ == "__main__":
    args = parser.parse_args()
    df=pd.read_parquet(args.input_file)
    df_final=df.loc[df['validated']==1]

    df_final['final_prompt_en']=df_final.apply(lambda row: row['instruction'].replace('<markprompt>[Your Prompt]</markprompt>.', row['prompt_en']).replace('<markprompt>[Your Prompt]</markprompt>?', row['prompt_en']), axis=1)
    df_final['final_prompt_en']=df_final.apply(lambda row: row['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>', row['prompt_en']), axis=1)

    client = anthropic.Client(api_key=args.key)
    df_final[args.model+' replies']=np.NaN
    df_final[args.model+' logprobs']=np.NaN

    temp_dict={0:0,1:0.7,2:0,3:0,4:0,5:0,6:0.7,7:0.7,8:0.7,9:0}

    if args.model=='haiku':
        model='claude-3-haiku-20240307'
    elif args.model=='sonnet':
        model='claude-3-sonnet-20240229'
    elif args.model=='opus':
        model='claude-3-opus-20240229'
    
    results_full=[]
    cleaned_reasults=[]
    for i in df_final.index:
        if i%200==0:
            df_final.to_parquet(args.output_file)
        try:
            result=gather_answers(i,df_final, model=model)
            df_final.at[i,args.model+' replies']=result.content[0].text
            # df_final.at[i,args.model+' logprobs']=str(result.choices[0].logprobs.content)
            results_full.append(result)
            cleaned_reasults.append(result.content[0].text)
        except:
            print('we are going to sleep for a while')
            time.sleep(5)

    df_final.to_parquet(args.output_file)





    