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
                    default='mixtral-8x7b-32768',
                    choices=['mixtral-8x7b-32768','llama3-8b-8192','llama3-70b-8192','llama2-70b-4096','gemma-7b-it'],
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
from groq import Groq
import os
import numpy as np
import time

def gather_answers(index,df,model='mixtral-8x7b-32768'):
    temperature=temp_dict[df.loc[index]['dataset_id']]
    if df.loc[index]['dataset_id']==3:
        response = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": df.loc[index]['task_def']+ ' Only respond by giving the rating, do not provide other information.'},
            {"role":'assistant',"content":'Understood'},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": df.loc[index]['task_def']},
            {"role":'assistant',"content":'Understood'},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
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

    os.environ["GROQ_API_KEY"]=args.key

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)
    df_final_set[args.model+' replies']=np.NaN
    # df_final_set[args.model+' logprobs']=np.NaN
 
    temp_dict={0:0,1:0.7,2:0,3:0,4:0,5:0,6:0.7,7:0.7,8:0.7,9:0}
    
    results_full=[]
    cleaned_results=[]
    for i in df_final_set.index:
        if i%200==0:
            df_final_set.to_parquet(args.output_file)
        try:
            result=gather_answers(i,df_final_set, model=args.model)
            df_final_set.at[i,args.model+' replies']=result.choices[0].message.content
            # df_final_set.at[i,args.model+' logprobs']=str(result.choices[0].logprobs.content)
            results_full.append(result)
            cleaned_results.append(result.choices[0].message.content)
            time.sleep(2)
        except:
            print('we are going to sleep for a while')
            time.sleep(5)

    df_final_set.to_parquet(args.output_file)





    