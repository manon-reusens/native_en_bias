import anthropic
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
                    help="add the model you want to use"
)

parser.add_argument(
                    "--output_file",
                    action="store",
                    type=str,
                    help="Give the full path of where you want to save the output file",
)
parser.add_argument(
                    "--mode",
                    action="store",
                    type=str,
                    default='standard',
                    choices=['standard','add_all_native','add_all_non_native','guess_native','reformulate'],
                    help="Give the full path of where you want to save the output file",
)
parser.add_argument(
                    "--get_gold_label",
                    action="store",
                    type=str,
                    default='False',
                    choices=["True","False","add_prompt_then_true"],
                    help="True if you get the gold label for the annotations using full prompt, if not gold label Then False, otherwise add_prompt_then_true",

)


def gather_answers(index,df,model='gpt-3.5-turbo'):
    temperature=temp_dict[df.loc[index]['dataset_id']]
    if df.loc[index]['dataset_id']==3:
        task_def=df.loc[index]['task_def']+ ' Only respond with the rating.'
    elif df.loc[index]['dataset_id']==1:
        task_def=df.loc[index]['task_def']+  ' Only respond with the predicted last sentence.'
    elif (df.loc[index]['dataset_id']==5 ) or (df.loc[index]['dataset_id']==4):
         task_def=df.loc[index]['task_def']+  ' Only respond with "yes" or "no".'
    elif df.loc[index]['dataset_id']==6:
        task_def=df.loc[index]['task_def']+ ' Only respond with the news article.'
    elif df.loc[index]['dataset_id']==7:
        task_def=df.loc[index]['task_def']+ '  Only respond with the paragraph.'
    elif df.loc[index]['dataset_id']==8:
        task_def=df.loc[index]['task_def']+ ' Only respond with the paraphrased sentence.'
    elif df.loc[index]['dataset_id']==9:
        task_def=df.loc[index]['task_def']+ ' Only respond with the letter indicating the most corresponding reason.'
    else:
        task_def=df.loc[index]['task_def']

    if args.mode=='add_all_native':
        system_prompt="You are a helpful assistant. Respond as if you are interacting with a native English speaker"
    elif args.mode=='add_all_non_native':
        system_prompt="You are a helpful assistant. Respond as if you are interacting with a non-native English speaker"
    else:
        system_prompt="You are a helpful assistant."

    if args.mode=='guess_native':
        response1 = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
            {"role": "user", 'content':'Guess whether the writer of the following prompt is a native or non-native English speaker: '+df.loc[index]['prompt_en']}
            ],
            temperature=0,
            max_tokens=4096
            )
        response2 = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
            {"role": "user", 'content':'Guess whether the writer of the following prompt is a native or non-native English speaker: '+df.loc[index]['prompt_en']},
            {"role":"assistant","content":response1.content[0].text},
            {"role": "user", "content": 'Next, execute  the following task taking this information into account. '+task_def},
            {"role":"assistant","content":'Understood'},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
            max_tokens=4096
            )
        return response1,response2
    else:
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
            {"role": "user", "content": task_def},
            {"role":"assistant","content":'Understood'},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
            max_tokens=4096
        )
        return response

if __name__ == "__main__":
    args = parser.parse_args()
    df=pd.read_parquet(args.input_file)
    if args.get_gold_label=='False':
        df_final=df.loc[df['validated']==1]

    if args.get_gold_label=='False':
        df_final['final_prompt_en']=df_final.apply(lambda row: row['instruction'].replace('<markprompt>[Your Prompt]</markprompt>.', row['prompt_en']).replace('<markprompt>[Your Prompt]</markprompt>?', row['prompt_en']), axis=1)
        df_final['final_prompt_en']=df_final.apply(lambda row: row['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>', row['prompt_en']), axis=1)
    elif args.get_gold_label=='True':
        df_final['final_prompt_en']=df_final['instruction']


    client = anthropic.Client(api_key=args.key)
    if args.mode=='add_all_native':
        col_replies=args.model+' replies_all_native'
        col_logprobs=args.model+' logprobs_all_native'
    elif args.mode=='add_all_non_native':
        col_replies=args.model+' replies_all_non_native'
        col_logprobs=args.model+' logprobs_all_non_native'
    elif args.mode=='standard':
        col_replies=args.model+' replies'
        col_logprobs=args.model+' logprobs'
    elif args.mode=='guess_native':
        col_replies=args.model+' replies_guess_native'
        col_logprobs=args.mode+' logprobs_guess_native'
        col_guess=args.model+' guessed_native'
        col_guess_logprobs=args.model+' guessed_native_logprobs'
    elif args.mode=='reformulate':
        col_replies=args.model+' replies_reformulate'
        col_logprobs=args.mode+' logprobs_reformulate'

    df_final[col_replies]=None
    df_final[col_logprobs]=None
    if args.mode=='guess_native':
        df_final[col_guess]=None
        df_final[col_guess_logprobs]=None

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
            if args.mode=='guess_native':
                result_guess,result=gather_answers(i,df_final, model=model)
                df_final.at[i,col_guess]=result_guess.content[0].text
                    # df_final.at[i,col_guess_logprobs]=str(result_guess.choices[0].logprobs.content)
            else:
                result=gather_answers(i,df_final, model=model)
            df_final.at[i,col_replies]=result.content[0].text
            # df_final.at[i,args.model+' logprobs']=str(result.choices[0].logprobs.content)
            results_full.append(result)
            cleaned_reasults.append(result.content[0].text)
        except:
            print('we are going to sleep for a while')
            time.sleep(5)

    df_final.to_parquet(args.output_file)





    