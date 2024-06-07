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
                    choices=['gpt3.5', 'gpt4','gpt4_o'],
                    help="add the score you want to calculate"
)

parser.add_argument(
                    "--set",
                    action="store",
                    type=str,
                    default='all',
                    choices=['all', '8'],
                    help="add the score you want to calculate"
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
                    choices=['standard','add_all_native','add_all_non_native','guess_native','add_history'],
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
parser.add_argument(
                    "--trial",
                    action="store",
                    type=str,
                    default='False',
                    choices=["True","False"],
                    help="True if you want to add the task definition to the system prompt.",

)

import openai
import pandas as pd
from openai import OpenAI
import os
import numpy as np
import time

def gather_answers(index,df,model='gpt-3.5-turbo'):
    temperature=temp_dict[df.loc[index]['dataset_id']]
    #change the task definitions
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

    #change task definition
    # if args.mode=='guess_native':
    #     task_def='First, guess whether the prompt is made by a native or a non-native speaker. Then, taking this information into account execute the following task. '+task_def
    
    #change the system prompt
    if args.mode=='add_all_native':
        system_prompt="You are a helpful assistant. Respond as if you are interacting with a native English speaker"
    elif args.mode=='add_all_non_native':
        system_prompt="You are a helpful assistant. Respond as if you are interacting with a non-native English speaker"
    else:
        system_prompt="You are a helpful assistant."

    if args.get_gold_label=='add_prompt_then_true':
        system_prompt1="You are a helpful assistant. You will get three main parts: a task definition, desired output, and a prompt_to_annotate. Your task is to come up with prompts that should be filled in at the placeholder <markprompt>[Your Prompt]</markprompt> in the prompt_to_annotate, such that the desired output would be outputted by a model when given the task deifnition and full prompt. Only respond with the replacement of the placeholder."
        system_prompt2="You are a helpful assistant."

    if args.trial=='True':
        system_prompt=system_prompt+' '+ task_def
    

    if args.mode=='guess_native':
        response1=client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": 'Guess whether the writer of the following prompt is a native or non-native English speaker: '+df.loc[index]['prompt_en']}],
            temperature=0,
            logprobs=True,
            top_logprobs=5,
            seed=42
            )
        response2=client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": 'Guess whether the writer of the following prompt is a native or non-native English speaker: '+df.loc[index]['prompt_en']},
            {"role":'assistant',"content":response1.choices[0].message.content},
            {"role": "user", "content": 'Next, execute  the following task taking this information into account. '+task_def},
            {"role":'assistant',"content":'Understood'},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
            seed=42
        )
        return response1,response2
    elif args.get_gold_label=='add_prompt_then_true':
        print('we are in the correcr if statement')
        prompt=client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": system_prompt1},
            {"role": "user", "content": 'task definition: '+task_def+'instruction: '+df.loc[index]['final_prompt_en'].replace('</markprompt>','').replace('<markprompt>','')+' the desired output: '+df.loc[index]['req_output']}
            ],
            temperature=0,
            seed=42
        )
        text_prompt=prompt.choices[0].message.content
        print(text_prompt)
        full_prompt=df.loc[index]['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>.', text_prompt).replace('<markprompt>[Your Prompt]</markprompt>?', text_prompt).replace('<markprompt>[Your Prompt]</markprompt>',text_prompt)
        response=client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": system_prompt2},
            {"role": "user", "content": task_def},
            {"role":'assistant',"content":'Understood'},
            {"role": "user", "content": full_prompt}
            ],
            temperature=temperature,
            seed=42
        )
        return prompt,response
    elif args.trial=='True':
        response = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
            seed=42
        )
        return response
    elif args.mode=='add_history':
        history=df.loc[(df.index!=index) & (df['user_id']==df.loc[index]['user_id'])].sample(n=5)['prompt_en']
        response = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role":"user","content": 'Here is some extra text written by the same person'+history},
            {"role":'assistant',"content":'Ok.'},
            {"role": "user", "content": task_def},
            {"role":'assistant',"content":'Understood'},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
            seed=42
        )
        return history, response
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_def},
            {"role":'assistant',"content":'Understood'},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
            logprobs=True,
            top_logprobs=5,
            seed=42
        )
        return response

if __name__ == "__main__":
    print('we start the experiment')
    args = parser.parse_args()
    df=pd.read_parquet(args.input_file)
    if args.get_gold_label=='False':
        df_final=df.loc[df['validated']==1]
    else:
        df_final=df

    if args.get_gold_label=='False':
        df_final['final_prompt_en']=df_final.apply(lambda row: row['instruction'].replace('<markprompt>[Your Prompt]</markprompt>.', row['prompt_en']).replace('<markprompt>[Your Prompt]</markprompt>?', row['prompt_en']), axis=1)
        df_final['final_prompt_en']=df_final.apply(lambda row: row['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>', row['prompt_en']), axis=1)
    elif args.get_gold_label=='True':
        df_final['final_prompt_en']=df_final['instruction']
    elif args.get_gold_label=='add_prompt_then_true':
        df_final['final_prompt_en']=df_final['prompt_instruction']
        print(df_final)

    if args.set=='8':
        df_final_set=df_final.loc[(df_final['dataset_id']==8) ]#| (df_final['set_id']==30)]
    else:
        df_final_set=df_final

    os.environ['OPENAI_API_KEY']=args.key

    client = OpenAI()
    print('now we define the columns')
    if args.mode=='add_all_native':
        col_replies=args.model+' replies_all_native'
        col_logprobs=args.model+' logprobs_all_native'
    elif args.mode=='add_all_non_native':
        col_replies=args.model+' replies_all_non_native'
        col_logprobs=args.model+' logprobs_all_non_native'
    elif args.mode=='standard':
        col_replies=args.model+' replies'
        col_logprobs=args.model+' logprobs'
    elif args.mode=='add_history':
        col_replies=args.model+' replies_history'
        col_logprobs=args.model+' logprobs_history'
        col_history='added_history'
    elif args.mode=='guess_native':
        col_replies=args.model+' replies_guess_native'
        col_logprobs=args.mode+' logprobs_guess_native'
        col_guess=args.model+' guessed_native'
        col_guess_logprobs=args.model+' guessed_native_logprobs'
    elif args.mode=='reformulate':
        col_replies=args.model+' replies_reformulate'
        col_logprobs=args.mode+' logprobs_reformulate'

    if col_replies not in df_final_set.columns:
        df_final_set[col_replies]=None
        df_final_set[col_logprobs]=None
    if args.mode=='guess_native':
        if col_guess not in df_final_set.columns:
            df_final_set[col_guess]=None
            df_final_set[col_guess_logprobs]=None
    if args.get_gold_label=='add_prompt_then_true':
        print('we still need to define the annotation col')
        col_annotation=args.model+' prompt'
        if col_annotation not in df_final_set.columns:
            df_final_set[col_annotation]=None

    temp_dict={0:0,1:0.7,2:0,3:0,4:0,5:0,6:0.7,7:0.7,8:0.7,9:0}

    if args.model=='gpt3.5':
        model='gpt-3.5-turbo-0125'
    if args.model=='gpt4':
        model='gpt-4-0125-preview'
    if args.model=='gpt4_o':
        model='gpt-4o-2024-05-13'
    
    results_full=[]
    cleaned_reasults_gpt3=[]
    for i in df_final_set.index:
        if i%200==0:
            df_final_set.to_parquet(args.output_file)
        if df_final_set.loc[i][col_replies]==None:
            try:
                if args.mode=='guess_native':
                    result_guess,result=gather_answers(i,df_final_set, model=model)
                    df_final_set.at[i,col_guess]=result_guess.choices[0].message.content
                    df_final_set.at[i,col_guess_logprobs]=str(result_guess.choices[0].logprobs.content)
                elif args.mode=='add_history':
                    history,result=gather_answers(i,df_final_set, model=model)
                    df_final_set.at[i,col_history]=history
                elif args.get_gold_label=='add_prompt_then_true':
                    print('we will now gather the annotation')
                    result_prompt,result=gather_answers(i,df_final_set, model=model)
                    df_final_set.at[i,col_annotation]=result_prompt.choices[0].message.content
                else:
                    print('not in the correct loop to get the answer')
                    result=gather_answers(i,df_final_set, model=model)
                df_final_set.at[i,col_replies]=result.choices[0].message.content
                df_final_set.at[i,col_logprobs]=str(result.choices[0].logprobs.content)
                results_full.append(result)
                cleaned_reasults_gpt3.append(result.choices[0].message.content)
            except:
                print('we are going to sleep for a while')
                time.sleep(5)

    df_final_set.to_parquet(args.output_file)





    
