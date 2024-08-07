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
                    "--extra",
                    action="store",
                    type=str,
                    default='user',
                    choices=["user","native","non_native"],
                    help="True if you get the gold label for the annotations using full prompt, if not gold label Then False, otherwise add_prompt_then_true",

)
def make_groups(df):
    df['native_or_not']=df.apply(lambda row: 'native' if ('en' in row['nat_lang']) else 'non_native' , axis=1)
    df['strict_native_or_not']=df.apply(lambda row: 'strict native' if (row['nat_lang']=='{en}') else 'not strict native' , axis=1)
    df['western_native_or_not']=df.apply(lambda row: 'western native' if (row['user_id'] in [159,99,104,110,193,127,457,459,481,114,445,541,542,129,562,563,338,355,254,672,70,700,709,673,687,701,17,255,588,71,356]) else 'not western native' , axis=1)
    df['african_or_not']=df.apply(lambda row: 'african' if (row['user_id'] in [162,14,375,443,536,670,458,540]) else 'not african' , axis=1)
    df['student_or_not']=df.apply(lambda row: 'student' if (row['user_id'] in [7,190,67,191,9,192,10,11,12,68,69,312,313,13,37,581,580,582,251,640,352,353,583,600,601,660,354,602,620,621,14,670,671,38,622,15,193,685,585,703,686,16,704,586,708,39,697,695,698,699,645,587,253,355,314,254,672,70,700,709,673,687,701,17,255,588,71,356]) else 'not student' , axis=1)
    return df


def gather_answers(index,df,model='gpt-3.5-turbo'):
    print('we get in the function')
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
    df=make_groups(df)
    print('we have defined the task def')

    if args.mode=='add_all_native':
        system_prompt="You are a helpful assistant. Respond as if you are interacting with a native English speaker"
    elif args.mode=='add_all_non_native':
        system_prompt="You are a helpful assistant. Respond as if you are interacting with a non-native English speaker"
    else:
        system_prompt="You are a helpful assistant."

    if args.get_gold_label=='add_prompt_then_true':
        system_prompt1="You are a helpful assistant. You will get three main parts: a task definition, desired output, and a prompt_to_annotate. Your task is to come up with prompts that should be filled in at the placeholder <markprompt>[Your Prompt]</markprompt> in the prompt_to_annotate, such that the desired output would be outputted by a model when given the task deifnition and full prompt. Only respond with the replacement of the placeholder."
        system_prompt2="You are a helpful assistant."
        print('this is ok')
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
    elif args.mode=='add_history':
        if args.extra=='non_native':
            history=', '.join(list(df.loc[(df.index!=index) & (df['native_or_not']=='non_native')].sample(n=5)['prompt_en'].values))#(df['user_id']==df.loc[index]['user_id'])
        elif args.extra=='native':
            history=', '.join(list(df.loc[(df.index!=index) & (df['native_or_not']=='native')].sample(n=5)['prompt_en'].values))
        elif args.extra=='user':
            history=', '.join(list(df.loc[(df.index!=index) & (df['user_id']==df.loc[index]['user_id'])].sample(n=5)['prompt_en'].values))
        print(history)
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[
            {"role":"user","content": 'Here is some extra text written by the same person'+history},
            {"role":'assistant',"content":'Ok.'},
            {"role": "user", "content": task_def},
            {"role":'assistant',"content":'Understood'},
            {"role": "user", "content": df.loc[index]['final_prompt_en']}
            ],
            temperature=temperature,
            max_tokens=4096
        )
        return history, response
    elif args.get_gold_label=='add_prompt_then_true':
        prompt=client.messages.create(
            model=model,
            system=system_prompt1,
            messages=[
            {"role": "user", "content": 'task definition: '+task_def+'instruction: '+df.loc[index]['final_prompt_en'].replace('</markprompt>','').replace('<markprompt>','')+' the desired output: '+df.loc[index]['req_output']}
            ],
            temperature=0,
            max_tokens=4096)
        print(prompt.content[0].text)
        text_prompt=prompt.content[0].text
        full_prompt=df.loc[index]['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>.', text_prompt).replace('<markprompt>[Your Prompt]</markprompt>?', text_prompt).replace('<markprompt>[Your Prompt]</markprompt>',text_prompt)
        print(full_prompt)
        response=client.messages.create(
            model=model,
            system=system_prompt2,
            messages=[
            {"role": "user", "content": task_def},
            {"role":'assistant',"content":'Understood'},
            {"role": "user", "content": full_prompt}
            ],
            temperature=temperature,
            max_tokens=4096
        )
        print(response)
        return prompt,response
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
    print(args.get_gold_label)
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

    print('this is still okay')

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
    if col_replies not in df_final.columns:
        df_final[col_replies]=None
        df_final[col_logprobs]=None
        if args.mode=='guess_native':
            df_final[col_guess]=None
            df_final[col_guess_logprobs]=None
        if args.get_gold_label=='add_prompt_then_true':
            col_annotation=args.model+' prompt'
            if col_annotation not in df_final.columns:
                df_final[col_annotation]=None
        if args.mode=='add_history':
            if col_history not in df_final.columns:
                df_final[col_history]=None

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
        if df_final.loc[i][col_replies]==None:
            try:
                if args.mode=='guess_native':
                    result_guess,result=gather_answers(i,df_final, model=model)
                    df_final.at[i,col_guess]=result_guess.content[0].text
                    # df_final.at[i,col_guess_logprobs]=str(result_guess.choices[0].logprobs.content)
                elif args.mode=='add_history':
                    print('we are gathering the answer')
                    history,result=gather_answers(i,df_final, model=model)
                    df_final.at[i,col_history]=history
                elif args.get_gold_label=='add_prompt_then_true':
                    result_prompt,result=gather_answers(i,df_final, model=model)
                    print(result_prompt.content[0].text)
                    df_final.at[i,args.model+' prompt']=str(result_prompt.content[0].text)
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





    
