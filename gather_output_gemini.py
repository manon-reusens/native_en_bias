import google.generativeai as genai
import pandas as pd
import argparse
import os
import numpy as np
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold,generation_types

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
                    default='gemini_flash',
                    choices=['gemini pro', 'gemini_flash'],
                    help="add the model you want to use"
)

parser.add_argument(
                    "--output_file",
                    action="store",
                    type=str,
                    help="Give the full path of where you want to save the output file",
)


def gather_answers(index,df,model='gemini-1.5-flash'):
    
    temperature=temp_dict[df.loc[index]['dataset_id']]
    model = genai.GenerativeModel(model,system_instruction="You are a helpful assistant.",generation_config=genai.types.GenerationConfig(
                                # Only one candidate for now.
                                candidate_count=1,
                                temperature=temperature,
                                max_output_tokens=4096),
                                safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    })
    if df.loc[index]['dataset_id']==3:
        task_def=df.loc[index]['task_def']+ ' Only respond with the rating.'
    elif df.loc[index]['dataset_id']==1:
        task_def=df.loc[index]['task_def']+  ' Only respond with the predicted last sentence.'
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

    chat=model.start_chat(history=[
        {
            'role': 'user',
            'parts': [task_def]
        },
        {
            'role': 'model',
            'parts': ['Understood'],
        },
    ]) 
    response=chat.send_message(df.loc[index]['final_prompt_en'])
    return response

if __name__ == "__main__":
    args = parser.parse_args()
    df=pd.read_parquet(args.input_file)
    df_final=df.loc[df['validated']==1]

    df_final['final_prompt_en']=df_final.apply(lambda row: row['instruction'].replace('<markprompt>[Your Prompt]</markprompt>.', row['prompt_en']).replace('<markprompt>[Your Prompt]</markprompt>?', row['prompt_en']), axis=1)
    df_final['final_prompt_en']=df_final.apply(lambda row: row['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>', row['prompt_en']), axis=1)

    genai.configure(api_key=args.key)
    #df_final[args.model+' replies']=None
    #df_final[args.model+' logprobs']=None

    temp_dict={0:0,1:0.7,2:0,3:0,4:0,5:0,6:0.7,7:0.7,8:0.7,9:0}

    if args.model=='gemini_flash':
        model='gemini-1.5-flash'
    elif args.model=='gemini pro':
        model='gemini-1.5-pro'
    
    results_full=[]
    cleaned_reasults=[]
    for i in df_final.index:
        if i%200==0:
            df_final.to_parquet(args.output_file)
        if df_final.loc[i][args.model+' replies']==None:
            try:
                result=gather_answers(i,df_final, model=model)
                df_final.at[i,args.model+' replies']=str(result.text)
                # df_final.at[i,args.model+' logprobs']=str(result.choices[0].logprobs.content)
                results_full.append(result)
                cleaned_reasults.append(result.text)
                time.sleep(8)
            except generation_types.StopCandidateException:
                print(df_final.loc[i]['prompt_en'])
                print('we are going to sleep for a while')
                time.sleep(5)

    df_final.to_parquet(args.output_file)





    
