import pandas as pd
import argparse
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] ='/scratch/leuven/344/vsc34470'
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2Model, Qwen2Config
os.environ['TRANSFORMERS_CACHE'] ='/scratch/leuven/344/vsc34470'

parser = argparse.ArgumentParser(description='Gather output from chatbots')
parser.add_argument(
                    "--input_file",
                    action="store",
                    type=str,
                    help="Give the full path to the input file",
)
parser.add_argument(
                    "--model",
                    action="store",
                    type=str,
                    default='Qwen1.5-0.5B-Chat',
                    choices=['Qwen1.5-7B','Qwen1.5-7B-Chat','Qwen1.5-0.5B-Chat','Qwen1.5-32B-Chat','Qwen1.5-110B-Chat','meta-llama/Llama-2-7b-chat-hf'],
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
                    "--extra",
                    action="store",
                    type=str,
                    default='user',
                    choices=["ueser","native","non_native"],
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
    temperature=temp_dict[df.loc[index]['dataset_id']]
    prompt = df.loc[index]['final_prompt_en']
    if df.loc[index]['dataset_id']==3:
        task_def=df.loc[index]['task_def']+ ' Only respond with the rating.'
    elif df.loc[index]['dataset_id']==1:
        task_def=df.loc[index]['task_def']+  ' Only respond with the predicted last sentence.'
    elif (df.loc[index]['dataset_id']==5 ) or (df.loc[index]['dataset_id']==4) :
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

    if args.mode=='add_all_native':
        system_prompt="You are a helpful assistant. Respond as if you are interacting with a native English speaker"
    elif args.mode=='add_all_non_native':
        system_prompt="You are a helpful assistant. Respond as if you are interacting with a non-native English speaker"
    else:
        system_prompt="You are a helpful assistant."

    if args.get_gold_label=='add_prompt_then_true':
        system_prompt1="You are a helpful assistant. You will get three main parts: a task definition, desired output, and a prompt_to_annotate. Your task is to come up with prompts that should be filled in at the placeholder <markprompt>[Your Prompt]</markprompt> in the prompt_to_annotate, such that the desired output would be outputted by a model when given the task deifnition and full prompt. Only respond with the replacement of the placeholder."
        system_prompt2="You are a helpful assistant."

    if args.mode=='guess_native':
        messages=[{"role":"system","content":system_prompt},
                  {"role": "user", "content": 'Guess whether the writer of the following prompt is a native or non-native English speaker: '+df.loc[index]['prompt_en']}]
    elif args.get_gold_label=='add_prompt_then_true':
        messages=[{"role":"system","content":system_prompt1},
                  {"role": "user", "content": 'task definition: '+task_def+'instruction: '+df.loc[index]['final_prompt_en'].replace('</markprompt>','').replace('<markprompt>','')+' the desired output: '+df.loc[index]['req_output']}]
    elif args.mode=='add_history':
        if args.extra=='non_native':
            history=', '.join(list(df.loc[(df.index!=index) & (df['native_or_not']=='non_native')].sample(n=5)['prompt_en'].values))#(df['user_id']==df.loc[index]['user_id'])
        elif args.extra=='native':
            history=', '.join(list(df.loc[(df.index!=index) & (df['native_or_not']=='native')].sample(n=5)['prompt_en'].values))
        elif args.extra=='user':
            history=', '.join(list(df.loc[(df.index!=index) & (df['user_id']==df.loc[index]['user_id'])].sample(n=5)['prompt_en'].values))
        messages=[{"role":"system","content":system_prompt},
                  {"role": "user", "content":'Here is some extra text written by the same person'+history}, {"role":"assistant","content":'Ok.'},
                    {"role": "user", "content": task_def}, {"role":"assistant","content":'Understood'},
                    {"role":"user", "content":prompt}]
    
    else:
        messages = [{"role":"system","content":system_prompt},
                    {"role": "user", "content": task_def}, {"role":"assistant","content":'Understood'},
                    {"role":"user", "content":prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    #to do: 'guess_native'
    if args.mode=='guess_native':
        model.generation_config.top_p=None
        model.generation_config.temperature=None
        model.generation_config.top_k=None
        #gather first response
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=4096, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        messages=[{"role":"system","content":system_prompt},
                  {"role": "user", "content": 'Guess whether the writer of the following prompt is a native or non-native English speaker: '+df.loc[index]['prompt_en']},
                  {"role":"assistant","content":str(response1)},
                  {"role": "user", "content": 'Next, execute  the following task taking this information into account. '+task_def},
                  {"role":'assistant',"content":'Understood'},
                  {"role": "user", "content": df.loc[index]['final_prompt_en']}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
    elif args.get_gold_label=='add_prompt_then_true':
        model.generation_config.top_p=None
        model.generation_config.temperature=None
        model.generation_config.top_k=None
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=4096, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        text_prompt=response1
        full_prompt=df.loc[index]['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>.', text_prompt).replace('<markprompt>[Your Prompt]</markprompt>?', text_prompt).replace('<markprompt>[Your Prompt]</markprompt>',text_prompt)
        
        messages=[{"role":"system","content":system_prompt2},
                  {"role": "user", "content": task_def},
                  {"role":'assistant',"content":'Understood'},
                  {"role": "user", "content": full_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

    if temperature==0:
        model.generation_config.top_p=None
        model.generation_config.temperature=None
        model.generation_config.top_k=None
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=4096, do_sample=False)
    else:
        model.generation_config.top_p=0.8
        model.generation_config.top_k=20
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=4096, do_sample=True,temperature=temperature)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if args.mode=='guess_native':
        return response1, response
    elif args.get_gold_label=='add_prompt_then_true':
        return response1, response
    elif args.mode=='add_history':
        return history, response
    else:
        return response

if __name__ == "__main__":
    args = parser.parse_args()
    device = "cuda"
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


    temp_dict={0:0,1:0.7,2:0,3:0,4:0,5:0,6:0.7,7:0.7,8:0.7,9:0}
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
        if col_replies not in df_final.columns:
            df_final[col_guess]=None
            df_final[col_guess_logprobs]=None
    if args.mode=='add_history':
        if col_history not in df_final.columns:
            df_final[col_history]=None

    if args.get_gold_label=='add_prompt_then_true':
        col_annotation=args.model+' prompt'
        if col_annotation not in df_final.columns:
            df_final[col_annotation]=None

    if 'Qwen' in args.model:
        model = AutoModelForCausalLM.from_pretrained("Qwen/"+args.model, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/"+args.model)
    elif 'Llama' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    results_full=[]
    cleaned_reasults=[]
    for i in df_final.index:
        if i%200==0:
            df_final.to_parquet(args.output_file)
        if df_final.loc[i][col_replies]==None:
            if args.mode=='guess_native':
                guess, result=gather_answers(i,df_final, model=model)
                df_final.at[i,col_guess]=str(guess)
            if args.mode=='add_history':
                history,result=gather_answers(i,df_final, model=model)
                df_final.at[i,col_history]=history
            elif args.get_gold_label=='add_prompt_then_true':
                result_prompt,result=gather_answers(i,df_final, model=model)
                df_final.at[i,col_annotation]=str(result_prompt)
            result=gather_answers(i,df_final, model=model)
            df_final.at[i,col_replies]=str(result)
        # df_final.at[i,args.model+' logprobs']=str(result.choices[0].logprobs.content)
            results_full.append(result)
            cleaned_reasults.append(result)

    df_final.to_parquet(args.output_file)






