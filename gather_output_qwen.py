import pandas as pd
import argparse
import numpy as np
import os
os.environ['TRANSFORMERS_CACHE']='/scratch/leuven/344/vsc34470/cache/'
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2Model, Qwen2Config


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
                    choices=['Qwen1.5-7B-Chat','Qwen1.5-0.5B-Chat','Qwen1.5-32B-Chat','Qwen1.5-110B-Chat','meta-llama/Llama-2-7b-chat-hf'],
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
            elif args.get_gold_label=='add_prompt_then_true':
                result_prompt,result=gather_answers(i,df_final, model=model)
                df_final.at[i,col_annotation]=str(result_prompt)
            result=gather_answers(i,df_final, model=model)
            df_final.at[i,col_replies]=str(result)
        # df_final.at[i,args.model+' logprobs']=str(result.choices[0].logprobs.content)
            results_full.append(result)
            cleaned_reasults.append(result)

    df_final.to_parquet(args.output_file)






