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
                    choices=['Qwen1.5-7B-Chat','Qwen1.5-0.5B-Chat','Qwen1.5-32B-Chat','Qwen1.5-110B-Chat'],
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
    prompt = df.loc[index]['final_prompt_en']
    if df.loc[index]['dataset_id']==3:
            messages = [{"role":"system","content":"You are a helpful assistant."},
                {"role": "user", "content": df.loc[index]['task_def']+ ' Only respond with the rating.'}, {"role":"assistant","content":'Understood'},
                {"role":"user", "content":prompt}]
    elif df.loc[index]['dataset_id']==1:
            messages = [{"role":"system","content":"You are a helpful assistant."},
                {"role": "user", "content": df.loc[index]['task_def']+' Only respond with the predicted last sentence.'}, {"role":"assistant","content":'Understood'},
                {"role":"user", "content":prompt}]
    elif df.loc[index]['dataset_id']==6:
            messages = [{"role":"system","content":"You are a helpful assistant."},
                {"role": "user", "content": df.loc[index]['task_def']+' Only respond with the news article.'}, {"role":"assistant","content":'Understood'},
                {"role":"user", "content":prompt}]
    elif df.loc[index]['dataset_id']==7:
            messages = [{"role":"system","content":"You are a helpful assistant."},
                {"role": "user", "content": df.loc[index]['task_def']+' Only respond with the paragraph.'}, {"role":"assistant","content":'Understood'},
                {"role":"user", "content":prompt}]
    elif df.loc[index]['dataset_id']==8:
            messages = [{"role":"system","content":"You are a helpful assistant."},
                {"role": "user", "content": df.loc[index]['task_def']+' Only respond with the paraphrased sentence.'}, {"role":"assistant","content":'Understood'},
                {"role":"user", "content":prompt}]
    else:
        messages = [{"role":"system","content":"You are a helpful assistant."},
                    {"role": "user", "content": df.loc[index]['task_def']}, {"role":"assistant","content":'Understood'},
                    {"role":"user", "content":prompt}]
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

    return response

if __name__ == "__main__":
    args = parser.parse_args()
    device = "cuda"
    df=pd.read_parquet(args.input_file)
    df_final=df.loc[df['validated']==1]

    df_final['final_prompt_en']=df_final.apply(lambda row: row['instruction'].replace('<markprompt>[Your Prompt]</markprompt>.', row['prompt_en']).replace('<markprompt>[Your Prompt]</markprompt>?', row['prompt_en']), axis=1)
    df_final['final_prompt_en']=df_final.apply(lambda row: row['final_prompt_en'].replace('<markprompt>[Your Prompt]</markprompt>', row['prompt_en']), axis=1)

    temp_dict={0:0,1:0.7,2:0,3:0,4:0,5:0,6:0.7,7:0.7,8:0.7,9:0}
    df_final[args.model+' replies']=None
    df_final[args.model+' logprobs']=None
    
    model = AutoModelForCausalLM.from_pretrained("Qwen/"+args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/"+args.model)
    
    results_full=[]
    cleaned_reasults=[]
    for i in df_final.index:
        if i%200==0:
            df_final.to_parquet(args.output_file)
        result=gather_answers(i,df_final, model=model)
        df_final.at[i,args.model+' replies']=str(result)
        # df_final.at[i,args.model+' logprobs']=str(result.choices[0].logprobs.content)
        results_full.append(result)
        cleaned_reasults.append(result)

    df_final.to_parquet(args.output_file)






