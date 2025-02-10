import numpy as np
import pandas as pd
import os
cache_dir='/scratch/leuven/344/vsc34470'
os.environ['TRANSFORMERS_CACHE'] =cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List
import openai
import json
# based on https://github.com/dmg-illc/JUDGE-BENCH
# obij Summeval json, zo wij ook prompt daarop baserensummeval

#fluency en relevance

class LLM_as_a_judge():
    def __init__(self,df,model,run,eval_model,start_path,api_key='',mode=''):
        #based on human annotations for CNN dailymail
        self.df=df
        self.doc_gen=''
        self.doc_comp=''
        self.model=model #model has to be one of the following: [Qwen1.5-7B-Chat, haiku, sonnet, gpt3.5, gpt4_o]
        self.eval_model=eval_model
        self.api_key=api_key
        self.start_path=start_path
        self.run=run
        self.task_mapping = {
            'task1553': ('news article', 'summary'),
            'task1161': ('article', 'title'),
            'task177': ('paraphrased sentence', 'sentence'),
            'task105': ('closing sentence', 'story'),
        }
        self.mode=mode
    def make_user_prompts(self,row):
        start_prompt_one=f'You will be given a '+row["doc_gen"]+" generated based on a "+row["doc_comp"]+".\n\nYour task is to rate the "+row["doc_gen"]+" on one metric.\n\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed."
        start_prompt_two=f'You will be given a '+row["doc_gen"]+" and a "+row["doc_comp"]+". \n\nYour task is to rate the "+row["doc_gen"]+" on one metric.\n\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed."
            
        fluency = f"Evaluation Criteria:\n\nFluency (1-3): the quality of the "+row["doc_gen"]+" in terms of grammar, spelling, punctuation, word choice, and sentence structure. Assign a score on a scale of 1 to 3 where: \n\n- 1: Poor. The "+row["doc_gen"]+" has many errors that make it hard to understand or sound unnatural.\n- 2: Fair. The "+row["doc_gen"]+" has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.\n- 3: Good. The "+row["doc_gen"]+" has few or no errors and is easy to read and follow. \n\n\nEvaluation Form (scores ONLY):\n\n- Fluency:"
        coherence_art=f"Evaluation Criteria:\n\nCoherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby \"the "+row["doc_gen"]+" should be well-structured and well-organized. The "+row["doc_gen"]+" should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic.\"\n\nEvaluation Steps:\n\n1. Read the "+row["doc_comp"]+" carefully and identify the main topic and key points.\n2. Read the "+row["doc_gen"]+" and compare it to the "+row["doc_comp"]+". Check if the "+row["doc_gen"]+" covers the main topic and key points of the "+row["doc_comp"]+", and if it presents them in a clear and logical order.\n3. Assign a score for coherence on a scale of 1 to 5, where 1: Very low coherence ; 2: Low coherence; 3: Mediocre coherence ; 4: High coherence ; 5: Very high coherence.\n\n\nEvaluation Form (scores ONLY):\n\n- Coherence:"
        coherence_story=f"Evaluation Criteria:\n\nCoherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby \"the sentences should be well-structured and well-organized. The sentences should not just be a heap of related information, but should build from sentence to a coherent story.\"\n\nEvaluation Steps:\n\n1. Read the "+row["doc_comp"]+" carefully and identify the main topic and key points.\n2. Read the "+row["doc_gen"]+" and compare it to the "+row["doc_comp"]+". Check if the sentences are clear and in a logical order.\n3. Assign a score for coherence on a scale of 1 to 5, where 1: Very low coherence ; 2: Low coherence; 3: Mediocre coherence ; 4: High coherence ; 5: Very high coherence. \n\n\nEvaluation Form (scores ONLY):\n\n- Coherence:"
        coherence_paraphrase=f"Evaluation Criteria:\n\nCoherence (1-5) - The overall quality of the paraphrased sentence in terms of logical flow, structure, and alignment with the original sentence. A coherent paraphrase should preserve the meaning of the original sentence, avoid redundancy, and introduce variation without altering the main idea. The paraphrased sentence should not feel disjointed or incomplete but should read smoothly as a standalone sentence. \"\n\nEvaluation Steps:\n\n1. Read the "+row["doc_comp"]+" carefully and identify the main topic and key points. \n2. Read the "+row["doc_gen"]+" and compare it to the "+row["doc_comp"]+". \n3. Assign a score for coherence on a scale of 1 to 5, where 1: Very low coherence ; 2: Low coherence; 3: Mediocre coherence ; 4: High coherence ; 5: Very high coherence.\n\n\nEvaluation Form (scores ONLY):\n\n- Coherence:"
        relevance = f"Evaluation Criteria:\n\nRelevance (1-5) - inclusion of important content from the "+row["doc_comp"]+". The "+row["doc_gen"]+" should include all important information from the "+row["doc_comp"]+". \n\nEvaluation Steps:\n\n1. Read the "+row["doc_comp"]+" and the "+row["doc_gen"]+" carefully.\n2. Compare the "+row["doc_gen"]+" to the "+row["doc_comp"]+" and identify the main points of the "+row["doc_comp"]+".\n3. Assess how well the "+row["doc_gen"]+" covers the main points of the "+row["doc_comp"]+", and how much irrelevant or redundant information it contains.\n4. Assign a relevance score from 1 to 5 where 1: Very low relevance ; 2: Low relevance; 3: Mediocre relevance ; 4: High relevance ; 5: Very high relevance. \n\n\nEvaluation Form (scores ONLY):\n\n- Relevance:"
        relevance_story = f"Evaluation Criteria:\n\nRelevance (1-5) - The degree to which the generated "+row["doc_gen"]+" effectively reflects the main themes and purpose of the "+row["doc_comp"]+". A relevant closing sentence should provide a meaningful and appropriate conclusion, aligning with the tone and key points of the narrative. \n\nEvaluation Steps:\n\n1. Read the "+row["doc_comp"]+" and the "+row["doc_gen"]+" carefully.\n2. Compare the "+row["doc_gen"]+" to the "+row["doc_comp"]+" and identify the main points of the "+row["doc_comp"]+".\n3. Assess how well the "+row["doc_gen"]+" concludes the "+row["doc_comp"]+", and how much irrelevant or redundant information it contains.\n4. Assign a relevance score from 1 to 5 where 1: Very low relevance ; 2: Low relevance; 3: Mediocre relevance ; 4: High relevance ; 5: Very high relevance.  \n\n\nEvaluation Form (scores ONLY):\n\n- Relevance:"
        
        if self.mode=='':
            prompt_fluency= start_prompt_one + f'\n\n'+ row["doc_gen"]+': \n'+ row[self.model+' replies']+f' \n\n {fluency}'
        else:
            prompt_fluency= start_prompt_one + f'\n\n'+ row["doc_gen"]+': \n'+ row[self.model+' replies_'+self.mode]+f' \n\n {fluency}'
        print(prompt_fluency)
        
        
        if  'article' in row['doc_gen']:
            if self.mode=='':
                prompt_coherence= start_prompt_two + f'\n\n'+ row["doc_gen"]+': \n'+ row[self.model+' replies']+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {coherence_art}"
                prompt_relevance= start_prompt_two + f"\n\n "+row['doc_gen']+": \n "+ row[self.model+' replies']+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {relevance}"
            else:
                prompt_coherence= start_prompt_two + f'\n\n'+ row["doc_gen"]+': \n'+ row[self.model+' replies_'+self.mode]+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {coherence_art}"
                prompt_relevance= start_prompt_two + f"\n\n "+row['doc_gen']+": \n "+ row[self.model+' replies_'+self.mode]+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {relevance}"
        elif 'closing' in row['doc_gen']:
            if self.mode=='':
                prompt_coherence=start_prompt_two + f"\n\n "+row['doc_gen']+": \n "+ row[self.model+' replies']+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {coherence_story}"
                prompt_relevance= start_prompt_two + f"\n\n "+row['doc_gen']+": \n"+ row[self.model+' replies']+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {relevance_story}"
            else:
                prompt_coherence=start_prompt_two + f"\n\n "+row['doc_gen']+": \n "+ row[self.model+' replies_'+self.mode]+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {coherence_story}"
                prompt_relevance= start_prompt_two + f"\n\n "+row['doc_gen']+": \n"+ row[self.model+' replies_'+self.mode]+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {relevance_story}"
        elif 'paraphrased' in row['doc_gen']:
            if self.mode=='':
                prompt_coherence=start_prompt_two + f"\n\n "+row['doc_gen']+": \n"+ row[self.model+' replies']+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {coherence_paraphrase}"
                prompt_relevance= start_prompt_two + f"\n\n "+row['doc_gen']+": \n"+ row[self.model+' replies']+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {relevance}"
            else:
                prompt_coherence=start_prompt_two + f"\n\n "+row['doc_gen']+": \n"+ row[self.model+' replies_'+self.mode]+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {coherence_paraphrase}"
                prompt_relevance= start_prompt_two + f"\n\n "+row['doc_gen']+": \n"+ row[self.model+' replies_'+self.mode]+f" \n\n "+row['doc_comp']+": \n "+row['final_prompt_en']+f" \n\n {relevance}"

        else:
            print(row['doc_gen'])
        user_prompts=[prompt_fluency,prompt_coherence,prompt_relevance]

        return user_prompts
    def gather_gpt4_results(self):
        for ind, row in self.df.iterrows():
            if 'task1553' in row['nat_instr_id']:
                self.doc_gen='news article'
                self.doc_comp='summary'
            if 'task1161' in row['nat_instr_id']:
                self.doc_gen='article'
                self.doc_comp='title'
            if 'task177' in row['nat_instr_id']:
                self.doc_gen='paraphrased sentence'
                self.doc_comp='sentence'
            if 'task105' in row['nat_instr_id']:
                self.doc_gen='closing sentence'
                self.doc_comp='story'
            user_prompts=self.make_user_prompts(row)

            system_prompts="You will help annotating a dataset. Answer the questions as asked, do not provide extra explanations, only choose one of the provided options."
            input_list=[[{"role":"system", "content":system_prompts},
                            {"role":"user", "content":u_prompt}] for u_prompt in user_prompts] 
            if self.eval_model=='gpt4o':
                api='gpt4o'
                batch_input=[{ "custom_id": f"request-{ind}-{input_list.index(input_mes)}", "method": "POST", "url": "/v1/chat/completions", "body": { "model": "gpt-4o-2024-08-06", "messages": input_mes, "max_tokens": 1000 }} for input_mes in input_list]
              
            if ind %20 ==0:
                full_batch_input_list=[]
                corr_ind_file=ind
                with open(self.start_path+f"/batches/batchinput_{self.eval_model}_{ind}.jsonl","w") as f:
                    for d in batch_input:
                        json.dump(d,f)
                        f.write("\n")
                        full_batch_input_list.append(d)
            else:
                with open(self.start_path+f"/batches/batchinput_{self.eval_model}_{corr_ind_file}.jsonl", "a") as f:
                    for d in batch_input:
                        json.dump(d,f)
                        f.write("\n")
                        full_batch_input_list.append(d)
            if ((ind+1)%10==0) or (ind+1==len(self.df)):
                if self.eval_model=='gpt4o':
                    client = openai.OpenAI(api_key=self.api_key)
                    response = client.files.create(file=open(self.start_path+f"/batches/batchinput_{self.eval_model}_{corr_ind_file}.jsonl",'rb'), purpose='batch' )
                    file_id = response.id 
                    response = client.batches.create( input_file_id=file_id, endpoint='/v1/chat/completions', completion_window='24h' ) 
                    batch_id = response.id
                print(batch_id)
                with open(self.start_path+f"/batches/{self.eval_model}_batchids.txt", "a") as f:
                    f.write(batch_id)  # add batch_id
                    f.write("\n")

    def gather_llama70B_results(self):
        tokenizer = AutoTokenizer.from_pretrained(self.eval_model)
        model = LLM(model=self.eval_model, tensor_parallel_size=2,seed=42,dtype='bfloat16',max_model_len=4096,download_dir='/scratch/leuven/344/vsc34470')
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        # Map tasks to doc_gen and doc_comp
        values_gen=["news article","article", "paraphrased sentence", "closing sentence"]
        values_comp=["summary", "title", "sentence", "story"]
        conditions=[
                self.df['nat_instr_id'].str.contains('task1553', na=False),
                self.df['nat_instr_id'].str.contains('task1161', na=False),
                self.df['nat_instr_id'].str.contains('task177', na=False),
                self.df['nat_instr_id'].str.contains('task105', na=False),
                ]
        
        self.df['doc_gen']=np.select(conditions,values_gen, default='other')
        self.df['doc_comp']=np.select(conditions,values_comp, default='other')

        # Create user prompts for all rows
        self.df['user_prompts'] = self.df.apply(self.make_user_prompts, axis=1)
        #print(self.df[self.df['doc_gen']=='other'])
        input_list = []
        for _, row in self.df.iterrows():
            system_prompt = "You will help annotating a dataset. Answer the questions as asked do not provide extra explanations, only choose one of the provided options."
            user_prompts = row['user_prompts']
            print(user_prompts)
            input_list.extend(
                [
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": u_prompt},
                    ]
                    for u_prompt in user_prompts
                ]
            )
        batch_size=16
        output_list = []
        for i in range(0, len(input_list), batch_size):
            batch = input_list[i:i+batch_size]
            batch_tok = [tokenizer.apply_chat_template(user_input, tokenize=False, add_special_tokens=False, add_generation_prompt=True) for user_input in batch]
            batch_output = model.generate(batch_tok, sampling_params=sampling_params, use_tqdm=True)
            output_list.extend([output.outputs[0].text.strip() for output in batch_output])
            #with open(f'/scratch/leuven/344/vsc34470/non-native/batches/batch{i}_everything_{self.model}-run_{self.run}-eval_model_{self.eval_model.replace("/","")}.txt', 'w',encoding='utf8') as f:
                #f.write(str(output_list))
            

        self.assign_responses(output_list)

        return self.df
    
    def assign_responses(self, response_texts):
        # Number of prompts per row (fluency, coherence, relevance)
        num_scores_per_row = 3

        # Reshape the flat list of responses into rows
        reshaped_responses = [
            response_texts[i:i + num_scores_per_row]
            for i in range(0, len(response_texts), num_scores_per_row)
        ]

        # Ensure the reshaped list matches the DataFrame size
        if len(reshaped_responses) != len(self.df):
            raise ValueError("Mismatch between number of rows and responses generated.")

        # Assign each score to the correct column
        self.df[['fluency', 'coherence', 'relevance']] = pd.DataFrame(reshaped_responses, index=self.df.index)
  

    def __call__(self):
        self.gather_llama70B_results()
        print(self.df)
        return self.df
