from generation import LLM_as_a_judge 
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Gather scores using an llm as a judge')
parser.add_argument(
                    "--input_path",
                    action="store",
                    type=str,
                    help="Give the full path to the input file",
)
parser.add_argument(
                    "--model_name",
                    action="store",
                    type=str,
                    help="Give the model name you want to evaluate",
)
parser.add_argument(
                    "--run_number",
                    action="store",
                    type=str,
                    help="Give the run number of the model you want to evaluate",
)
parser.add_argument(
                    "--mode",
                    action="store",
                    type=str,
                    default='',
                    help="Give the run number of the model you want to evaluate",
)

if __name__ == "__main__":
    args = parser.parse_args()
    path= args.input_path
    model=args.model_name
    run=args.run_number
    mode=args.mode
    df=pd.read_parquet(path+f'/generative_results_{mode}_{model}_run_{run}.parquet').reset_index()
    eval_model='meta-llama/Llama-3.3-70B-Instruct'
    api_key=''
    if model=='qwen7B_chat':
        model_rep='Qwen1.5-7B-Chat'
    elif model=='gpt3':
        model_rep='gpt3.5'
    else:
        model_rep=model
    judge=LLM_as_a_judge(df,model_rep,run,eval_model,path,api_key,mode)

    full_df=judge()
    full_df.to_parquet(f'{path}/annotated_results_llm_as_a_judge-model_{mode}_{model}-run_{run}.parquet')
