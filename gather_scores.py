import argparse 
import os
import pandas as pd
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from classification import Accuracy_Runner
from generation import BertScoreRunner, BleuRunner, RougeRunner
from generation.bart_score import BartScoreRunner


parser = argparse.ArgumentParser(description='Gather scores given the output from chatbots')
parser.add_argument(
                    "--input_file",
                    action="store",
                    type=str,
                    help="Give the full path to the input file",
)

parser.add_argument(
                    "--score",
                    action="store",
                    type=str,
                    default='all',
                    choices=['all','accuracy','bleu','rouge','bertscore','bartscore','parascore','factscore'],
                    help="add the score you want to calculate"
)

parser.add_argument(
                    "--output_file",
                    action="store",
                    type=str,
                    help="Give the full path of where you want to save the output file",
)
parser.add_argument(
                    "--column",
                    action="store",
                    type=str,
                    default='gpt3.5 replies',
                    help="Give the full path of where you want to save the output file",
)
parser.add_argument(
                    "--generated_ground_truth",
                    action="store",
                    default=None,
                    help="add new ground truth file for the generative tasks."
)

def merge_datasets(df1,df_groundtruth):
        #rgs = parser.parse_args()
    if 'Qwen1.5-7B-Chat' in args.column:
        for i in df_groundtruth.index:
            if len(df_groundtruth.at[i,'Qwen1.5-7B-Chat replies'].split(", '"))>1:
                df_groundtruth.at[i,'Qwen1.5-7B-Chat replies']=df_groundtruth.at[i,'Qwen1.5-7B-Chat replies'].split(", '")[1].replace("')","")
            elif len(df_groundtruth.at[i,'Qwen1.5-7B-Chat replies'].split(', "'))>1:
                df_groundtruth.at[i,'Qwen1.5-7B-Chat replies']=df_groundtruth.at[i,'Qwen1.5-7B-Chat replies'].split(', "')[1].replace('")',"")

    df_groundtruth=df_groundtruth.rename(columns={args.column.replace('_history','').replace('_all_native','').replace('_all_non_native','').replace('_guess_native',''):'generated_req_output'})
    df_groundtruth=df_groundtruth.rename(columns={'gpt3.5 replies':'generated_req_output'})
    df_nec=df_groundtruth[['nat_instr_id','generated_req_output']]
    df_merged=pd.merge(df1,df_nec,on='nat_instr_id')
    return df_merged

if __name__ == "__main__":
    args = parser.parse_args()
    df=pd.read_parquet(args.input_file)
    if args.generated_ground_truth != None:
        df_new=pd.read_parquet(args.generated_ground_truth)
        df=merge_datasets(df,df_new)

    if args.column=='Qwen1.5-7B-Chat replies_guess_native':
        for i in df.index:
            if len(df.at[i,'Qwen1.5-7B-Chat replies_guess_native'].split(", '"))>1:
                df.at[i,'Qwen1.5-7B-Chat replies_guess_native']=df.at[i,'Qwen1.5-7B-Chat replies_guess_native'].split(", '")[1].replace("')","")
            elif len(df.at[i,'Qwen1.5-7B-Chat replies_guess_native'].split(', "'))>1:
                df.at[i,'Qwen1.5-7B-Chat replies_guess_native']=df.at[i,'Qwen1.5-7B-Chat replies_guess_native'].split(', "')[1].replace('")',"")

    if args.score=='all' or args.score=='accuracy':
        acc_run=Accuracy_Runner(df,args.column) 
        df=acc_run()
    
    if args.score=='all' or args.score=='bleu':
        bleurun=BleuRunner(df,args.column,args.generated_ground_truth)
        df=bleurun()

    if args.score=='all' or args.score=='rouge':
        rouge_run=RougeRunner(df,args.column,args.generated_ground_truth)
        df=rouge_run()

    if args.score=='all' or args.score=='bertscore':
        bert_run=BertScoreRunner(df,args.column,args.generated_ground_truth)
        df=bert_run()

    if args.score=='all' or args.score=='bartscore':
        bart_run=BartScoreRunner(df,args.column,args.generated_ground_truth) 
        df=bart_run()


    df.to_parquet(args.output_file)
