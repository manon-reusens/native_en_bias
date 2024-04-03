import argparse 
import os
import pandas as pd
import sys

current_dir = os.getcwd()
sys.path.append(current_dir)

from classification import Accuracy_Runner

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

if __name__ == "__main__":
    args = parser.parse_args()
    df=pd.read_parquet(args.input_file)

    if args.score=='all' or args.score=='accuracy':
        acc_run=Accuracy_Runner(df,'gpt3.5 replies') #make variable
        df_acc=acc_run()
    
    df_acc.to_parquet(args.output_file)

    # def gather_results_native_nonnative(self,dataset_id):
    #     df_classresults_corr=self.df[(self.df['dataset_id_x']==dataset_id)& ((df_classresults['level_en']=='C1') | (df_classresults['level_en']=='C2')) & (df_classresults['user_id']>50)]
    #     users_to_exclude=df_classresults_corr['user_id'].value_counts().loc[~df_classresults_corr['user_id'].value_counts().ge(10)] #note: 10 should be half of the available examples
    #     for index in users_to_exclude.index:
    #         df_classresults_corr=df_classresults_corr.loc[df_classresults_corr['user_id']!=index]
        
    #     results=df_classresults_corr.groupby('user_id')['accuracy_score_gpt3'].mean()
    #     final=pd.merge(results,df_classresults[['user_id','nat_lang']],on='user_id',how='inner').drop_duplicates()

    #     score_en=[]
    #     score_non_en=[]
    #     for i,row in final.iterrows():
    #         if 'en' in row['nat_lang']:
    #             score_en.append(row['accuracy_score_gpt3'])
    #         else:
    #             score_non_en.append(row['accuracy_score_gpt3'])
    #     print('the overall score for EN for set 10 is ', sum(score_en)/len(score_en))
    #     print('the overall score for non-EN for set 10 is ', sum(score_non_en)/len(score_non_en))
        
    #     return final, score_en, score_non_en