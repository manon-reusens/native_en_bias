import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Gather reuslt files')
parser.add_argument(
                    "--persistent_dir",
                    action="store",
                    type=str,
                    help="Give the full path to the persistent_dir where the individual results are saved",
)
parser.add_argument(
                    "--model",
                    action="store",
                    type=str,
                    default='gpt3',
                    choices=['all','gpt3', 'gpt4_o','haiku','opus','sonnet','qwen0.5B_chat','qwen7B_chat','llama2_7B_chat'],
                    help="add the model for which you would have to gather the results. if you add all, then we will compare the base versions for all models"
)
parser.add_argument(
                    "--mode",
                    action="store",
                    type=str,
                    default='standard',
                    choices=['standard','dataset_id','standard_no3'],
                    help="Give the full path of where you want to save the output csv file",
)
parser.add_argument(
                    "--approach",
                    action="store",
                    type=str,
                    default='appr2',
                    choices=['appr1','appr2'],
                    help="Give the full path of where you want to save the output csv file",
)

def fill_and_getna(df,acc_col):
    id_class=[0,2,3,4,5,9]
    id_gen=[1,6,7,8]
    print('we have '+str(len(df.loc[df['dataset_id'].isin(id_class) & (df[acc_col].isnull())]))+' missing values for the classification task, which we will fill by the incorrect value')
    index=df.loc[df['dataset_id'].isin(id_class) & (df[acc_col].isnull())].index
    print(index)
    print('we have '+str(len(df.loc[df['dataset_id'].isin(id_gen) & (df['BLEU'].isnull())]))+' missing values for the generation task')
    for i in index:
        df.at[i,acc_col]=0

def make_groups(df):
    df['native_or_not']=df.apply(lambda row: 'native' if ('en' in row['nat_lang']) else 'non_native' , axis=1)
    df['strict_native_or_not']=df.apply(lambda row: 'strict native' if (row['nat_lang']=='{en}') else 'not strict native' , axis=1)
    df['western_native_or_not']=df.apply(lambda row: 'western native' if (row['user_id'] in [159,99,104,110,193,127,457,459,481,114,445,541,542,129,562,563,338,355,254,672,70,700,709,673,687,701,17,255,588,71,356]) else 'not western native' , axis=1)
    return df

def get_results(df,group,mode,scores,prompt):
    if group=='native':
        column='native_or_not'
    elif group=='strict native':
        column='strict_native_or_not'
    elif group=='western':
        column='western_native_or_not'

    if scores=='classification':
        scores='accuracy'
    elif scores=='generation':
        scores='bertscore_f1'
    elif scores=='all':
        scores=['accuracy','BLEU','ROUGE_1','ROUGE_2','ROUGE_L', 'ROUGE_LSUM', 'bertscore_precision', 'bertscore_recall','bertscore_f1', 'bartscore_faithful', 'bartscore_precision','bartscore_recall', 'bartscore_f1']
    else:
        print('Scores should be one of the following: classification, generation, all')
    
    #then depending on the mode
    if mode=='overall':
        res=df.groupby([column,'example_id']).mean(numeric_only=True).groupby([column]).mean(numeric_only=True)[scores]
    if mode=='set':
        res=df.groupby([column,'example_id','set_id']).mean(numeric_only=True).groupby([column,'set_id']).mean(numeric_only=True)[scores]
    if mode=='dataset':
        res=df.groupby([column,'example_id','dataset_id']).mean(numeric_only=True).groupby([column,'dataset_id']).mean(numeric_only=True)[scores]
    res=res.reset_index()
    res['model']=args.model+'_'+prompt
    return res

def find_guess_native_or_not(df,model_name_acc):
    for index, row in df.iterrows():
        if ('native' in row[model_name_acc+' guessed_native']) & ('non-native' not in row[model_name_acc+' guessed_native']):
            df.at[index,'final_guess']='native'
        elif ('non-native' in row[model_name_acc+' guessed_native']) & (' native' not in row[model_name_acc+' guessed_native']):
            df.at[index,'final_guess']='non_native'
        else:
            df.at[index,'final_guess']='unknown'
    return df

if __name__ == "__main__":
    args = parser.parse_args()
    if args.approach=='appr1':
        new_persistent_dir=args.persistent_dir+'/scores_original_gold_label_appr1/'
    elif args.approach=='appr2':
        new_persistent_dir=args.persistent_dir+'/scores_generated_gold_label_appr2/'
    
    if args.model=='all':
        models=['gpt3', 'gpt4_o','haiku','sonnet','qwen7B_chat']#,'llama2_7B_chat','qwen0.5B_chat']
        for i in models:
            if args.approach=='appr1':
                df=pd.read_parquet(new_persistent_dir+'29_05_'+i+'_scores.parquet')
            elif args.approach=='appr2':
                df=pd.read_parquet(new_persistent_dir+'29_05_'+i+'_scores_generated_req_output.parquet')
            if i=='gpt3':
                model_name_acc='gpt3.5'
            elif i=='qwen0.5B_chat':
                model_name_acc='Qwen1.5-0.5B-Chat'
            elif i=='qwen7B_chat':
                model_name_acc='Qwen1.5-7B-Chat'
            elif i=='llama2_7B_chat':
                model_name_acc='meta-llama/Llama-2-7b-chat-hf'
            else:
                model_name_acc=i
            
            df.rename(columns={'accuracy_score_'+model_name_acc+' replies':'accuracy'},inplace=True)
            df=make_groups(df)
            if args.mode=='standard_no3':
                df=df.loc[df['dataset_id']!=3]
            for col in ['native','western','strict native']:
                if args.mode=='standard':
                    df_res=get_results(df,col,'overall','all',args.mode)
                    df_res.to_parquet(new_persistent_dir+'/tables/'+args.mode+'/29_05_individual_results_'+i+'_'+col+'_'+args.mode+'.parquet')
                elif args.mode=='standard_no3':
                    df_res=get_results(df,col,'overall','all',args.mode)
                    df_res.to_parquet(new_persistent_dir+'/tables/'+args.mode+'/29_05_individual_results_'+i+'_'+col+'_'+args.mode+'.parquet')
                else:
                    df_res=get_results(df,col,'dataset','all','standard')
                    df_res.to_parquet(new_persistent_dir+'/tables/dataset/29_05_individual_results_'+i+'_'+col+'_'+args.mode+'.parquet')
                
        
    else:
        if args.approach=='appr1':
            df=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_scores.parquet')
            df_nat=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_all_native_scores.parquet')
            df_non_nat=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_all_non_native_scores.parquet')
            df_guess=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_guess_native_scores.parquet')
        elif args.approach=='appr2':
            df=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_scores_generated_req_output.parquet')
            df_nat=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_all_native_scores_generated_req_output.parquet')
            df_non_nat=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_all_non_native_scores_generated_req_output.parquet')
            df_guess=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_guess_native_scores_generated_req_output.parquet')
            # df_history=pd.read_parquet(new_persistent_dir+'/29_05_'+args.model+'_history_scores_generated_req_output.parquet')



        if args.model=='gpt3':
            model_name_acc='gpt3.5'
        elif args.model=='qwen0.5B_chat':
            model_name_acc='Qwen1.5-0.5B-Chat'
        elif args.model=='qwen7B_chat':
            model_name_acc='Qwen1.5-7B-Chat'
        elif args.model=='llama2_7B_chat':
            model_name_acc='meta-llama/Llama-2-7b-chat-hf'
        else:
            model_name_acc=args.model
        
        df.rename(columns={'accuracy_score_'+model_name_acc+' replies':'accuracy'},inplace=True)
        df_nat.rename(columns={'accuracy_score_'+model_name_acc+' replies_all_native':'accuracy'},inplace=True)
        df_nat.rename(columns={'accuracy_score_'+model_name_acc+' replies':'accuracy'},inplace=True)
        df_non_nat.rename(columns={'accuracy_score_'+model_name_acc+' replies_all_non_native':'accuracy'},inplace=True)
        df_non_nat.rename(columns={'accuracy_score_'+model_name_acc+' replies':'accuracy'},inplace=True)
        df_guess.rename(columns={'accuracy_score_'+model_name_acc+' replies_guess_native':'accuracy'},inplace=True)
        # df_history.rename(columns={'accuracy_score_'+model_name_acc+' replies_history':'accuracy'},inplace=True)

        df_guess=find_guess_native_or_not(df_guess,model_name_acc)
        if args.mode=='standard_no3':
                df=df.loc[df['dataset_id']!=3]
                df_nat=df_nat.loc[df_nat['dataset_id']!=3]
                df_non_nat=df_non_nat.loc[df_non_nat['dataset_id']!=3]
                df_guess=df_guess.loc[df_guess['dataset_id']!=3]

        df=make_groups(df)
        df_nat=make_groups(df_nat)
        df_non_nat=make_groups(df_non_nat)
        df_guess=make_groups(df_guess)
        df_guess_correct=df_guess.loc[df_guess['final_guess']==df_guess['native_or_not']]
        print('original length dataset is', len(df_guess))
        print('the amount of native speakers ', len(df_guess.loc[df_guess['native_or_not']=='native']))
        print('the amount of correctly guessed native speakers ', len(df_guess_correct.loc[df_guess_correct['native_or_not']=='native']))
        print('the amount of correctly guessed non-native speakers ', len(df_guess_correct.loc[df_guess_correct['native_or_not']=='non_native']))
    
        if (args.mode=='standard') or (args.mode=='standard_no3'):
            df_res=get_results(df,'native','overall','all','standard')
            df_res_nat=get_results(df_nat,'native','overall','all','native')
            df_res_non_nat=get_results(df_non_nat,'native','overall','all','non_native')
            df_res_guess=get_results(df_guess,'native','overall','all','guess_native')
            df_res_guess_correct=get_results(df_guess_correct,'native','overall','all','guess_native_correct')
            #df_res_history=get_results(df_guess_correct,'native','overall','all','history')
            #,df_res_history
            df_full=pd.concat([df_res,df_res_nat,df_res_non_nat,df_res_guess,df_res_guess_correct]).set_index(['model','native_or_not']).T.round(decimals=2)
        else:
            df_res=get_results(df,'native','dataset','all','standard')
            df_res_nat=get_results(df_nat,'native','dataset','all','native')
            df_res_non_nat=get_results(df_non_nat,'native','dataset','all','non_native')
            df_res_guess=get_results(df_guess,'native','dataset','all','guess_native')
            df_res_guess_correct=get_results(df_guess_correct,'native','dataset','all','guess_native_correct') 
            #df_res_history=get_results(df_guess_correct,'native','dataset','all','history')
            #,df_res_guess,df_res_guess_correct,df_res_history
            df_full=pd.concat([df_res,df_res_nat,df_res_non_nat,df_res_guess,df_res_guess_correct]).set_index(['model','native_or_not']).T.round(decimals=2)
    
        df_full.to_csv(new_persistent_dir+'/tables/all_prompts/'+args.model+'_'+args.mode+'.csv')
