import pandas as pd
import datetime
from transformers import BertTokenizer,BertModel
import matplotlib.pyplot as plt
from umap import UMAP
import plotly.express as px
import torch
import re
import argparse

parser = argparse.ArgumentParser(description='Gather output from chatbots')
parser.add_argument(
                    "--input_file",
                    action="store",
                    type=str,
                    help="Give the full path to the input file",
)
parser.add_argument(
                    "--output_dir",
                    action="store",
                    type=str,
                    help="Give the full path to the output directory",
)

def add_time(df,maxmin=10):
    """
        This function is used to calculate the time spent per annotation by the different annotators.

        The function requires the following inputs: 
        df: dataset containing annotations and timestamps
        maxmin: given that people can stop at any time, we have to estimate an upper limit for the amount spent on the annotations

        output: dataframe
    """
    max_min=datetime.timedelta(minutes=maxmin)
    df = df.sort_values(by=['user_id', 'time'])
    df['time_diff'] = df.groupby('user_id')['time'].diff()
    df['time_diff_capped']=df['time_diff'].where(df['time_diff']<max_min, max_min)
    return df

def count_words(df):
    df['word_count_annotation']=df['prompt_en'].str.split().apply(len)
    if 'gpt3.5 replies' in df.columns:
        df['word_count_generated']=df['gpt3.5 replies'].str.split().apply(len)
    elif 'gpt4 replies' in df.columns:
        df['word_count_generated']=df['gpt4 replies'].str.split().apply(len)
    return df

def amazon_food(df):
    if 'gpt3.5 replies' in df.columns:
        column='gpt3.5 replies'
    if 'gpt4 replies' in df.columns:
        column='gpt4 replies'
    for index, row in df.iterrows():
        if row['dataset_id']==3:
            extracted_number=0
            
            for i in re.findall(r'(\d+)',row[column]):
                extracted_number=int(i)+extracted_number
            df.at[index,'predicted_score_amazon']=extracted_number
    return df

def make_groups(df):
    df['native_or_not']=df.apply(lambda row: 'native' if ('en' in row['nat_lang']) else 'non_native' , axis=1)
    df['strict_native_or_not']=df.apply(lambda row: 'strict native' if (row['nat_lang']=='{en}') else 'not strict native' , axis=1)
    df['western_native_or_not']=df.apply(lambda row: 'western native' if (row['user_id'] in [159,99,104,110,193,127,457,459,481,114,445,541,542,129,562,563,338,355,254,672,70,700,709,673,687,701,17,255,588,71,356]) else 'not western native' , axis=1)
    df['african_or_not']=df.apply(lambda row: 'african' if (row['user_id'] in [162,14,375,443,536,670,458,540]) else 'not african' , axis=1)
    df['african_or_not']=df.apply(lambda row: 'student' if (row['user_id'] in [7,190,67,191,9,192,10,11,12,68,69,312,313,13,37,581,580,582,251,640,352,353,583,600,601,660,354,602,620,621,14,670,671,38,622,15,193,685,585,703,686,16,704,586,708,39,697,695,698,699,645,587,253,355,314,254,672,70,700,709,673,687,701,17,255,588,71,356]) else 'not student' , axis=1)
    return df

def get_cls_embeddings(tokenizer, model,text,batch_size=32):
    
    embeddings=[]
    for i in range(0,len(text),batch_size):
        batch=text[i:i+batch_size]
        encoded_input=tokenizer(batch,return_tensors='pt',padding=False,truncation=False) #check what to do... miss wel
        with torch.no_grad():
            output=model(**encoded_input)

        #extract CLS embeddings
        cls_embeddings=output.last_hidden_state[:,0,:]
        embeddings.extend(cls_embeddings)

    #ensure embeddings are detached from GPU and converted to numpy if running on CUDA
    embeddings=[e.detach().cpu.numpy() for e in embeddings]
    return embeddings

def annotation_embeddings(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    df['embeddings_prompt']=get_cls_embeddings(tokenizer, model,df['prompt_en'].to_list())
    if 'gpt3.5 replies' in df.columns:
        df['embeddings_gpt3']=get_cls_embeddings(tokenizer, model,df['gpt3.5 replies'].to_list())
    elif 'gpt4 replies' in df.columns:
        df['embeddings_gpt4']=get_cls_embeddings(tokenizer, model,df['gpt4 replies'].to_list())
    return df

if __name__ == "__main__":
    args = parser.parse_args()
    df=pd.read_parquet(args.input_file)
    df=add_time(df)
    df=count_words(df)
    df=amazon_food(df)
    df=make_groups(df)
    df=annotation_embeddings(df)

    if 'gpt3.5 replies' in df.columns:
        output_file='statistics_gpt3.5_replies.parquet' 
    elif 'gpt4 replies' in df.columns:
        output_file='statistics_gpt4_replies.parquet' 
    df.to_parquet(args.ouput_dir+'/'+output_file)

    #create time statistics
    average_time_diff = df.groupby(['set_id','native_or_not','strict_native_or_not','western_native_or_not','african_or_not'])['time_diff_capped'].mean()
    sum_time_diff = df.groupby(['set_id','native_or_not','strict_native_or_not','western_native_or_not','african_or_not','african_or_not'])['time_diff_capped'].sum()

    average_time_diff.to_parquet(args.ouput_dir+'/average_time_per_set_id_and_group.parquet')
    sum_time_diff.to_parquet(args.ouput_dir+'/total_time_per_set_id_and_group.parquet')

    #create word count statistics
    average_word_count_prompt = df.groupby(['dataset_id','native_or_not','strict_native_or_not','western_native_or_not','african_or_not','african_or_not'])['word_count_annotation'].mean()
    average_word_count_model = df.groupby(['dataset_id','native_or_not','strict_native_or_not','western_native_or_not','african_or_not','african_or_not'])['word_count_generated'].sum()

    average_word_count_prompt.to_parquet(args.ouput_dir+'/average_wordcount_prompt_per_dataset_id_and_group.parquet')
    average_word_count_model.to_parquet(args.ouput_dir+'/average_wordcount_model_per_dataset_id_and_group.parquet')

    #create statistics about distribution AmazonFood
    df_amazon=df.loc[df['dataset_id']==3]
    for i in ['native_or_not','strict_native_or_not','western_native_or_not','african_or_not']:
        for value in df_amazon[i].unique:
            max=df_amazon.loc[df_amazon[i]==value]['predicted_score_amazon'].max()
            df_amazon.loc[df_amazon[i]==value][[i,'predicted_score_amazon']].plot().hist(bins=range(0,max + 1, 1))
            plt.savefig(args.ouput_dir+'/hist_amazonfood_'+value+'.svg')

    #create UMAP visualization from the embeddings
    if 'gpt3.5 replies' in df.columns:
        embeddins=['embeddings_prompt','embeddings_gpt4']
    elif 'gpt4 replies' in df.columns:
        embeddings=['embeddings_prompt','embeddings_gpt4']
    for j in ['embeddings_prompt','embeddings_gpt4']:
        for i in ['native_or_not','strict_native_or_not','western_native_or_not','african_or_not']:
            features = df[['embeddings_prompt',i]]

            umap_2d = UMAP(n_components=2, init='random', random_state=0)

            proj_2d = umap_2d.fit_transform(features)

            fig_2d = px.scatter(
                proj_2d, x=0, y=1,
                color=df[i], labels={'color': i}
            )
            fig_2d.write_image(args.ouput_dir+"/umap_"+j+'_'+i+".svg")
            fig_2d.show()
