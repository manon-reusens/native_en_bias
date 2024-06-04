import pandas as pd
import json
import numpy as np
import re
from evaluate import load


class BertScoreRunner:
    def __init__(self,df,column,ground_truth):
        """initializes the BertscoreRunner.
        
        Args:
                df: the dataset for which the bertscores have to be calculated
                column: the column containing the replies for which bertscores have to be calculated
        """
        self.df = df
        self.column=column
        self.ground_truth=ground_truth
    def __call__(self):
        bertscore = load("bertscore") #to do add the correct dataset ids
        predictions=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)][self.column]
        if self.ground_truth!=None:
            print('we are comparing with the newly generated column')
            references=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)]['generated_req_output']
        else:    
            references=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)]['req_output']
        results = bertscore.compute(predictions=list(predictions), references=list(references), lang="en", model_type="microsoft/deberta-xlarge-mnli")

        indices=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)].index

        if len(indices) == len(results['precision']):
            self.df.loc[indices, 'bertscore_precision'] = results['precision']
            self.df.loc[indices, 'bertscore_recall'] = results['recall']
            self.df.loc[indices, 'bertscore_f1'] = results['f1']
        else:
            print('The length of the scores does not match the subset of the DataFrame containing the generation datasets')

        return self.df