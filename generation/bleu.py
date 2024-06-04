import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

class BleuRunner:
    def __init__(self,df,column,ground_truth=None):
        """initializes the BertscoreRunner.
        
        Args:
                df: the dataset for which the bleu scores have to be calculated
                column: the column containing the replies for which bleu scores have to be calculated
        """
        self.df = df
        self.column=column
        self.ground_truth=ground_truth
    def __call__(self):
        if self.ground_truth!=None:
            self.df['reference']=self.df.apply(lambda x: [x['generated_req_output']],axis=1)
        else:
            self.df['reference']=self.df.apply(lambda x: [x['req_output']],axis=1)

        self.df['BLEU']=self.df.apply(lambda row: sentence_bleu(row['reference'],row[self.column]) if (row['dataset_id'] in [1,6,7,8]) else np.nan , axis=1)

        self.df=self.df.drop('reference', axis=1)

        return self.df
