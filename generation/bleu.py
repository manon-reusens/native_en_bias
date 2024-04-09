import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

class BleuRunner:
    def __init__(self,df,column):
        """initializes the BertscoreRunner.
        
        Args:
                df: the dataset for which the bleu scores have to be calculated
                column: the column containing the replies for which bleu scores have to be calculated
        """
        self.df = df
        self.column=column
    def __call__(self):
        self.df['BLEU']=np.nan

        self.df['reference']=self.df.apply(lambda x: [x['req_output_x']],axis=1)

        self.df['BLEU']=self.df.apply(lambda row: sentence_bleu(row['reference'],row[self.column]) if (row['dataset_id_x'] in [1,6,7,8]) else np.nan , axis=1)

        self.df=self.df.drop('reference', axis=1)
        print(df)

        return self.df
