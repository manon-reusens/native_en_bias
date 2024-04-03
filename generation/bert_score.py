import pandas as pd
import json
import numpy as np
import re
from evaluate import load


class BertScore:
    def __init__(self,df):
        self.df = df
    def __call__(self):
        bertscore = load("bertscore") #to do add the correct dataset ids
        dataset_ids=[]
        predictions=self.df.loc[(self.df['dataset_id_x']==6) | (self.df['dataset_id_x']==8)]['gpt3.5 replies']
        references=self.df.loc[(self.df['dataset_id_x']==6) | (self.df['dataset_id_x']==8)]['req_output_x']
        results = bertscore.compute(predictions=list(predictions), references=list(references), lang="en")