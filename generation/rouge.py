import pandas as pd
import evaluate

class RougeRunner:
    def __init__(self,df,column):
        """initializes the BertscoreRunner.
        
        Args:
                df: the dataset for which the bleu scores have to be calculated
                column: the column containing the replies for which bleu scores have to be calculated
        """
        self.df = df
        self.column=column
    def __call__(self):
        rouge = evaluate.load('rouge')
        
        predictions=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)][self.column]
        references=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)]['req_output']
        results = rouge.compute(predictions=list(predictions), references=list(references), use_aggregator=False)

        indices=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)].index

        if len(indices) == len(results['rouge1']):
            self.df.loc[indices, 'ROUGE_1'] = results['rouge1']
            self.df.loc[indices, 'ROUGE_2'] = results['rouge2']
            self.df.loc[indices, 'ROUGE_L'] = results['rougeL']
            self.df.loc[indices, 'ROUGE_LSUM']= results['rougeLsum']
        else:
            print('The length of the scores does not match the subset of the DataFrame containing the generation datasets')

        return self.df
