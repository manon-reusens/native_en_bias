from .bart import BARTScorer

class BartScoreRunner:
    def __init__(self,df,column):
        """initializes the BertscoreRunner.
        
        Args:
                df: the dataset for which the bertscores have to be calculated
                column: the column containing the replies for which bertscores have to be calculated
        """
        self.df = df
        self.column=column
        self.df['full_instruction']=self.df['task_def']+' '+self.df['final_prompt_en']
        self.predictions=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)][self.column]
        self.references=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)]['req_output']
        self.input=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)]['full_instruction']

    def __call__(self,batch_size=4):
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')

        results_precision = bart_scorer.score(list(self.references), list(self.predictions),batch_size=batch_size)
        results_recall = bart_scorer.score(list(self.predictions), list(self.references),batch_size=batch_size)
        results_faithful=bart_scorer.score(list(self.input), list(self.predictions),batch_size=batch_size)
        multiplication=[2*a*b for a,b in zip(results_precision,results_recall)]
        sum=[a+b for a,b in zip(results_precision,results_recall)]
        results_f1=[a/b for a,b in zip(multiplication,sum)]

        indices=self.df.loc[(self.df['dataset_id']==1) | (self.df['dataset_id']==6)|(self.df['dataset_id']==7) | (self.df['dataset_id']==8)].index

        if len(indices) == len(results_precision)== len(results_recall)== len(results_faithful):
            self.df.loc[indices, 'bartscore_faithful'] = results_faithful
            self.df.loc[indices, 'bartscore_precision'] = results_precision
            self.df.loc[indices, 'bartscore_recall'] = results_recall
            self.df.loc[indices, 'bartscore_f1'] = results_f1
        else:
            print('The length of the scores does not match the subset of the DataFrame containing the generation datasets')

        self.df=self.df.drop('full_instruction', axis=1)

        return self.df