import re
import pandas as pd

class Accuracy_Runner:
    """
    Gathers the accuracy scores for the different datasets

    """
    def __init__(self, df,column):
        """initializes the accuracy runner.
        
        Args:
                df: the dataset for which the accuracy scores have to be calculated
                column: the column containing the replies for which accuracy scores have to be calculated
        """
        self.df=df
        self.column=column

    def __call__(self):
        self._calculate()

        return self.df

    def _calculate(self):
        self.check_amazon_food()
        self.check_timetravel_sent()
        self.check_abductivenli_sent()
        self.check_tweetqa_mctaco()
        self.check_commonsense()

        return self.df

    def check_amazon_food(self):
        for index, row in self.df.iterrows():
            if row['dataset_id_x']==3:
                extracted_number=0
                for i in re.findall(r'(\d+)',row[self.column]):
                    extracted_number=int(i)+extracted_number
                gold_label=int(row['req_output_x'])
                self.df.at[index,'accuracy_score_'+self.column]=int(extracted_number==gold_label)
        return self.df

    def check_timetravel_sent(self):
        for index, row in self.df.iterrows():
            if row['dataset_id_x']==2:
                gold_label=row['req_output_x']
                if not (('option 1' in row[self.column].lower()) and ('option 2' in row[self.column].lower())):
                    self.df.at[index,'accuracy_score_'+self.column]=int(gold_label.lower() in row[self.column].lower())
                else:self.df.at[index,'accuracy_score_'+self.column]=0
        return self.df

    def check_abductivenli_sent(self):
        for index, row in self.df.iterrows():
            if row['dataset_id_x']==0:
                gold_label=row['req_output_x']
                if gold_label==row[self.column]:
                    self.df.at[index,'accuracy_score_'+self.column]=1
                elif str('middle '+gold_label) in row[self.column].lower() and not ((str('middle 1') in row[self.column].lower()) and (str('middle 2') in row[self.column].lower())):
                    self.df.at[index,'accuracy_score_'+self.column]=1
                else:
                    self.df.at[index,'accuracy_score_'+self.column]=0
        return self.df
    def check_tweetqa_mctaco(self):
        for index, row in self.df.iterrows():
            if row['dataset_id_x']==5 or row['dataset_id_x']==4 :
                gold_label=row['req_output_x']
                if not ('no' in row[self.column].lower() and 'yes' in row[self.column].lower()):
                    self.df.at[index,'accuracy_score_'+self.column]=int(gold_label.lower().replace('.','') in row[self.column].lower())
                else: print(row['eval'].lower())
        return self.df
    def check_commonsense(self):
        for index, row in self.df.iterrows():
            if row['dataset_id_x']==9 :
                gold_label=row['req_output_x']
                num_in_text=0
                if 'A' in row[self.column]:
                    num_in_text+=1
                if 'B' in row[self.column]:
                    num_in_text+=1
                if 'C' in row[self.column]:
                    num_in_text+=1
                if num_in_text==1:
                    self.df.at[index,'accuracy_score_'+self.column]=int(gold_label in row[self.column])
                else: print(row['eval'])
        return self.df
    

    
