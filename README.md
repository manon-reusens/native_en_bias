# Native Design Bias: Studying the Impact of English Nativeness on Language Model Performance

## Installation
```
$ conda create --name native_bias
$ conda activate native bias
$ pip install -r requirements.txt
```
## Dataset
The dataset folder contains the annotations gathered for this paper.
This is used for the experiments.

## Experiments
To gather the output of the different LLMs, the following py-files should be used:
* gather_output_claude.py
* gather_output_gpt.py
* gather_output_qwen.py

## Example Haiku
The results for Haiku on the dataset without adding information about the nativeness of the prompter to the model are gathered as follows:

```
$ python gather_output_claude.py \
$                 --input_file "dataset/final_dataset.parquet" \
$                 --key [Claude KEY] \
$                 --model 'haiku' \
$                 --output_file "[OUTPUT_FILE]" \
$                 --mode "standard" \
```

To gather the gold label, --get_gold_label should be added and set to 'True' on the original_datasets.parquet file containing the unique examples.
To get the results for the expetiments where the correct and wrong nativeness are added, mode should be set to 'add_all_native' or 'add_all_non_native'. To get the results for the experiments where the model first guesses about the nativeness, the mode should be set to 'guess_native'. To gather the results for Sonnet, model should be set to 'sonnet'

## Output other models
for the other models, similar arguments should be used, but the python file should be set to 
* gather_output_gpt.py for gpt3.5 and gpt4_o
* gather_output_qwen for qwen

# Get final results
Once the chatbot's output is gathered, the final scores can be calculated as follows:

```
$ python gather_output_claude.py \
$                 --persistent_dir [DIRECTORY] \
$                 --model [MODEL] \
$                 --mode 'standard' \
$                 --approach 'appr2' \
```
The different modes are standard, standard_no3, and dataset_id. Standard gives the overall averages. Standard_no3 gives the averages for all datasets, except the Amazonfood dataset. Dataset_id provides the average performance per dataset.
The different approaches that can be used. Approach 1 calculates the performance using the original dataset output. Approach 2 uses the generated responses as ground truth.
