# CodeLLaMa Fine-tuned
This repository houses code dedicated to fine-tuning the [Code Llama 2](https://huggingface.co/codellama/CodeLlama-7b-hf) model for the specific task of predicting method names. In essence, when presented with code snippets from programming languages like Python and Java, the model is trained to produce an appropriate method name as its output.   
This specialized task is referred to as [code infilling](https://huggingface.co/blog/codellama#code-infilling), wherein the model is conditioned to generate code that aligns  with an existing code's beginning and end sections. The model is fine-tuned with data scraped from the official [Intellij community repository](repositoryhttps://github.com/JetBrains/intellij-community).  
The main objective of this repository is to showcase the adaptation of downstream models and the outcomes to validate the feasibility of the task. Potential future improvements, such as saving models, expanding to support additional languages, and refining the code and methodology, could be considered.  
 
This module is developed using ```python 3.9``` and ```CUDA Version: 11.7``` and requires GPU access.
While other combinations might be possible, they haven't been verified through testing.

## Setup
Steps to steup the module.

1. Clone intellij-community repository:
```
git clone https://github.com/JetBrains/intellij-community
```

2. Clone this repository:
```
git clone https://github.com/KosPsych/codellama-finetune
```

3. Add path to a dataset folder and the intellij repository to ```constants.py```

4. Create virtual environment:
```
conda create --name <name> python==3.9
```

5. Install dependencies:
```
pip install -r requirements.txt
```


## Running
If you've followed the aforementioned steps, the module should now be installed and prepared for use.

### Usage 1
Retrieve methods from the repository using this module. It's designed to parse a repository and capture `Java` and `Python` methods, saving them in a text file with the delimiter "###END###".
To initiate this process, execute the following command:
```
python3 src/dataset_creation.py n_files
```
Here, n_files represents the number of files you wish to process from the specified repository. For instance, if n_files is set to 100, the module will scrape 100 Java files and 100 Python files.

> Prior to rerunning the process, ensure that any files generated within the dataset folder are removed.

### Usage 2
Using the dataset created by ```python3 src/dataset_creation.py n_files``` one can fine tune Code Llama to predict method names with:
```
python3 src/experiments.py lang
```
In the provided code snippet, lang represents the selected programming language, which can be either java or python. This code assesses the base model's performance on the dataset, showcasing accuracy and similarity (measured via the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)) for the task of predicting method names. Following this, the model undergoes fine-tuning using [peft](https://github.com/huggingface/peft). The refined model is then reassessed on the dataset to monitor metric enhancements. The outcomes are presented in the terminal.
