from nltk.metrics.distance import edit_distance
import torch
import editdistance
from datasets import Dataset

class EvaluationMetrics:
    def __init__(self):
        self.all_samples = 0  # True Positives
        self.correct = 0  # False Positives
        self.similarity = 0
       
    def update(self, generated_text, actual_label):
        """Update counters based on generated text and actual label.""" 

        self.similarity += 1 - edit_distance(generated_text, actual_label)/ max( len(generated_text), len(actual_label))
        self.all_samples +=1 
        if generated_text == actual_label:
            self.correct +=1 
          
    def accuracy(self):
        """Compute accuracy."""
        return self.correct / self.all_samples if self.all_samples != 0 else 0
    
    def compute_similarity(self):
        """Compute Levenshtein similarity"""
        return self.similarity / self.all_samples if self.all_samples != 0 else 0


    def display_metrics(self):
        """Display computed metrics."""
        print(f"Accuracy: {self.accuracy()}")
        print(f"L-similarity: {self.compute_similarity()}")


def find_name(method):

    """
    Extracts the method name from a given method string.

    Args:
    - method (dict): A dictionary containing a 'method' key with a string value representing the method signature.

    Returns:
    - str: Extracted method name appended with '_<EOT>'.
    """

    method = method['method']
    end_index = method.find("(", 0)
    start_index = method[:end_index][::-1].find(" ") 
    method_name = method[:end_index][-start_index:].strip()
    return method_name + '_<EOT>'


def replace(method):
    """
    Replaces the extracted method name in the method string with '<FILL_ME>'.

    Args:
    - method (dict): A dictionary containing a 'method' key with a string value representing the method signature.

    Returns:
    - str: Modified method string with the method name replaced by '<FILL_ME>'.
    """
    
    method_name = find_name(method).replace('_<EOT>', '')
    method['method'] = method['method'].replace(method_name, '<FILL_ME>', 1).strip()
    return method['method']
    
   

def formatting_prompts_func(example):
    """
    Formats the prompts using the provided example dictionary.

    Args:
    - example (dict): A dictionary containing 'method' and 'method_name' keys with lists of strings representing method signatures and names respectively.

    Returns:
    - list: A list of formatted prompt strings.
    """

    output_texts = []
    for i in range(len(example['method'])):
        text = f"### Question: {example['method'][i]}\n ### Answer: {example['method_name'][i]}"
        output_texts.append(text)
    return output_texts


def get_method_name(prompt, lang):
  """
    Extracts the method name from the provided prompt based on the specified language.

    Args:
    - prompt (str): The prompt string containing the method signature.
    - lang (str): The programming language of the method ('python' or 'java').

    Returns:
    - str: Extracted method name.
  """    

  if lang == 'python':
            start_index = prompt.find("def") + 4
            end_index = prompt.find("(", start_index)
            method_name = prompt[start_index:end_index].strip()
            
  elif lang == 'java':
            end_index = prompt.find("(", 0)      
            start_index = prompt[:end_index][::-1].find(" ") 
            method_name = prompt[:end_index][-start_index:].strip()    
  return method_name


def create_dataset(path):
    """
    Reads a file, creates a dataset, and maps method names and method replacements.

    Args:
    - path (str): The path to the input file.

    Returns:
    - Dataset: A processed Hugging Face Dataset with method names and replacements.
    """

    with open(path, "r") as file:
        file_content = file.read().split("###END###")
        
    dataset_dict = {
        "method": file_content  
    }

    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    dataset = dataset.map(lambda method: {'method_name': find_name(method)})
    return dataset.map(lambda method: {'method': replace(method)})



     