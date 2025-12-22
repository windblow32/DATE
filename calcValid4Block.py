from turtle import st
from ModelShare_with_DSR_final import Model, Rule, Predicate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import json
import pickle
import pandas as pd
from typing import Dict, List, Optional, Tuple


def load_json_data(file_path: str) -> Dict:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the parsed JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_rule_and_data(block_data: Dict, block_index: int) -> Tuple[
    Optional[str], Optional[List], Optional[float], Optional[int]]:
    """
    Get the rule, corresponding data, and support value at the specified index.
    
    Args:
        block_data: Dictionary containing the JSON data
        block_index: Index of the rule/data block to retrieve (0-based)
        
    Returns:
        Tuple of (rule, data, support) or (None, None, None) if index is out of range or data is invalid
    """
    if not isinstance(block_data, dict):
        return None, None, None

    rules = [k for k in block_data.keys() if not k.startswith('_')]
    if block_index < 0 or block_index >= len(rules):
        return None, None, None

    rule = rules[block_index]
    rule_data = block_data[rule]

    # Handle different possible structures
    if isinstance(rule_data, dict) and 'data' in rule_data:
        data = rule_data['data']
        support = rule_data.get('support')
        modelID = rule_data.get('modelID')
    else:
        data = rule_data
        support = None
        modelID = -1

    return rule, data, support, modelID


def load_model(dataset, model_id):
    """加载模型"""
    filename = f"model/{dataset}_model_{model_id}.pkl"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Model {model_id} not found")
        return None


try:
    # Example 1: Basic usage
    # Load the JSON data
    json_path = "current_block/bank-marketing_ROU=0.15_f1-error_block.json"
    data_path = "dataset/clf_num/bank-marketing.csv"
    data = load_json_data(json_path)

    df = pd.read_csv(data_path, nrows=0)
    # 获取DataFrame的列名，即表头
    header = df.columns.tolist()

    # Process top N blocks
    for index in range(0, 18):
        try:
            rule, block_data, support, modelID = get_rule_and_data(data, index)
            # block_data into dataframe

            df = pd.DataFrame(block_data, columns=header)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

            m = DecisionTreeClassifier(random_state=42)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_val)
            # error rate
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Accuracy: {accuracy}")

        except Exception as e:
            print(f"Error processing block {index}: {e}")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON file - {e}")
except IndexError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
