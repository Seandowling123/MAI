from transformers import BertTokenizer, BertForSequenceClassification
import torch
import csv
from datetime import datetime
import re
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import time
import os

def load_text_file(file_path):
    try:
        with open(file_path, 'r', encoding="latin-1") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return f"File not found: {file_path}"
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"
    
