# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_excel('ADHD Symptom Self-Assessment (Responses).xlsx')

print("Model training completed!")