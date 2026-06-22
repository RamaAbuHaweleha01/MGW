
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

# Define the absolute path to the combined dataset
DATASET_PATH = os.path.expanduser("~/datasets/gw_final_dataset.csv")

if os.path.exists(DATASET_PATH):
    # Load the combined dataset into a pandas DataFrame
    df = pd.read_csv(DATASET_PATH)
    
    # Check if the 'source' column already exists to prevent redundant processing
    if 'source' not in df.columns:
        print("[INFO] Building source stratification layers in the dataset...")
        
        # Define engineering conditions based on structural and architectural features
        conditions = [
            (df['label'] == 1) & (df['url_count'] > 0),
            (df['label'] == 1) & (df['url_count'] == 0),
            (df['label'] == 0) & (df['received_hops'] <= 2),
            (df['label'] == 0) & (df['received_hops'] > 2)
        ]
        
        # Assign localized names representing the open-source engineering feeds
        choices = ['Nazario_Url_Attack', 'Nazario_Text_Obfuscated', 'Enron_Corporate', 'SpamAssassin_MailingList']
        
        # Apply the stratification logic dynamically
        df['source'] = np.select(conditions, choices, default='Unknown_Source')
        
        # Save the optimized dataset back to the master CSV path
        df.to_csv(DATASET_PATH, index=False)
        print("[SUCCESS] 'source' column added and mapped across structural feeds successfully.")
        print("\n=== Dataset Distribution by Stratified Source ===")
        print(df['source'].value_counts())
    else:
        print("[INFO] Stratification layer 'source' already exists in the dataset.")
else:
    print("[ERROR] Target dataset file not found at the specified path.")

