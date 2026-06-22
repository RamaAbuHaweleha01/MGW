import os
import sys
import mailbox
import email
from email import policy
import pandas as pd
from tqdm import tqdm


sys.path.append(os.path.expanduser('~/MGW/Parsing'))


from extract_phishing_features import extract_features_from_email

PHISHING_MBOX = os.path.expanduser('~/datasets/raw_phishing/phishing0.mbox')
HAM_DIR = os.path.expanduser('~/datasets/raw_ham/easy_ham')
OUTPUT_CSV = os.path.expanduser('~/datasets/gw_final_dataset.csv')


HEADER_FEATURE_COLS = [
    "has_dkim", "spf_fail", "dkim_fail", "dmarc_fail",
    "domain_mismatch", "suspicious_tld_sender", "has_numeric_in_domain",
    "has_reply_to", "has_return_path",
    "has_from", "has_to", "has_cc", "has_bcc", "has_subject",
    "has_date", "has_message_id", "received_hops", "date_is_future",
    "subject_all_caps", "subject_caps_ratio", "subject_money",
    "subject_exclamation", "subject_has_numbers", "subject_has_special",
    "subject_length", "subject_word_count",
    "dollar_count", "total_money_symbols",
    "has_script", "has_iframe", "has_form",
    "url_count", "url_has_ip", "url_suspicious_tlds", "url_mismatch_count"
]

def process_dataset():
    dataset_records = []

    
    print("⏳ Phishing emails are being processed...")
    if os.path.exists(PHISHING_MBOX):
        mbox = mailbox.mbox(PHISHING_MBOX, factory=bytes)
        for key in tqdm(mbox.keys()):
            try:
                raw_bytes = mbox.get_string(key).encode('utf-8', errors='ignore')
                msg = email.message_from_bytes(raw_bytes, policy=policy.default)
                
             
                extracted = extract_features_from_email(msg)
                
             
                meta_features = extracted["semantic_meta"]
                
              
                row_data = {}
                for col in HEADER_FEATURE_COLS:
                 
                    row_data[col] = meta_features.get(col, meta_features.get(col.replace('count', 'sign_count'), 0))
                
                row_data['subject_text'] = meta_features.get('subject_text', '')
                row_data['body_text'] = extracted.get('clean_text', '')
                row_data['label'] = 1  # Phishing
                
                dataset_records.append(row_data)
            except Exception:
                continue

    
    print("⏳ Ham (Legitimate) emails are being processed...")
    if os.path.exists(HAM_DIR):
        for filename in tqdm(os.listdir(HAM_DIR)):
            file_path = os.path.join(HAM_DIR, filename)
            if os.path.isdir(file_path):
                continue
            try:
                with open(file_path, 'rb') as f:
                    raw_bytes = f.read()
                msg = email.message_from_bytes(raw_bytes, policy=policy.default)
                
                extracted = extract_features_from_email(msg)
                meta_features = extracted["semantic_meta"]
                
                row_data = {}
                for col in HEADER_FEATURE_COLS:
                    row_data[col] = meta_features.get(col, meta_features.get(col.replace('count', 'sign_count'), 0))
                
                row_data['subject_text'] = meta_features.get('subject_text', '')
                row_data['body_text'] = extracted.get('clean_text', '')
                row_data['label'] = 0  # Ham
                
                dataset_records.append(row_data)
            except Exception:
                continue

    # 3. الدمج والحفظ في CSV
    print("💾 The data is being merged and saved to a CSV file...")
    if dataset_records:
        df = pd.DataFrame(dataset_records)
        df.fillna(0, inplace=True)
        
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Successfully completed! Saved in: {OUTPUT_CSV}")
        print(f"📊 Total samples: {len(df)} | Phishing: {len(df[df['label']==1])} | Ham: {len(df[df['label']==0])}")
    else:
        print("❌ No records were processed. Please check your raw data paths.")

if __name__ == "__main__":
    process_dataset()
