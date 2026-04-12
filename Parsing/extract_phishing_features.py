#!/usr/bin/env python3
"""
Comprehensive Email Feature Extractor for Phishing Email Dataset
Parses CSV dataset and extracts all possible features from emails
"""

import pandas as pd
import numpy as np
import re
import json
import os
import hashlib
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# For text processing
import html
from urllib.parse import urlparse

class PhishingEmailFeatureExtractor:
    """
    Extract comprehensive features from phishing email dataset
    """
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.features_df = None
        self.feature_descriptions = {}
        
    def load_dataset(self):
        """Load the CSV dataset"""
        print(f"📂 Loading dataset from {self.csv_file}...")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(self.csv_file, encoding=encoding)
                print(f"✅ Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if self.df is None:
            raise Exception("Could not load file with any encoding")
        
        print(f"📊 Dataset shape: {self.df.shape}")
        print(f"📋 Columns: {list(self.df.columns)}")
        
        return self.df
    
    def explore_dataset(self):
        """Explore and understand the dataset structure"""
        print("\n" + "="*60)
        print("🔍 DATASET EXPLORATION")
        print("="*60)
        
        # Basic info
        print(f"\n📊 Dataset Info:")
        print(f"  - Total emails: {len(self.df)}")
        print(f"  - Total columns: {len(self.df.columns)}")
        
        # Check for null values
        null_counts = self.df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"\n⚠️  Columns with null values:")
            for col in null_counts[null_counts > 0].index:
                print(f"  - {col}: {null_counts[col]} nulls ({null_counts[col]/len(self.df)*100:.1f}%)")
        
        # Data types
        print(f"\n📋 Column data types:")
        for col in self.df.columns:
            print(f"  - {col}: {self.df[col].dtype}")
        
        # Sample data
        print(f"\n📝 Sample data (first 2 rows):")
        print(self.df.head(2).to_string())
        
        # Check for label column
        label_variations = ['label', 'class', 'type', 'category', 'spam', 'phishing', 'is_spam']
        found_label = None
        for col in self.df.columns:
            if col.lower() in label_variations:
                found_label = col
                break
        
        if found_label:
            print(f"\n🏷️  Label column found: '{found_label}'")
            print(f"   Value counts:")
            print(self.df[found_label].value_counts())
        else:
            print(f"\n⚠️  No obvious label column found")
    
    def extract_header_features(self, row):
        """Extract features from email headers if available"""
        features = {}
        
        # Check for email headers in columns
        header_fields = ['from', 'to', 'cc', 'bcc', 'subject', 'date', 
                        'message-id', 'reply-to', 'return-path', 'dkim']
        
        for field in header_fields:
            # Look for variations of column names
            col_variations = [field, field.replace('-', '_'), field.upper(), field.title()]
            found = False
            
            for col in self.df.columns:
                if col.lower() in [v.lower() for v in col_variations]:
                    value = row.get(col, '')
                    if pd.notna(value):
                        features[f'has_{field.replace("-", "_")}'] = 1
                        
                        # Extract specific features for certain fields
                        if field == 'from':
                            features['from_length'] = len(str(value))
                            # Extract domain if email
                            email_match = re.search(r'@([^>\s]+)', str(value))
                            if email_match:
                                features['from_domain'] = email_match.group(1)
                                features['has_from_domain'] = 1
                            else:
                                features['from_domain'] = 'unknown'
                                features['has_from_domain'] = 0
                        
                        elif field == 'to':
                            # Count recipients
                            recipients = re.findall(r'@[^>\s,]+', str(value))
                            features['recipient_count'] = len(recipients)
                        
                        elif field == 'subject':
                            features['subject_length'] = len(str(value))
                            features['subject_word_count'] = len(str(value).split())
                            features['subject_has_reply'] = 1 if str(value).lower().startswith('re:') else 0
                            features['subject_has_fwd'] = 1 if str(value).lower().startswith('fwd:') else 0
                            features['subject_has_urgent'] = 1 if 'urgent' in str(value).lower() else 0
                            features['subject_has_alert'] = 1 if 'alert' in str(value).lower() else 0
                            features['subject_has_verify'] = 1 if 'verify' in str(value).lower() else 0
                            features['subject_all_caps'] = 1 if str(value).isupper() else 0
                            features['subject_caps_ratio'] = sum(1 for c in str(value) if c.isupper()) / (len(str(value)) + 1)
                            features['subject_has_numbers'] = 1 if any(c.isdigit() for c in str(value)) else 0
                            features['subject_has_special'] = 1 if any(c in '!@#$%^&*()_+{}[]|\\:;"' for c in str(value)) else 0
                            features['subject_exclamation'] = str(value).count('!')
                            features['subject_question'] = str(value).count('?')
                            features['subject_money'] = 1 if any(c in '$€£¥' for c in str(value)) else 0
                        
                        elif field == 'date':
                            try:
                                # Parse date if possible
                                date_str = str(value)
                                # Simple date features
                                features['has_date'] = 1
                                features['date_length'] = len(date_str)
                                
                                # Try to extract hour if time present
                                time_match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', date_str)
                                if time_match:
                                    features['hour_sent'] = int(time_match.group(1))
                                    features['minute_sent'] = int(time_match.group(2))
                                else:
                                    features['hour_sent'] = -1
                                    features['minute_sent'] = -1
                            except:
                                features['has_date'] = 0
                                features['hour_sent'] = -1
                                features['minute_sent'] = -1
                    else:
                        features[f'has_{field.replace("-", "_")}'] = 0
                    found = True
                    break
            
            if not found:
                features[f'has_{field.replace("-", "_")}'] = 0
        
        return features
    
    def extract_body_features(self, text):
        """Extract comprehensive features from email body text"""
        features = {}
        
        if pd.isna(text) or not isinstance(text, str):
            text = ''
        
        text = str(text)
        
        # Basic statistics
        features['body_length'] = len(text)
        features['body_word_count'] = len(text.split())
        features['body_line_count'] = len(text.split('\n'))
        features['body_paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Character analysis
        if text:
            total_chars = len(text)
            features['uppercase_count'] = sum(1 for c in text if c.isupper())
            features['lowercase_count'] = sum(1 for c in text if c.islower())
            features['digit_count'] = sum(1 for c in text if c.isdigit())
            features['space_count'] = sum(1 for c in text if c.isspace())
            features['punctuation_count'] = sum(1 for c in text if c in '.,!?;:\'"-')
            features['special_char_count'] = sum(1 for c in text if not c.isalnum() and not c.isspace())
            
            # Ratios
            features['uppercase_ratio'] = features['uppercase_count'] / total_chars
            features['lowercase_ratio'] = features['lowercase_count'] / total_chars
            features['digit_ratio'] = features['digit_count'] / total_chars
            features['space_ratio'] = features['space_count'] / total_chars
            features['punctuation_ratio'] = features['punctuation_count'] / total_chars
            features['special_char_ratio'] = features['special_char_count'] / total_chars
            
            # Average word length
            words = text.split()
            features['avg_word_length'] = sum(len(w) for w in words) / len(words) if words else 0
            
            # Unique words
            unique_words = set(w.lower() for w in words)
            features['unique_word_count'] = len(unique_words)
            features['unique_word_ratio'] = len(unique_words) / len(words) if words else 0
        else:
            # Default values for empty text
            zero_features = ['uppercase_count', 'lowercase_count', 'digit_count', 'space_count',
                           'punctuation_count', 'special_char_count', 'uppercase_ratio',
                           'lowercase_ratio', 'digit_ratio', 'space_ratio', 'punctuation_ratio',
                           'special_char_ratio', 'avg_word_length', 'unique_word_count',
                           'unique_word_ratio']
            for feat in zero_features:
                features[feat] = 0
        
        # URL extraction and analysis
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?'
        urls = re.findall(url_pattern, text)
        features['url_count'] = len(urls)
        features['unique_url_count'] = len(set(urls))
        
        # Detailed URL analysis
        url_features = self._analyze_urls(urls)
        features.update(url_features)
        
        # Email addresses in body
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        features['email_in_body_count'] = len(emails)
        features['unique_email_in_body_count'] = len(set(emails))
        
        # Phone numbers
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]\d{3}[-.\s]\d{4}'
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        features['phone_count'] = len(set(phones))
        
        # IP addresses
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, text)
        features['ip_address_count'] = len(ips)
        
        # Money symbols
        features['dollar_sign_count'] = text.count('$')
        features['euro_sign_count'] = text.count('€')
        features['pound_sign_count'] = text.count('£')
        features['yen_sign_count'] = text.count('¥')
        features['total_money_symbols'] = sum([text.count(s) for s in '$€£¥'])
        
        # Exclamation and question marks
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['exclamation_ratio'] = features['exclamation_count'] / total_chars if total_chars else 0
        features['question_ratio'] = features['question_count'] / total_chars if total_chars else 0
        
        # HTML detection
        features['has_html_tags'] = 1 if re.search(r'<[^>]+>', text) else 0
        features['html_tag_count'] = len(re.findall(r'<[^>]+>', text))
        
        # Common phishing keywords
        phishing_keywords = [
            'urgent', 'verify', 'account', 'bank', 'paypal', 'suspended', 
            'click', 'login', 'password', 'credit', 'social security', 'ssn',
            'limited', 'unusual', 'activity', 'verify', 'confirm', 'update',
            'security', 'fraud', 'claim', 'prize', 'winner', 'lottery',
            'inheritance', 'million', 'billion', 'dollars', 'transfer',
            'western union', 'money gram', 'wire transfer', 'bank account',
            'routing number', 'credit card', 'debit card', 'expire', 'deadline'
        ]
        
        keyword_counts = {}
        text_lower = text.lower()
        for keyword in phishing_keywords:
            count = text_lower.count(keyword)
            features[f'keyword_{keyword.replace(" ", "_")}'] = count
            keyword_counts[keyword] = count
        
        features['total_phishing_keywords'] = sum(keyword_counts.values())
        features['unique_phishing_keywords'] = sum(1 for v in keyword_counts.values() if v > 0)
        
        # Urgency indicators
        urgency_words = ['urgent', 'immediately', 'asap', 'deadline', 'expire', 'limited time']
        features['urgency_score'] = sum(text_lower.count(word) for word in urgency_words)
        
        # Fear indicators
        fear_words = ['suspended', 'terminated', 'closed', 'blocked', 'restricted', 'unauthorized']
        features['fear_score'] = sum(text_lower.count(word) for word in fear_words)
        
        # Curiosity indicators
        curiosity_words = ['winner', 'won', 'prize', 'selected', 'chosen', 'lucky']
        features['curiosity_score'] = sum(text_lower.count(word) for word in curiosity_words)
        
        # HTML decoding (check for encoded content)
        try:
            decoded = html.unescape(text)
            features['has_html_entities'] = 1 if decoded != text else 0
            features['html_entity_count'] = len(re.findall(r'&[#a-zA-Z0-9]+;', text))
        except:
            features['has_html_entities'] = 0
            features['html_entity_count'] = 0
        
        # Check for JavaScript
        features['has_javascript'] = 1 if 'javascript' in text_lower else 0
        features['has_onclick'] = 1 if 'onclick' in text_lower else 0
        features['has_onload'] = 1 if 'onload' in text_lower else 0
        features['has_form'] = 1 if '<form' in text_lower else 0
        features['has_input'] = 1 if '<input' in text_lower else 0
        
        return features
    
    def _analyze_urls(self, urls):
        """Perform detailed analysis of URLs"""
        features = {
            'url_avg_length': 0,
            'url_max_length': 0,
            'url_min_length': 0,
            'url_has_ip': 0,
            'url_has_port': 0,
            'url_has_https': 0,
            'url_has_http': 0,
            'url_count_https': 0,
            'url_count_http': 0,
            'url_suspicious_tlds': 0,
            'url_has_subdomains': 0,
            'url_max_dots': 0,
            'url_avg_slashes': 0,
            'url_has_at_symbol': 0,
            'url_has_double_slash': 0,
            'url_has_hyphen': 0,
            'url_has_underscore': 0,
            'url_has_percent_encoding': 0
        }
        
        if not urls:
            return features
        
        # Suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.club', 
                          '.online', '.site', '.work', '.date', '.men', '.loan']
        
        url_lengths = []
        for url in urls:
            url_lengths.append(len(url))
            
            # Parse URL
            try:
                parsed = urlparse(url)
                
                # Check for IP address in domain
                if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc):
                    features['url_has_ip'] += 1
                
                # Check for port
                if ':' in parsed.netloc:
                    features['url_has_port'] += 1
                
                # Protocol
                if parsed.scheme == 'https':
                    features['url_has_https'] += 1
                    features['url_count_https'] += 1
                elif parsed.scheme == 'http':
                    features['url_has_http'] += 1
                    features['url_count_http'] += 1
                
                # Check TLD
                for tld in suspicious_tlds:
                    if parsed.netloc.endswith(tld):
                        features['url_suspicious_tlds'] += 1
                
                # Count dots in domain
                dot_count = parsed.netloc.count('.')
                features['url_max_dots'] = max(features['url_max_dots'], dot_count)
                if dot_count > 2:
                    features['url_has_subdomains'] += 1
                
                # Count slashes in path
                slash_count = parsed.path.count('/')
                features['url_avg_slashes'] += slash_count
                
                # Check for suspicious characters
                if '@' in url:
                    features['url_has_at_symbol'] += 1
                if '//' in parsed.path:
                    features['url_has_double_slash'] += 1
                if '-' in parsed.netloc:
                    features['url_has_hyphen'] += 1
                if '_' in parsed.netloc:
                    features['url_has_underscore'] += 1
                if '%' in url:
                    features['url_has_percent_encoding'] += 1
                    
            except:
                continue
        
        # Calculate averages
        features['url_avg_length'] = sum(url_lengths) / len(url_lengths) if url_lengths else 0
        features['url_max_length'] = max(url_lengths) if url_lengths else 0
        features['url_min_length'] = min(url_lengths) if url_lengths else 0
        features['url_avg_slashes'] = features['url_avg_slashes'] / len(urls) if urls else 0
        
        # Convert counts to binary presence
        binary_features = ['url_has_ip', 'url_has_port', 'url_has_https', 'url_has_http',
                          'url_suspicious_tlds', 'url_has_subdomains', 'url_has_at_symbol',
                          'url_has_double_slash', 'url_has_hyphen', 'url_has_underscore',
                          'url_has_percent_encoding']
        
        for feat in binary_features:
            features[feat] = 1 if features[feat] > 0 else 0
        
        return features
    
    def extract_attachment_features(self, row):
        """Extract features related to attachments if available"""
        features = {}
        
        # Check for attachment-related columns
        attachment_cols = [col for col in self.df.columns if 'attach' in col.lower()]
        
        if attachment_cols:
            for col in attachment_cols:
                value = row.get(col, '')
                if pd.notna(value):
                    value_str = str(value)
                    
                    # Count attachments
                    if 'attachment_count' not in features:
                        features['attachment_count'] = 0
                    
                    # Try to parse attachment count
                    try:
                        count = int(value_str)
                        features['attachment_count'] += count
                    except:
                        features['attachment_count'] += 1
                    
                    # Check attachment types
                    if '.exe' in value_str.lower() or '.bat' in value_str.lower():
                        features['has_executable_attachment'] = 1
                    if '.zip' in value_str.lower() or '.rar' in value_str.lower():
                        features['has_archive_attachment'] = 1
                    if '.pdf' in value_str.lower():
                        features['has_pdf_attachment'] = 1
                    if '.doc' in value_str.lower() or '.xls' in value_str.lower():
                        features['has_document_attachment'] = 1
                    if '.jpg' in value_str.lower() or '.png' in value_str.lower():
                        features['has_image_attachment'] = 1
        
        # Set defaults
        default_attachment_features = {
            'attachment_count': 0,
            'has_executable_attachment': 0,
            'has_archive_attachment': 0,
            'has_pdf_attachment': 0,
            'has_document_attachment': 0,
            'has_image_attachment': 0
        }
        
        for key, default in default_attachment_features.items():
            if key not in features:
                features[key] = default
        
        return features
    
    def extract_metadata_features(self, row):
        """Extract metadata features"""
        features = {
            'row_index': row.name if hasattr(row, 'name') else 0,
            'data_source': self.csv_file
        }
        
        # Check for ID columns
        id_cols = [col for col in self.df.columns if 'id' in col.lower()]
        for col in id_cols:
            features[f'has_{col}'] = 1 if pd.notna(row.get(col, '')) else 0
        
        return features
    
    def extract_all_features(self):
        """Extract all features from the dataset"""
        print("\n" + "="*60)
        print("🔧 EXTRACTING FEATURES")
        print("="*60)
        
        if self.df is None:
            self.load_dataset()
        
        all_features = []
        feature_names = set()
        
        # Determine which column contains the email text
        text_candidates = ['body', 'text', 'content', 'message', 'email', 'mail']
        text_column = None
        for col in self.df.columns:
            if col.lower() in text_candidates:
                text_column = col
                break
        
        if text_column is None:
            # Use the first string column as text
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    text_column = col
                    break
        
        print(f"📝 Using '{text_column}' as text column")
        
        # Process each row
        total_rows = len(self.df)
        for idx, row in self.df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processing row {idx}/{total_rows}...")
            
            features = {}
            
            # Extract metadata features
            metadata_features = self.extract_metadata_features(row)
            features.update(metadata_features)
            
            # Extract header features
            header_features = self.extract_header_features(row)
            features.update(header_features)
            
            # Extract body features
            text_content = row.get(text_column, '') if text_column else ''
            body_features = self.extract_body_features(text_content)
            features.update(body_features)
            
            # Extract attachment features
            attachment_features = self.extract_attachment_features(row)
            features.update(attachment_features)
            
            # Add label if available
            label_cols = ['label', 'class', 'type', 'category', 'spam', 'phishing']
            for col in label_cols:
                if col in self.df.columns and pd.notna(row.get(col)):
                    features['label'] = row[col]
                    features['label_source'] = col
                    break
            
            all_features.append(features)
            
            # Update feature names
            feature_names.update(features.keys())
        
        # Create DataFrame
        self.features_df = pd.DataFrame(all_features)
        
        # Reorder columns to put metadata first
        meta_cols = [col for col in self.features_df.columns if col in ['row_index', 'data_source', 'label', 'label_source']]
        other_cols = [col for col in self.features_df.columns if col not in meta_cols]
        self.features_df = self.features_df[meta_cols + other_cols]
        
        print(f"\n✅ Extracted {len(feature_names)} features from {total_rows} emails")
        
        return self.features_df
    
    def save_features(self, output_file='extracted_features.csv'):
        """Save extracted features to CSV"""
        if self.features_df is not None:
            self.features_df.to_csv(output_file, index=False)
            print(f"💾 Features saved to {output_file}")
    
    def save_feature_descriptions(self, output_file='feature_descriptions.json'):
        """Save feature descriptions to JSON"""
        descriptions = {
            'row_index': 'Index of the row in original dataset',
            'data_source': 'Source CSV file name',
            'label': 'Email label (spam/phishing or legitimate) if available',
            'label_source': 'Column name where label was found',
            
            # Header features
            'has_from': 'Whether From header exists',
            'has_to': 'Whether To header exists',
            'has_cc': 'Whether CC header exists',
            'has_subject': 'Whether Subject header exists',
            'from_length': 'Length of From field',
            'from_domain': 'Domain extracted from From email',
            'recipient_count': 'Number of recipients in To field',
            'subject_length': 'Length of subject line',
            'subject_word_count': 'Number of words in subject',
            'subject_has_reply': 'Whether subject starts with Re:',
            'subject_has_fwd': 'Whether subject starts with Fwd:',
            'subject_has_urgent': 'Whether subject contains "urgent"',
            'subject_has_verify': 'Whether subject contains "verify"',
            'subject_all_caps': 'Whether subject is all uppercase',
            'subject_caps_ratio': 'Ratio of uppercase letters in subject',
            'subject_has_numbers': 'Whether subject contains numbers',
            'subject_has_special': 'Whether subject contains special characters',
            'subject_exclamation': 'Number of exclamation marks in subject',
            'subject_question': 'Number of question marks in subject',
            'subject_money': 'Whether subject contains money symbols',
            'hour_sent': 'Hour when email was sent (-1 if unknown)',
            
            # Body features
            'body_length': 'Total length of email body in characters',
            'body_word_count': 'Number of words in email body',
            'body_line_count': 'Number of lines in email body',
            'body_paragraph_count': 'Number of paragraphs',
            'uppercase_count': 'Count of uppercase letters',
            'lowercase_count': 'Count of lowercase letters',
            'digit_count': 'Count of digits',
            'space_count': 'Count of spaces',
            'punctuation_count': 'Count of punctuation marks',
            'special_char_count': 'Count of special characters',
            'uppercase_ratio': 'Ratio of uppercase letters',
            'lowercase_ratio': 'Ratio of lowercase letters',
            'digit_ratio': 'Ratio of digits',
            'space_ratio': 'Ratio of spaces',
            'punctuation_ratio': 'Ratio of punctuation',
            'special_char_ratio': 'Ratio of special characters',
            'avg_word_length': 'Average word length',
            'unique_word_count': 'Number of unique words',
            'unique_word_ratio': 'Ratio of unique words',
            
            # URL features
            'url_count': 'Number of URLs in body',
            'unique_url_count': 'Number of unique URLs',
            'url_avg_length': 'Average URL length',
            'url_max_length': 'Maximum URL length',
            'url_min_length': 'Minimum URL length',
            'url_has_ip': 'Whether any URL uses IP address instead of domain',
            'url_has_port': 'Whether any URL specifies a port',
            'url_has_https': 'Whether any URL uses HTTPS',
            'url_has_http': 'Whether any URL uses HTTP',
            'url_count_https': 'Number of HTTPS URLs',
            'url_count_http': 'Number of HTTP URLs',
            'url_suspicious_tlds': 'Whether any URL uses suspicious TLD',
            'url_has_subdomains': 'Whether any URL has multiple subdomains',
            'url_max_dots': 'Maximum number of dots in URL domain',
            'url_avg_slashes': 'Average number of slashes in URL path',
            'url_has_at_symbol': 'Whether any URL contains @ symbol',
            'url_has_double_slash': 'Whether any URL contains // in path',
            'url_has_hyphen': 'Whether any URL contains hyphen in domain',
            'url_has_underscore': 'Whether any URL contains underscore',
            'url_has_percent_encoding': 'Whether any URL contains % encoding',
            
            # Contact features
            'email_in_body_count': 'Number of email addresses in body',
            'unique_email_in_body_count': 'Number of unique email addresses',
            'phone_count': 'Number of phone numbers',
            'ip_address_count': 'Number of IP addresses',
            
            # Money features
            'dollar_sign_count': 'Number of $ symbols',
            'euro_sign_count': 'Number of € symbols',
            'pound_sign_count': 'Number of £ symbols',
            'yen_sign_count': 'Number of ¥ symbols',
            'total_money_symbols': 'Total money symbols',
            
            # Emphasis features
            'exclamation_count': 'Number of ! marks',
            'question_count': 'Number of ? marks',
            'exclamation_ratio': 'Ratio of ! marks',
            'question_ratio': 'Ratio of ? marks',
            
            # HTML features
            'has_html_tags': 'Whether HTML tags are present',
            'html_tag_count': 'Number of HTML tags',
            'has_html_entities': 'Whether HTML entities are present',
            'html_entity_count': 'Number of HTML entities',
            'has_javascript': 'Whether JavaScript is present',
            'has_onclick': 'Whether onclick attribute is present',
            'has_onload': 'Whether onload attribute is present',
            'has_form': 'Whether form tags are present',
            'has_input': 'Whether input tags are present',
            
            # Phishing indicators
            'total_phishing_keywords': 'Total count of phishing-related keywords',
            'unique_phishing_keywords': 'Number of unique phishing keywords',
            'urgency_score': 'Count of urgency-related words',
            'fear_score': 'Count of fear-inducing words',
            'curiosity_score': 'Count of curiosity-triggering words',
            
            # Attachment features
            'attachment_count': 'Number of attachments',
            'has_executable_attachment': 'Whether executable files are attached',
            'has_archive_attachment': 'Whether archive files are attached',
            'has_pdf_attachment': 'Whether PDF files are attached',
            'has_document_attachment': 'Whether document files are attached',
            'has_image_attachment': 'Whether image files are attached'
        }
        
        # Add all keyword features
        phishing_keywords = [
            'urgent', 'verify', 'account', 'bank', 'paypal', 'suspended', 
            'click', 'login', 'password', 'credit', 'social_security',
            'limited', 'unusual', 'activity', 'confirm', 'update',
            'security', 'fraud', 'claim', 'prize', 'winner', 'lottery',
            'inheritance', 'million', 'billion', 'dollars', 'transfer',
            'western_union', 'money_gram', 'wire_transfer', 'bank_account',
            'routing_number', 'credit_card', 'debit_card', 'expire', 'deadline'
        ]
        
        for keyword in phishing_keywords:
            descriptions[f'keyword_{keyword}'] = f'Count of "{keyword}" in email'
        
        with open(output_file, 'w') as f:
            json.dump(descriptions, f, indent=2)
        
        print(f"📝 Feature descriptions saved to {output_file}")
    
    def print_feature_summary(self):
        """Print a summary of extracted features"""
        if self.features_df is None:
            print("No features extracted yet")
            return
        
        print("\n" + "="*60)
        print("📊 FEATURE EXTRACTION SUMMARY")
        print("="*60)
        
        print(f"\n📋 Total features extracted: {len(self.features_df.columns)}")
        print(f"📋 Total emails processed: {len(self.features_df)}")
        
        # Feature categories
        feature_categories = {
            'Metadata': ['row_index', 'data_source', 'label', 'label_source'],
            'Headers': ['has_from', 'has_to', 'has_cc', 'has_subject', 'from_length', 
                       'recipient_count', 'subject_length', 'subject_word_count'],
            'URLs': ['url_count', 'unique_url_count', 'url_avg_length', 'url_has_ip'],
            'Text Statistics': ['body_length', 'body_word_count', 'uppercase_ratio', 
                               'digit_ratio', 'avg_word_length'],
            'Phishing Indicators': ['total_phishing_keywords', 'urgency_score', 
                                   'fear_score', 'curiosity_score'],
            'Attachments': ['attachment_count', 'has_executable_attachment']
        }
        
        print("\n📁 Feature Categories:")
        for category, features in feature_categories.items():
            present_features = [f for f in features if f in self.features_df.columns]
            print(f"  {category}: {len(present_features)} features")
        
        # Sample statistics
        print("\n📈 Sample Statistics (first 5 emails):")
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns[:10]
        print(self.features_df[numeric_cols].head().to_string())
        
        # Check for label column
        if 'label' in self.features_df.columns:
            print(f"\n🏷️  Label distribution:")
            print(self.features_df['label'].value_counts())
    
    def run_pipeline(self, output_prefix='phishing_dataset'):
        """Run the complete feature extraction pipeline"""
        
        # Step 1: Load dataset
        self.load_dataset()
        
        # Step 2: Explore dataset
        self.explore_dataset()
        
        # Step 3: Extract all features
        self.extract_all_features()
        
        # Step 4: Save features
        self.save_features(f'{output_prefix}_features.csv')
        
        # Step 5: Save feature descriptions
        self.save_feature_descriptions(f'{output_prefix}_feature_descriptions.json')
        
        # Step 6: Print summary
        self.print_feature_summary()
        
        print("\n" + "="*60)
        print("✅ FEATURE EXTRACTION COMPLETE")
        print("="*60)
        
        return self.features_df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from phishing email dataset')
    parser.add_argument('csv_file', help='Path to the CSV file containing emails')
    parser.add_argument('--output', '-o', default='phishing_features',
                       help='Output prefix for feature files (default: phishing_features)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: File {args.csv_file} not found!")
        return
    
    # Initialize extractor
    extractor = PhishingEmailFeatureExtractor(args.csv_file)
    
    # Run pipeline
    features_df = extractor.run_pipeline(output_prefix=args.output)
    
    print(f"\n📁 Output files:")
    print(f"  - {args.output}_features.csv")
    print(f"  - {args.output}_feature_descriptions.json")


if __name__ == "__main__":
    main()

# Add global function for easy access
_extractor_instance = None

def get_extractor(csv_file=None):
    """Get or create a feature extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        if csv_file and os.path.exists(csv_file):
            _extractor_instance = PhishingEmailFeatureExtractor(csv_file)
        else:
            # Use default CSV files
            default_csv = "/home/rama/MGW/Parsing/enron_features_features.csv"
            if os.path.exists(default_csv):
                _extractor_instance = PhishingEmailFeatureExtractor(default_csv)
            else:
                # Create a minimal extractor without CSV
                _extractor_instance = PhishingEmailFeatureExtractor(None)
    return _extractor_instance

# Also add a simplified feature extraction function for real-time analysis
def extract_features_from_email(message):
    """
    Extract features from an email message object in real-time.
    This is for live email analysis.
    """
    from email.utils import parseaddr, parsedate_to_datetime
    from datetime import datetime
    import re
    
    features = {}
    
    # Extract From header
    from_addr = message.get('From', '')
    from_clean = parseaddr(from_addr)[1] or from_addr
    from_domain = from_clean.split('@')[-1] if '@' in from_clean else ''
    
    features['from_domain_length'] = len(from_domain)
    features['from_localpart_length'] = len(from_clean.split('@')[0]) if '@' in from_clean else 0
    features['has_numeric_in_domain'] = 1 if any(c.isdigit() for c in from_domain) else 0
    
    # Reply-To
    reply_to = message.get('Reply-To', '')
    features['has_reply_to'] = 1 if reply_to else 0
    if reply_to:
        reply_clean = parseaddr(reply_to)[1] or reply_to
        reply_domain = reply_clean.split('@')[-1] if '@' in reply_clean else ''
        features['reply_to_domain_matches_from'] = 1 if reply_domain == from_domain else 0
    else:
        features['reply_to_domain_matches_from'] = 0
    
    # Return-Path
    return_path = message.get('Return-Path', '')
    features['has_return_path'] = 1 if return_path else 0
    if return_path:
        return_clean = return_path.strip('<>')
        return_domain = return_clean.split('@')[-1] if '@' in return_clean else ''
        features['return_path_matches_from'] = 1 if return_domain == from_domain else 0
    else:
        features['return_path_matches_from'] = 0
    
    # Authentication
    auth_results = message.get('Authentication-Results', '')
    auth_lower = auth_results.lower()
    features['has_auth_results'] = 1 if auth_results else 0
    features['spf_fail'] = 1 if 'spf=fail' in auth_lower else 0
    features['dkim_fail'] = 1 if 'dkim=fail' in auth_lower else 0
    features['dmarc_fail'] = 1 if 'dmarc=fail' in auth_lower else 0
    features['has_dkim'] = 1 if message.get('DKIM-Signature') else 0
    
    # Subject
    subject = message.get('Subject', '')
    features['subject_length'] = len(subject)
    features['subject_has_urgent'] = 1 if re.search(r'urgent|immediate|asap', subject, re.I) else 0
    features['subject_has_verify'] = 1 if re.search(r'verify|confirm|validate', subject, re.I) else 0
    features['subject_has_account'] = 1 if re.search(r'account|password|credential', subject, re.I) else 0
    features['subject_has_suspended'] = 1 if re.search(r'suspended|locked|closed', subject, re.I) else 0
    features['subject_caps_ratio'] = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
    
    # Received headers
    received_headers = message.get_all('Received', [])
    features['received_count_normalized'] = min(len(received_headers) / 10.0, 1.0)
    
    # Date
    date_str = message.get('Date', '')
    features['has_date'] = 1 if date_str else 0
    if date_str:
        try:
            date_obj = parsedate_to_datetime(date_str)
            features['date_is_weekend'] = 1 if date_obj.weekday() >= 5 else 0
            features['date_is_future'] = 1 if date_obj > datetime.now() else 0
        except:
            features['date_is_weekend'] = 0
            features['date_is_future'] = 0
    else:
        features['date_is_weekend'] = 0
        features['date_is_future'] = 0
    
    # Message-ID
    msg_id = message.get('Message-ID', '')
    features['has_message_id'] = 1 if msg_id else 0
    
    # Domain reputation
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']
    features['has_suspicious_tld'] = 1 if any(from_domain.endswith(tld) for tld in suspicious_tlds) else 0
    
    free_emails = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
    features['is_free_email'] = 1 if from_domain in free_emails else 0
    
    # Extract body features
    body_text = ""
    try:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_text = payload.decode('utf-8', errors='ignore')
                        break
        else:
            payload = message.get_payload(decode=True)
            if payload:
                body_text = payload.decode('utf-8', errors='ignore')
    except:
        pass
    
    body_lower = body_text.lower()
    features['body_length'] = len(body_text)
    features['word_count'] = len(body_text.split())
    
    # URL features
    urls = re.findall(r'https?://[^\s]+', body_lower)
    features['url_count'] = len(urls)
    features['has_url'] = 1 if urls else 0
    
    # Phishing keywords
    phishing_keywords = ['verify', 'confirm', 'update', 'validate', 'secure', 'suspended']
    features['phishing_keyword_count'] = sum(1 for kw in phishing_keywords if kw in body_lower)
    
    # Urgency keywords
    urgency_keywords = ['urgent', 'immediately', 'asap', 'critical']
    features['urgency_keyword_count'] = sum(1 for kw in urgency_keywords if kw in body_lower)
    
    # Sensitive requests
    sensitive_keywords = ['password', 'credit card', 'ssn', 'bank account']
    features['sensitive_request_count'] = sum(1 for kw in sensitive_keywords if kw in body_lower)
    
    # Manipulation phrases
    manipulation_phrases = ['account will be closed', 'legal action', 'immediate action required']
    features['manipulation_count'] = sum(1 for phrase in manipulation_phrases if phrase in body_lower)
    
    # Stylometric features
    features['caps_ratio'] = sum(1 for c in body_text if c.isupper()) / max(len(body_text), 1)
    features['exclamation_count'] = body_text.count('!')
    features['question_count'] = body_text.count('?')
    
    return features

