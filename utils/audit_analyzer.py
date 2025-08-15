import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import streamlit as st

class AuditAnalyzer:
    """Analyze transactions for audit red flags and suspicious patterns"""
    
    def __init__(self):
        self.setup_audit_patterns()
    
    def setup_audit_patterns(self):
        """Define audit red flag patterns and keywords based on ISA audit standards"""
        self.suspicious_keywords = {
            # Vague or Generic Descriptions
            'vague_generic': [
                'misc', 'miscellaneous', 'general', 'other', 'adjustment', 
                'correction', 'reversal', 'unknown', 'n/a', 'various', 'sundry'
            ],
            
            # Unusual Keywords
            'unusual': [
                'gift', 'donation', 'charity', 'contribution', 'political', 'sponsorship'
            ],
            
            # Personal/Non-Business Terms
            'personal': [
                'personal', 'private', 'loan to staff', 'advance to employee', 
                'salary adjustment', 'employee loan', 'staff advance'
            ],
            
            # Keywords Indicating Irregularities
            'irregularities': [
                'write-off', 'loss', 'fraud', 'error', 'cash short', 'overpayment', 
                'underpayment', 'shortage', 'variance', 'discrepancy'
            ],
            
            # Related Party Transactions
            'related_party': [
                'director', 'shareholder', 'owner', 'affiliate', 'subsidiary', 
                'intercompany', 'related party', 'management'
            ],
            
            # Suspicious Vendor/Payee Names
            'suspicious_vendor': [
                'cash', 'supplier', 'vendor', 'unknown vendor', 'generic supplier',
                'misc vendor', 'various suppliers'
            ],
            
            # High-Risk Terms
            'high_risk': [
                'off-book', 'unrecorded', 'adjustment entry', 'manual entry', 
                'correction', 'suspense', 'clearing', 'temporary'
            ],
            
            # Incomplete descriptions
            'incomplete': [
                'tbd', 'tbc', 'pending', 'temp', 'temporary', 'hold', 'n/a'
            ]
        }
        
        self.high_risk_patterns = [
            r'\b(?:cash|petty)\s+cash\b',  # Petty cash
            r'\bmiscellaneous\s+expense\b',  # Miscellaneous expense
            r'\bsuspense\s+account\b',  # Suspense account
            r'\badjustment\s+entry\b',  # Adjustment entry
            r'\berror\s+correction\b',  # Error correction
            r'\btemporary\s+entry\b',  # Temporary entry
        ]
        
        self.round_number_patterns = [
            r'\b\d+000(?:\.00)?\b',  # Ends in 000
            r'\b\d+00(?:\.00)?\b',   # Ends in 00
            r'\b\d+\.00\b'           # Exact dollar amounts
        ]
    
    def analyze_descriptions(self, df, description_column):
        """Analyze transaction descriptions for audit red flags"""
        if description_column not in df.columns:
            return df
        
        df = df.copy()
        
        # Initialize risk scoring columns
        df['audit_risk_score'] = 0.0
        df['risk_factors'] = ''
        df['selection_reason'] = ''
        
        # Analyze each description
        for idx, row in df.iterrows():
            description = str(row[description_column]).lower() if pd.notna(row[description_column]) else ''
            risk_score = 0.0
            risk_factors = []
            
            # Check for empty or very short descriptions
            if len(description.strip()) <= 3:
                risk_score += 0.4
                risk_factors.append('Empty/minimal description')
            
            # Check for suspicious keywords by category
            for category, keywords in self.suspicious_keywords.items():
                for keyword in keywords:
                    if keyword in description:
                        if category == 'high_risk':
                            risk_score += 0.4
                            risk_factors.append(f'High-risk term: {keyword}')
                        elif category == 'irregularities':
                            risk_score += 0.35
                            risk_factors.append(f'Irregularity indicator: {keyword}')
                        elif category == 'related_party':
                            risk_score += 0.3
                            risk_factors.append(f'Related party indicator: {keyword}')
                        elif category == 'personal':
                            risk_score += 0.3
                            risk_factors.append(f'Personal/non-business term: {keyword}')
                        elif category == 'vague_generic':
                            risk_score += 0.25
                            risk_factors.append(f'Vague/generic description: {keyword}')
                        elif category == 'unusual':
                            risk_score += 0.2
                            risk_factors.append(f'Unusual keyword: {keyword}')
                        elif category == 'suspicious_vendor':
                            risk_score += 0.2
                            risk_factors.append(f'Suspicious vendor name: {keyword}')
                        elif category == 'incomplete':
                            risk_score += 0.15
                            risk_factors.append(f'Incomplete description: {keyword}')
                        break  # Only count each category once per description
            
            # Check for high-risk patterns
            for pattern in self.high_risk_patterns:
                if re.search(pattern, description):
                    risk_score += 0.3
                    risk_factors.append('High-risk pattern detected')
            
            # Check for very generic descriptions
            if self.is_generic_description(description):
                risk_score += 0.25
                risk_factors.append('Generic description')
            
            # Check for repeated characters (potential data entry errors)
            if self.has_repeated_characters(description):
                risk_score += 0.15
                risk_factors.append('Repeated characters')
            
            # Cap the risk score at 1.0
            risk_score = min(risk_score, 1.0)
            
            df.at[idx, 'audit_risk_score'] = risk_score
            df.at[idx, 'risk_factors'] = '; '.join(risk_factors)
            df.at[idx, 'selection_reason'] = self.generate_selection_reason(risk_factors, risk_score)
        
        return df
    
    def is_generic_description(self, description):
        """Check if description is too generic"""
        generic_patterns = [
            r'^\s*payment\s*$',
            r'^\s*transfer\s*$',
            r'^\s*expense\s*$',
            r'^\s*entry\s*$',
            r'^\s*transaction\s*$',
            r'^\s*item\s*\d*\s*$'
        ]
        
        for pattern in generic_patterns:
            if re.match(pattern, description):
                return True
        
        return False
    
    def has_repeated_characters(self, description):
        """Check for suspicious repeated characters"""
        # Look for 3 or more repeated characters
        pattern = r'(.)\1{2,}'
        return bool(re.search(pattern, description))
    
    def analyze_amounts(self, df, amount_column):
        """Analyze transaction amounts for suspicious patterns"""
        if amount_column not in df.columns:
            return df
        
        df = df.copy()
        
        # Convert amounts to numeric
        amounts = pd.to_numeric(df[amount_column], errors='coerce')
        
        # Initialize amount risk factors
        if 'amount_risk_factors' not in df.columns:
            df['amount_risk_factors'] = ''
        
        for idx, amount in amounts.items():
            if pd.isna(amount):
                continue
                
            amount_risks = []
            
            # Check for round numbers
            if self.is_round_number(amount):
                amount_risks.append('Round number')
            
            # Check for unusual amounts (statistical outliers)
            if self.is_statistical_outlier(amount, amounts):
                amount_risks.append('Statistical outlier')
            
            # Check for amounts just below thresholds
            if self.is_below_threshold(amount):
                amount_risks.append('Just below threshold')
            
            df.at[idx, 'amount_risk_factors'] = '; '.join(amount_risks)
        
        return df
    
    def is_round_number(self, amount):
        """Check if amount is suspiciously round"""
        amount_str = str(abs(amount))
        
        # Check various round number patterns
        for pattern in self.round_number_patterns:
            if re.search(pattern, amount_str):
                return True
        
        return False
    
    def is_statistical_outlier(self, amount, amounts_series):
        """Check if amount is a statistical outlier"""
        try:
            q1 = amounts_series.quantile(0.25)
            q3 = amounts_series.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            return amount < lower_bound or amount > upper_bound
        except:
            return False
    
    def is_below_threshold(self, amount):
        """Check if amount is just below common thresholds"""
        common_thresholds = [5000, 10000, 25000, 50000, 100000]
        tolerance = 100  # Within $100 of threshold
        
        for threshold in common_thresholds:
            if threshold - tolerance <= amount < threshold:
                return True
        
        return False
    
    def find_duplicates(self, df, columns_to_check):
        """Find potential duplicate transactions"""
        if not columns_to_check:
            return df
        
        df = df.copy()
        
        # Check for exact duplicates
        available_columns = [col for col in columns_to_check if col in df.columns]
        
        if available_columns:
            df['is_duplicate'] = df.duplicated(subset=available_columns, keep=False)
            
            # Mark duplicate groups
            duplicate_groups = df[df['is_duplicate']].groupby(available_columns).ngroup()
            df.loc[df['is_duplicate'], 'duplicate_group'] = duplicate_groups + 1
        
        return df
    
    def analyze_timing_patterns(self, df, date_column):
        """Analyze transaction timing for suspicious patterns"""
        if date_column not in df.columns:
            return df
        
        df = df.copy()
        
        # Convert to datetime
        dates = pd.to_datetime(df[date_column], errors='coerce')
        
        # Initialize timing risk factors
        df['timing_risk_factors'] = ''
        
        for idx, date in dates.items():
            if pd.isna(date):
                continue
            
            timing_risks = []
            
            # Check for weekend transactions
            if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                timing_risks.append('Weekend transaction')
            
            # Check for end-of-month transactions
            if date.day >= 28:
                timing_risks.append('End-of-month transaction')
            
            # Check for end-of-year transactions
            if date.month == 12 and date.day >= 20:
                timing_risks.append('End-of-year transaction')
            
            # Check for after-hours (assuming business hours 9-5)
            if hasattr(date, 'hour'):
                if date.hour < 9 or date.hour > 17:
                    timing_risks.append('After-hours transaction')
            
            df.at[idx, 'timing_risk_factors'] = '; '.join(timing_risks)
        
        return df
    
    def generate_selection_reason(self, risk_factors, risk_score):
        """Generate human-readable selection reasoning"""
        if not risk_factors:
            return "Selected based on criteria match"
        
        if risk_score >= 0.7:
            risk_level = "HIGH RISK"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"
        
        reason = f"{risk_level}: {'; '.join(risk_factors[:3])}"  # Show top 3 factors
        
        if len(risk_factors) > 3:
            reason += f" (and {len(risk_factors) - 3} more factors)"
        
        return reason
