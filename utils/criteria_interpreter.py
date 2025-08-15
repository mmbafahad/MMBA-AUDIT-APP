import pandas as pd
import numpy as np
import re
from utils.nlp_processor import NLPProcessor
from utils.audit_analyzer import AuditAnalyzer
import streamlit as st

class CriteriaInterpreter:
    """Interpret natural language criteria and apply them to transaction data"""
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.audit_analyzer = AuditAnalyzer()
    
    def parse_criteria(self, criteria_text, column_mapping):
        """Parse natural language criteria into structured format"""
        parsed = {
            'sample_size': None,
            'criteria_types': [],
            'amount_filters': [],
            'description_filters': [],
            'date_filters': [],
            'duplicate_check': False,
            'risk_analysis': False,
            'sort_by': None,
            'sort_order': 'desc',
            'text_patterns': [],
            'transaction_type_filter': None,
            'raw_text': criteria_text
        }
        
        # Extract keywords and patterns
        keywords = self.nlp_processor.extract_keywords(criteria_text)
        numbers = self.nlp_processor.extract_numbers(criteria_text)
        criteria_types = self.nlp_processor.identify_criteria_type(criteria_text, keywords)
        
        # Extract sample size
        parsed['sample_size'] = self.nlp_processor.extract_sample_size(criteria_text)
        
        # Set criteria types
        parsed['criteria_types'] = criteria_types
        
        # Parse amount-related criteria
        if 'amount' in criteria_types:
            parsed['amount_filters'] = self.parse_amount_criteria(criteria_text, keywords, numbers)
            
            # Check for credit/debit specific requests
            if keywords['credit_specific']:
                parsed['transaction_type_filter'] = 'credit'
                parsed['sort_by'] = 'credit_amount_only'
            elif keywords['debit_specific']:
                parsed['transaction_type_filter'] = 'debit'
                parsed['sort_by'] = 'debit_amount_only'
            else:
                parsed['sort_by'] = 'amount'
            
            # Determine sorting order for amount-based selection
            if any(word in criteria_text.lower() for word in ['highest', 'top', 'maximum', 'largest']):
                parsed['sort_order'] = 'desc'
            elif any(word in criteria_text.lower() for word in ['lowest', 'bottom', 'minimum', 'smallest']):
                parsed['sort_order'] = 'asc'
        
        # Parse description-related criteria
        if 'description' in criteria_types or parsed['text_patterns']:
            parsed['description_filters'] = self.parse_description_criteria(criteria_text, keywords)
            parsed['risk_analysis'] = True  # Always enable risk analysis for description-based criteria
        
        # Parse duplicate criteria
        if 'duplicates' in criteria_types:
            parsed['duplicate_check'] = True
        
        # Parse date criteria
        if 'date' in criteria_types:
            parsed['date_filters'] = self.parse_date_criteria(criteria_text, keywords)
        
        # Extract text patterns
        parsed['text_patterns'] = self.nlp_processor.extract_text_patterns(criteria_text)
        
        # Enable risk analysis if suspicious patterns mentioned
        if keywords['suspicious'] or any(word in criteria_text.lower() for word in 
                                       ['suspicious', 'risk', 'flag', 'audit', 'review']):
            parsed['risk_analysis'] = True
        
        return parsed
    
    def parse_amount_criteria(self, text, keywords, numbers):
        """Parse amount-specific criteria"""
        filters = []
        
        # Extract comparison operators
        comparisons = self.nlp_processor.parse_comparison_operators(text)
        filters.extend(comparisons)
        
        # Handle relative amount criteria
        if keywords['amounts']:
            if numbers:
                # Use first number found for threshold
                threshold = numbers[0]
                
                if any(word in text.lower() for word in ['above', 'over', 'greater', 'more than', '>']):
                    filters.append({'operator': 'greater_than', 'value': threshold})
                elif any(word in text.lower() for word in ['below', 'under', 'less', 'fewer than', '<']):
                    filters.append({'operator': 'less_than', 'value': threshold})
        
        # Handle round number criteria
        if any(word in text.lower() for word in ['round', 'ending in']):
            filters.append({'operator': 'round_numbers', 'value': None})
        
        return filters
    
    def parse_description_criteria(self, text, keywords):
        """Parse description-specific criteria"""
        filters = []
        
        # Empty/missing description criteria
        if keywords['empty'] or any(word in text.lower() for word in ['no description', 'blank', 'empty']):
            filters.append({'type': 'empty_description'})
        
        # Suspicious description criteria
        if keywords['suspicious']:
            filters.append({'type': 'suspicious_description'})
        
        # Vague description criteria
        if any(word in text.lower() for word in ['vague', 'unclear', 'generic']):
            filters.append({'type': 'vague_description'})
        
        return filters
    
    def parse_date_criteria(self, text, keywords):
        """Parse date-specific criteria"""
        filters = []
        
        # Weekend criteria
        if any(word in text.lower() for word in ['weekend', 'saturday', 'sunday']):
            filters.append({'type': 'weekend'})
        
        # End of period criteria
        if any(word in text.lower() for word in ['end of month', 'month end', 'year end']):
            filters.append({'type': 'end_of_period'})
        
        return filters
    
    def apply_criteria(self, df, parsed_criteria, column_mapping):
        """Apply parsed criteria to the dataframe"""
        result_df = df.copy()
        
        # First, run audit analysis if requested
        if parsed_criteria['risk_analysis']:
            if 'description' in column_mapping:
                result_df = self.audit_analyzer.analyze_descriptions(
                    result_df, column_mapping['description']
                )
            
            if 'amount' in column_mapping:
                result_df = self.audit_analyzer.analyze_amounts(
                    result_df, column_mapping['amount']
                )
            
            if 'date' in column_mapping:
                result_df = self.audit_analyzer.analyze_timing_patterns(
                    result_df, column_mapping['date']
                )
        
        # Apply amount filters
        if parsed_criteria['amount_filters'] and 'amount' in column_mapping:
            result_df = self.apply_amount_filters(
                result_df, parsed_criteria['amount_filters'], column_mapping['amount']
            )
        
        # Apply description filters
        if parsed_criteria['description_filters'] and 'description' in column_mapping:
            result_df = self.apply_description_filters(
                result_df, parsed_criteria['description_filters'], column_mapping['description']
            )
        
        # Apply date filters
        if parsed_criteria['date_filters'] and 'date' in column_mapping:
            result_df = self.apply_date_filters(
                result_df, parsed_criteria['date_filters'], column_mapping['date']
            )
        
        # Check for duplicates
        if parsed_criteria['duplicate_check']:
            duplicate_columns = [col for col in column_mapping.values() if col in result_df.columns]
            result_df = self.audit_analyzer.find_duplicates(result_df, duplicate_columns)
            result_df = result_df[result_df.get('is_duplicate', False)]
        
        # Apply text pattern matching
        if parsed_criteria['text_patterns'] and 'description' in column_mapping:
            result_df = self.apply_text_patterns(
                result_df, parsed_criteria['text_patterns'], column_mapping['description']
            )
        
        # Apply transaction type filter (credit/debit specific)
        if parsed_criteria.get('transaction_type_filter'):
            result_df = self.apply_transaction_type_filter(
                result_df, parsed_criteria['transaction_type_filter']
            )
        
        # Sort results if specified
        if parsed_criteria['sort_by'] and len(result_df) > 0:
            # Handle special sorting columns for credit/debit specific queries
            if parsed_criteria['sort_by'] == 'credit_amount_only':
                sort_column = 'credit_amount_only'
            elif parsed_criteria['sort_by'] == 'debit_amount_only':
                sort_column = 'debit_amount_only'
            elif parsed_criteria['sort_by'] in column_mapping:
                sort_column = column_mapping[parsed_criteria['sort_by']]
            else:
                sort_column = parsed_criteria['sort_by']
            
            if sort_column in result_df.columns:
                ascending = parsed_criteria['sort_order'] == 'asc'
                result_df = result_df.sort_values(sort_column, ascending=ascending)
                
                # Add selection reason for sorted results
                if 'selection_reason' not in result_df.columns:
                    result_df['selection_reason'] = ''
                order_desc = 'highest' if not ascending else 'lowest'
                if parsed_criteria['sort_by'] == 'credit_amount_only':
                    result_df['selection_reason'] = result_df['selection_reason'].astype(str) + f'; Selected based on {order_desc} credit amounts'
                elif parsed_criteria['sort_by'] == 'debit_amount_only':
                    result_df['selection_reason'] = result_df['selection_reason'].astype(str) + f'; Selected based on {order_desc} debit amounts'
                else:
                    result_df['selection_reason'] = result_df['selection_reason'].astype(str) + f'; Selected based on {order_desc} {parsed_criteria["sort_by"]} values'
                result_df['selection_reason'] = result_df['selection_reason'].str.strip('; ')
        
        # Apply sample size limit
        if parsed_criteria['sample_size'] and parsed_criteria['sample_size'] > 0 and len(result_df) > 0:
            result_df = result_df.head(parsed_criteria['sample_size'])
            
            # Update selection reason to include sample size
            if 'selection_reason' not in result_df.columns:
                result_df['selection_reason'] = ''
            result_df['selection_reason'] = result_df['selection_reason'].astype(str) + f'; Top {parsed_criteria["sample_size"]} selected'
            result_df['selection_reason'] = result_df['selection_reason'].str.strip('; ')
        
        # If no specific filters but risk analysis was requested, return high-risk transactions
        if (not any([parsed_criteria['amount_filters'], parsed_criteria['description_filters'],
                    parsed_criteria['date_filters'], parsed_criteria['duplicate_check']]) and
            parsed_criteria['risk_analysis'] and 'audit_risk_score' in result_df.columns):
            
            # Return transactions with medium to high risk
            result_df = result_df[result_df['audit_risk_score'] >= 0.3]
            result_df = result_df.sort_values('audit_risk_score', ascending=False)
            
            if parsed_criteria['sample_size']:
                result_df = result_df.head(parsed_criteria['sample_size'])
        
        return result_df
    
    def apply_amount_filters(self, df, filters, amount_column):
        """Apply amount-based filters"""
        # Handle special case for net_amount (debit/credit processing)
        if amount_column == 'net_amount' and amount_column in df.columns:
            numeric_column = amount_column
        else:
            # Ensure numeric column exists
            if f'{amount_column}_numeric' not in df.columns:
                df = self.audit_analyzer.analyze_amounts(df, amount_column)
            
            numeric_column = f'{amount_column}_numeric'
            if numeric_column not in df.columns:
                return df
        
        mask = pd.Series([True] * len(df), index=df.index)
        applied_filters = []
        
        for filter_item in filters:
            operator = filter_item['operator']
            value = filter_item['value']
            
            if operator == 'greater_than':
                filter_mask = df[numeric_column] > value
                mask &= filter_mask
                applied_filters.append(f'Amount > {value:,.2f}')
            elif operator == 'less_than':
                filter_mask = df[numeric_column] < value
                mask &= filter_mask
                applied_filters.append(f'Amount < {value:,.2f}')
            elif operator == 'equal_to':
                filter_mask = df[numeric_column] == value
                mask &= filter_mask
                applied_filters.append(f'Amount = {value:,.2f}')
            elif operator == 'round_numbers':
                # Check for round numbers
                round_mask = df[numeric_column].apply(
                    lambda x: self.audit_analyzer.is_round_number(x) if pd.notna(x) else False
                )
                mask &= round_mask
                applied_filters.append('Round number amounts')
        
        filtered_df = df[mask].copy()
        
        # Add selection reason for amount filters
        if len(applied_filters) > 0 and len(filtered_df) > 0:
            if 'selection_reason' not in filtered_df.columns:
                filtered_df['selection_reason'] = ''
            
            filter_reason = 'Amount criteria: ' + '; '.join(applied_filters)
            filtered_df['selection_reason'] = filtered_df['selection_reason'].astype(str) + '; ' + filter_reason
            filtered_df['selection_reason'] = filtered_df['selection_reason'].str.strip('; ')
        
        return filtered_df
    
    def apply_description_filters(self, df, filters, description_column):
        """Apply description-based filters"""
        mask = pd.Series([True] * len(df), index=df.index)
        
        for filter_item in filters:
            filter_type = filter_item['type']
            
            if filter_type == 'empty_description':
                empty_mask = (
                    df[description_column].isna() |
                    (df[description_column].astype(str).str.strip() == '') |
                    (df[description_column].astype(str).str.len() <= 3)
                )
                mask &= empty_mask
            
            elif filter_type == 'suspicious_description':
                if 'audit_risk_score' in df.columns:
                    mask &= df['audit_risk_score'] >= 0.4
                else:
                    # Fallback to keyword matching - flatten all suspicious keywords
                    all_keywords = []
                    for category, keywords in self.audit_analyzer.suspicious_keywords.items():
                        all_keywords.extend(keywords)
                    
                    suspicious_pattern = '|'.join([re.escape(keyword) for keyword in all_keywords])
                    suspicious_mask = df[description_column].astype(str).str.lower().str.contains(
                        suspicious_pattern, na=False, regex=True
                    )
                    mask &= suspicious_mask
            
            elif filter_type == 'vague_description':
                if 'audit_risk_score' in df.columns:
                    vague_mask = df['risk_factors'].astype(str).str.contains('Generic description', na=False)
                    mask &= vague_mask
        
        return df[mask]
    
    def apply_date_filters(self, df, filters, date_column):
        """Apply date-based filters"""
        # Convert to datetime if not already done
        if f'{date_column}_parsed' not in df.columns:
            df = self.audit_analyzer.analyze_timing_patterns(df, date_column)
        
        mask = pd.Series([True] * len(df), index=df.index)
        
        for filter_item in filters:
            filter_type = filter_item['type']
            
            if filter_type == 'weekend':
                if 'timing_risk_factors' in df.columns:
                    weekend_mask = df['timing_risk_factors'].astype(str).str.contains(
                        'Weekend transaction', na=False
                    )
                    mask &= weekend_mask
            
            elif filter_type == 'end_of_period':
                if 'timing_risk_factors' in df.columns:
                    end_period_mask = df['timing_risk_factors'].astype(str).str.contains(
                        'End-of-', na=False
                    )
                    mask &= end_period_mask
        
        return df[mask]
    
    def apply_text_patterns(self, df, patterns, description_column):
        """Apply text pattern matching to descriptions"""
        if not patterns:
            return df
        
        # Create regex pattern from all text patterns
        pattern_regex = '|'.join([re.escape(pattern.lower()) for pattern in patterns])
        
        mask = df[description_column].astype(str).str.lower().str.contains(
            pattern_regex, na=False, regex=True
        )
        
        return df[mask]
    
    def apply_transaction_type_filter(self, df, transaction_type):
        """Apply transaction type filter for credit/debit specific queries"""
        if 'transaction_type' not in df.columns:
            return df
        
        filtered_df = df.copy()
        
        if transaction_type == 'credit':
            filtered_df = filtered_df[filtered_df['transaction_type'] == 'Credit']
            if len(filtered_df) > 0:
                if 'selection_reason' not in filtered_df.columns:
                    filtered_df['selection_reason'] = ''
                filtered_df['selection_reason'] = filtered_df['selection_reason'].astype(str) + '; Credit transactions only'
                filtered_df['selection_reason'] = filtered_df['selection_reason'].str.strip('; ')
        
        elif transaction_type == 'debit':
            filtered_df = filtered_df[filtered_df['transaction_type'] == 'Debit']
            if len(filtered_df) > 0:
                if 'selection_reason' not in filtered_df.columns:
                    filtered_df['selection_reason'] = ''
                filtered_df['selection_reason'] = filtered_df['selection_reason'].astype(str) + '; Debit transactions only'
                filtered_df['selection_reason'] = filtered_df['selection_reason'].str.strip('; ')
        
        return filtered_df
    
    def get_interpretation_summary(self, parsed_criteria):
        """Generate human-readable summary of criteria interpretation"""
        summary_parts = []
        
        # Sample size
        if parsed_criteria['sample_size']:
            summary_parts.append(f"Sample size: {parsed_criteria['sample_size']} transactions")
        
        # Criteria types
        if parsed_criteria['criteria_types']:
            types_str = ', '.join(parsed_criteria['criteria_types'])
            summary_parts.append(f"Analysis types: {types_str}")
        
        # Amount filters
        if parsed_criteria['amount_filters']:
            for filter_item in parsed_criteria['amount_filters']:
                operator = filter_item['operator']
                value = filter_item.get('value')
                
                if operator == 'greater_than':
                    summary_parts.append(f"Amount > {value:,.2f}")
                elif operator == 'less_than':
                    summary_parts.append(f"Amount < {value:,.2f}")
                elif operator == 'round_numbers':
                    summary_parts.append("Round number amounts")
        
        # Description filters
        if parsed_criteria['description_filters']:
            desc_types = [f['type'] for f in parsed_criteria['description_filters']]
            summary_parts.append(f"Description analysis: {', '.join(desc_types)}")
        
        # Other filters
        if parsed_criteria['duplicate_check']:
            summary_parts.append("Duplicate detection enabled")
        
        if parsed_criteria['risk_analysis']:
            summary_parts.append("Audit risk analysis enabled")
        
        # Sorting
        if parsed_criteria['sort_by']:
            order = 'descending' if parsed_criteria['sort_order'] == 'desc' else 'ascending'
            summary_parts.append(f"Sorted by {parsed_criteria['sort_by']} ({order})")
        
        if not summary_parts:
            return "No specific criteria interpreted - showing all data"
        
        return " | ".join(summary_parts)
