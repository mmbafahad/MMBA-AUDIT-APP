"""
Prebuilt audit criteria for standard audit checks
"""
import pandas as pd
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime, date

class PrebuiltCriteriaProcessor:
    def __init__(self, audit_analyzer):
        self.audit_analyzer = audit_analyzer
        
    def apply_all_prebuilt_criteria(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                                   criteria_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Apply all enabled prebuilt criteria and return results with statistics"""
        # Apply account filter if specified
        working_df = self._apply_account_filter(df, column_mapping, criteria_config)
        
        all_results = []
        criteria_stats = {}
        
        # 1. High-Value Transactions
        if criteria_config.get('high_value_enabled', True):
            high_value_results = self._check_high_value_transactions(
                working_df, column_mapping, criteria_config.get('high_value_config', {})
            )
            if len(high_value_results) > 0:
                all_results.append(high_value_results)
                criteria_stats['High-Value Transactions'] = len(high_value_results)
        
        # 2. Missing Description
        if criteria_config.get('missing_desc_enabled', True):
            missing_desc_results = self._check_missing_descriptions(working_df, column_mapping)
            if len(missing_desc_results) > 0:
                all_results.append(missing_desc_results)
                criteria_stats['Missing Description'] = len(missing_desc_results)
        
        # 3. Suspicious Description
        if criteria_config.get('suspicious_desc_enabled', True):
            suspicious_desc_results = self._check_suspicious_descriptions(working_df, column_mapping)
            if len(suspicious_desc_results) > 0:
                all_results.append(suspicious_desc_results)
                criteria_stats['Suspicious Description'] = len(suspicious_desc_results)
        
        # 4. Duplicate Transaction Check
        if criteria_config.get('duplicate_enabled', True):
            duplicate_results = self._check_duplicate_transactions(
                working_df, column_mapping, criteria_config.get('duplicate_config', {})
            )
            if len(duplicate_results) > 0:
                all_results.append(duplicate_results)
                criteria_stats['Duplicate Transactions'] = len(duplicate_results)
        
        # 5. Round Amount Check
        if criteria_config.get('round_amount_enabled', True):
            round_amount_results = self._check_round_amounts(
                working_df, column_mapping, criteria_config.get('round_amount_config', {})
            )
            if len(round_amount_results) > 0:
                all_results.append(round_amount_results)
                criteria_stats['Round Amounts'] = len(round_amount_results)
        
        # 6. Backdated/Future-Dated Check
        if criteria_config.get('date_check_enabled', True):
            date_check_results = self._check_unusual_dates(
                working_df, column_mapping, criteria_config.get('date_config', {})
            )
            if len(date_check_results) > 0:
                all_results.append(date_check_results)
                criteria_stats['Unusual Dates'] = len(date_check_results)
        
        # Combine all results and remove duplicates while preserving all selection reasons
        if all_results:
            combined_df = self._combine_results_with_reasons(all_results)
            return combined_df, criteria_stats
        else:
            return df.iloc[0:0].copy(), criteria_stats
    
    def _check_high_value_transactions(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                                     config: Dict[str, Any]) -> pd.DataFrame:
        """Check for high-value transactions"""
        amount_column = column_mapping.get('amount', 'amount')
        if amount_column not in df.columns:
            return df.iloc[0:0].copy()
        
        result_df = df.copy()
        
        # Convert amount to numeric
        result_df[amount_column] = pd.to_numeric(result_df[amount_column], errors='coerce')
        
        # Apply type filter (credit/debit)
        transaction_type = config.get('transaction_type', 'both')
        if transaction_type == 'credit' and 'credit_amount_only' in result_df.columns:
            # Filter for credit transactions only
            result_df = result_df[result_df['credit_amount_only'] > 0]
            sort_column = 'credit_amount_only'
        elif transaction_type == 'debit' and 'debit_amount_only' in result_df.columns:
            # Filter for debit transactions only
            result_df = result_df[result_df['debit_amount_only'] > 0]
            sort_column = 'debit_amount_only'
        else:
            # Use absolute amount
            result_df['abs_amount'] = result_df[amount_column].abs()
            sort_column = 'abs_amount'
        
        # Apply threshold filter if specified
        threshold = config.get('threshold_amount')
        if threshold and threshold > 0:
            if transaction_type == 'credit' and 'credit_amount_only' in result_df.columns:
                result_df = result_df[result_df['credit_amount_only'] >= threshold]
            elif transaction_type == 'debit' and 'debit_amount_only' in result_df.columns:
                result_df = result_df[result_df['debit_amount_only'] >= threshold]
            else:
                result_df = result_df[result_df['abs_amount'] >= threshold]
        
        # Sort by amount (highest first)
        result_df = result_df.sort_values(sort_column, ascending=False)
        
        # Limit to specified number
        num_transactions = config.get('num_transactions', 10)
        result_df = result_df.head(num_transactions)
        
        # Add selection reason
        threshold_text = f" above {threshold}" if threshold and threshold > 0 else ""
        type_text = f" {transaction_type}" if transaction_type != 'both' else ""
        if len(result_df) > 0:
            result_df['selection_reason'] = f"High-Value Transaction: Top {num_transactions}{type_text} transactions{threshold_text}"
        
        return result_df
    
    def _check_missing_descriptions(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Check for transactions with missing descriptions"""
        description_column = column_mapping.get('description', 'description')
        if description_column not in df.columns:
            return df.iloc[0:0].copy()
        
        # Check for empty, null, or meaningless descriptions
        mask = (df[description_column].isna() | 
                (df[description_column].astype(str).str.strip() == '') |
                (df[description_column].astype(str).str.lower().isin(['nan', 'null', 'none', 'n/a', '-'])))
        
        result_df = df[mask].copy()
        if len(result_df) > 0:
            result_df['selection_reason'] = "Missing Description: Transaction has empty or missing description"
        
        return result_df
    
    def _check_suspicious_descriptions(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Check for transactions with suspicious descriptions"""
        description_column = column_mapping.get('description', 'description')
        if description_column not in df.columns:
            return df.iloc[0:0].copy()
        
        # Suspicious keywords
        suspicious_keywords = [
            'adjustment', 'correction', 'reversal', 'unknown', 'n/a', 'gift', 'donation',
            'political', 'personal', 'fraud', 'error', 'write-off', 'loss', 'cash short',
            'overpayment', 'underpayment', 'director', 'shareholder', 'owner', 'affiliate', 'subsidiary'
        ]
        
        # Create regex pattern for case-insensitive matching
        pattern = '|'.join([r'\b' + keyword + r'\b' for keyword in suspicious_keywords])
        
        mask = df[description_column].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
        result_df = df[mask].copy()
        
        # Add specific reason based on matched keyword
        def get_matched_keywords(text):
            if pd.isna(text):
                return ""
            text_lower = str(text).lower()
            matched = [kw for kw in suspicious_keywords if re.search(r'\b' + kw + r'\b', text_lower)]
            return ', '.join(matched) if matched else "suspicious pattern"
        
        if len(result_df) > 0:
            result_df['matched_keywords'] = result_df[description_column].apply(get_matched_keywords)
            result_df['selection_reason'] = "Suspicious Description: Contains keywords - " + result_df['matched_keywords']
            result_df = result_df.drop('matched_keywords', axis=1)
        
        return result_df
    
    def _check_duplicate_transactions(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                                    config: Dict[str, Any]) -> pd.DataFrame:
        """Check for duplicate transactions"""
        duplicate_by = config.get('duplicate_by', ['reference', 'amount'])
        
        columns_to_check = []
        reason_parts = []
        
        if 'reference' in duplicate_by and column_mapping.get('reference') in df.columns:
            columns_to_check.append(column_mapping['reference'])
            reason_parts.append('reference')
        
        if 'amount' in duplicate_by and column_mapping.get('amount') in df.columns:
            columns_to_check.append(column_mapping['amount'])
            reason_parts.append('amount')
        
        if not columns_to_check:
            return df.iloc[0:0].copy()
        
        # Find duplicates
        mask = df.duplicated(subset=columns_to_check, keep=False)
        result_df = df[mask].copy()
        
        if len(result_df) > 0 and reason_parts:
            reason_text = ' and '.join(reason_parts)
            result_df['selection_reason'] = f"Duplicate Transaction: Duplicate {reason_text} detected"
        
        return result_df
    
    def _check_round_amounts(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                           config: Dict[str, Any]) -> pd.DataFrame:
        """Check for round amount transactions"""
        amount_column = column_mapping.get('amount', 'amount')
        if amount_column not in df.columns:
            return df.iloc[0:0].copy()
        
        # Convert to numeric and get absolute values
        numeric_amounts = pd.to_numeric(df[amount_column], errors='coerce').abs()
        
        # Get roundness threshold
        roundness_type = config.get('roundness_type', '100')
        if roundness_type == 'custom':
            threshold = config.get('custom_threshold', 100)
        else:
            threshold = int(roundness_type)
        
        # Check for round amounts
        mask = (numeric_amounts % threshold == 0) & (numeric_amounts > 0)
        result_df = df[mask].copy()
        
        if len(result_df) > 0:
            result_df['selection_reason'] = f"Round Amount: Transaction amount is multiple of {threshold}"
        
        return result_df
    
    def _check_unusual_dates(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                           config: Dict[str, Any]) -> pd.DataFrame:
        """Check for backdated or future-dated transactions"""
        date_column = column_mapping.get('date', 'date')
        if date_column not in df.columns:
            return df.iloc[0:0].copy()
        
        start_date = config.get('expected_start_date')
        end_date = config.get('expected_end_date')
        
        if not start_date or not end_date:
            return df.iloc[0:0].copy()
        
        # Convert dates
        try:
            df_dates = pd.to_datetime(df[date_column], errors='coerce')
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except Exception:
            return df.iloc[0:0].copy()
        
        # Find transactions outside the expected date range
        mask = (df_dates < start_date) | (df_dates > end_date)
        result_df = df[mask].copy()
        
        if len(result_df) > 0:
            def get_date_reason(transaction_date):
                if pd.isna(transaction_date):
                    return "Invalid date"
                elif transaction_date < start_date:
                    return f"Backdated: Before expected start ({start_date.strftime('%Y-%m-%d')})"
                elif transaction_date > end_date:
                    return f"Future-dated: After expected end ({end_date.strftime('%Y-%m-%d')})"
                return "Date issue"
            
            result_df['date_reason'] = pd.to_datetime(result_df[date_column], errors='coerce').apply(get_date_reason)
            result_df['selection_reason'] = "Unusual Date: " + result_df['date_reason']
            result_df = result_df.drop('date_reason', axis=1)
        
        return result_df
    
    def _combine_results_with_reasons(self, results_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine results while preserving all selection reasons for duplicates"""
        if not results_list:
            return pd.DataFrame()
        
        # Ensure all dataframes have selection_reason column
        processed_results = []
        for df in results_list:
            if 'selection_reason' not in df.columns and len(df) > 0:
                df = df.copy()
                df['selection_reason'] = 'Selected by criteria'
            processed_results.append(df)
        
        # Filter out empty dataframes
        processed_results = [df for df in processed_results if len(df) > 0]
        
        if not processed_results:
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(processed_results, ignore_index=True)
        
        # Group by all columns except selection_reason to combine reasons for duplicates
        if 'selection_reason' in combined_df.columns and len(combined_df) > 0:
            # Get all columns except selection_reason for grouping
            group_columns = [col for col in combined_df.columns if col != 'selection_reason']
            
            if group_columns:  # Only group if there are columns to group by
                # Group and combine selection reasons
                def combine_reasons(group):
                    reasons = group['selection_reason'].unique()
                    return ' | '.join(reasons)
                
                # Group by all columns except selection_reason
                try:
                    grouped = combined_df.groupby(group_columns, as_index=False).agg({
                        'selection_reason': combine_reasons
                    })
                    return grouped
                except Exception:
                    # If grouping fails, just remove duplicates
                    return combined_df.drop_duplicates()
            else:
                return combined_df
        
        return combined_df.drop_duplicates()
    
    def _apply_account_filter(self, df: pd.DataFrame, column_mapping: Dict[str, str], 
                            criteria_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply account filter if specified in criteria config"""
        account_column = column_mapping.get('account')
        selected_accounts = criteria_config.get('selected_accounts', [])
        
        # If no account column mapped or no accounts selected, return full dataset
        if not account_column or not selected_accounts or account_column not in df.columns:
            return df
        
        # Filter for selected accounts
        filtered_df = df[df[account_column].isin(selected_accounts)].copy()
        return filtered_df
    
    def get_unique_accounts(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> List[str]:
        """Get unique account names from the account column"""
        account_column = column_mapping.get('account')
        
        if not account_column or account_column not in df.columns:
            return []
        
        # Get unique account names, excluding null/empty values
        accounts = df[account_column].dropna().astype(str).str.strip()
        unique_accounts = accounts[accounts != ''].unique().tolist()
        
        # Sort accounts for better UI
        return sorted(unique_accounts)