import pandas as pd
import numpy as np
import re
from io import StringIO
import streamlit as st

class DataProcessor:
    """Handle data loading and preprocessing for transaction files"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'xls']
    
    def load_file(self, uploaded_file):
        """Load and process uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings for CSV
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
                    
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic data cleaning
            df = self.clean_data(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load file: {str(e)}")
    
    def load_pasted_data(self, pasted_text):
        """Load and process pasted CSV data with enhanced parsing for Excel-copied data"""
        try:
            # Clean the pasted text
            pasted_text = pasted_text.strip()
            
            if not pasted_text:
                raise ValueError("No data provided")
            
            # Split into lines and detect the data structure
            lines = pasted_text.split('\n')
            
            # Remove empty lines
            lines = [line.strip() for line in lines if line.strip()]
            
            if len(lines) < 2:
                raise ValueError("Data must have at least a header row and one data row.")
            
            # Try different parsing approaches
            df = None
            
            # Method 1: Try tab-separated (most common from Excel copy-paste)
            try:
                data_io = StringIO(pasted_text)
                df = pd.read_csv(data_io, delimiter='\t', skipinitialspace=True)
                if len(df.columns) >= 2 and len(df) > 0:
                    # Check if we have meaningful data
                    non_empty_cols = sum(1 for col in df.columns if not df[col].isna().all())
                    if non_empty_cols >= 2:
                        return self.clean_data(df)
            except Exception:
                pass
            
            # Method 2: Try comma-separated
            try:
                data_io = StringIO(pasted_text)
                df = pd.read_csv(data_io, delimiter=',', skipinitialspace=True)
                if len(df.columns) >= 2 and len(df) > 0:
                    non_empty_cols = sum(1 for col in df.columns if not df[col].isna().all())
                    if non_empty_cols >= 2:
                        return self.clean_data(df)
            except Exception:
                pass
            
            # Method 3: Try semicolon-separated
            try:
                data_io = StringIO(pasted_text)
                df = pd.read_csv(data_io, delimiter=';', skipinitialspace=True)
                if len(df.columns) >= 2 and len(df) > 0:
                    non_empty_cols = sum(1 for col in df.columns if not df[col].isna().all())
                    if non_empty_cols >= 2:
                        return self.clean_data(df)
            except Exception:
                pass
            
            # Method 4: Try to detect multiple spaces as delimiter (common in formatted reports)
            try:
                # Replace multiple spaces with tabs
                processed_text = re.sub(r'\s{2,}', '\t', pasted_text)
                data_io = StringIO(processed_text)
                df = pd.read_csv(data_io, delimiter='\t', skipinitialspace=True)
                if len(df.columns) >= 2 and len(df) > 0:
                    non_empty_cols = sum(1 for col in df.columns if not df[col].isna().all())
                    if non_empty_cols >= 2:
                        return self.clean_data(df)
            except Exception:
                pass
            
            # Method 5: Try fixed-width parsing for highly structured data
            try:
                df = pd.read_fwf(StringIO(pasted_text), skipinitialspace=True)
                if len(df.columns) >= 2 and len(df) > 0:
                    non_empty_cols = sum(1 for col in df.columns if not df[col].isna().all())
                    if non_empty_cols >= 2:
                        return self.clean_data(df)
            except Exception:
                pass
            
            # If all methods failed, provide helpful error message
            raise ValueError("""
            Could not parse the pasted data. Please ensure your data:
            1. Has column headers in the first row
            2. Uses consistent delimiters (tabs, commas, or spaces)
            3. Has at least 2 columns and 1 data row
            
            Tip: Copy data directly from Excel or save as CSV and paste the content.
            """)
            
        except Exception as e:
            raise Exception(f"Failed to process pasted data: {str(e)}")
    
    def clean_data(self, df):
        """Perform basic data cleaning"""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Replace empty strings with NaN
        df = df.replace('', np.nan)
        
        return df
    
    def detect_column_types(self, df, column_mapping):
        """Detect and suggest column types based on content"""
        suggestions = {}
        
        for col in df.columns:
            sample_data = df[col].dropna().head(100)
            
            # Check for amount columns (numeric with currency symbols)
            if self.is_amount_column(sample_data):
                suggestions[col] = 'amount'
            
            # Check for date columns
            elif self.is_date_column(sample_data):
                suggestions[col] = 'date'
            
            # Check for description columns (longer text)
            elif self.is_description_column(sample_data):
                suggestions[col] = 'description'
            
            # Check for reference/ID columns
            elif self.is_reference_column(sample_data):
                suggestions[col] = 'reference'
        
        return suggestions
    
    def is_amount_column(self, series):
        """Check if column contains amount data"""
        try:
            # Remove currency symbols and convert to numeric
            cleaned = series.astype(str).str.replace(r'[,$£€¥]', '', regex=True)
            numeric_conversion = pd.to_numeric(cleaned, errors='coerce')
            numeric_ratio = numeric_conversion.notna().sum() / len(series)
            return numeric_ratio > 0.8
        except:
            return False
    
    def is_date_column(self, series):
        """Check if column contains date data"""
        try:
            date_conversion = pd.to_datetime(series, errors='coerce')
            date_ratio = date_conversion.notna().sum() / len(series)
            return date_ratio > 0.8
        except:
            return False
    
    def is_description_column(self, series):
        """Check if column contains description text"""
        try:
            # Check average length of text
            avg_length = series.astype(str).str.len().mean()
            return avg_length > 10  # Descriptions typically longer than 10 characters
        except:
            return False
    
    def is_reference_column(self, series):
        """Check if column contains reference numbers/IDs"""
        try:
            # Check for alphanumeric patterns typical of references
            str_series = series.astype(str)
            # Look for patterns like: numbers, alphanumeric codes
            pattern_match = str_series.str.match(r'^[A-Z0-9\-_]+$', case=False)
            pattern_ratio = pattern_match.sum() / len(series)
            return pattern_ratio > 0.7
        except:
            return False
    
    def prepare_amount_column(self, df, column_name):
        """Clean and prepare amount column for analysis"""
        if column_name not in df.columns:
            return df
        
        # Create a copy to avoid modifying original
        cleaned_amounts = df[column_name].astype(str).copy()
        
        # Remove currency symbols and commas
        cleaned_amounts = cleaned_amounts.str.replace(r'[,$£€¥]', '', regex=True)
        
        # Handle parentheses for negative numbers
        negative_mask = cleaned_amounts.str.contains(r'\(.*\)', regex=True, na=False)
        cleaned_amounts = cleaned_amounts.str.replace(r'[()]', '', regex=True)
        
        # Convert to numeric
        numeric_amounts = pd.to_numeric(cleaned_amounts, errors='coerce')
        
        # Apply negative sign where parentheses were found
        numeric_amounts.loc[negative_mask] *= -1
        
        # Add cleaned column to dataframe
        df[f'{column_name}_numeric'] = numeric_amounts
        
        return df
    
    def prepare_date_column(self, df, column_name):
        """Clean and prepare date column for analysis"""
        if column_name not in df.columns:
            return df
        
        # Convert to datetime
        df[f'{column_name}_parsed'] = pd.to_datetime(df[column_name], errors='coerce')
        
        return df
    
    def process_debit_credit_columns(self, df, debit_col, credit_col):
        """Process debit and credit columns into a single amount column"""
        if debit_col not in df.columns or credit_col not in df.columns:
            return df, None
        
        df = df.copy()
        
        # Clean and convert debit column
        debit_amounts = pd.to_numeric(
            df[debit_col].astype(str).str.replace(r'[,$£€¥()]', '', regex=True),
            errors='coerce'
        ).fillna(0)
        
        # Clean and convert credit column  
        credit_amounts = pd.to_numeric(
            df[credit_col].astype(str).str.replace(r'[,$£€¥()]', '', regex=True),
            errors='coerce'
        ).fillna(0)
        
        # Create net amount column (debit positive, credit negative)
        df['net_amount'] = debit_amounts - credit_amounts
        
        # Create transaction type based on which column has value
        df['transaction_type'] = ''
        df.loc[debit_amounts > 0, 'transaction_type'] = 'Debit'
        df.loc[credit_amounts > 0, 'transaction_type'] = 'Credit'
        df.loc[(debit_amounts > 0) & (credit_amounts > 0), 'transaction_type'] = 'Both'
        
        # Keep original debit and credit amounts for filtering
        df['debit_amount_only'] = debit_amounts
        df['credit_amount_only'] = credit_amounts
        
        return df, 'net_amount'
