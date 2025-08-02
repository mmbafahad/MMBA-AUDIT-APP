import pandas as pd
import numpy as np
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
        """Load and process pasted CSV data"""
        try:
            # Clean the pasted text
            pasted_text = pasted_text.strip()
            
            if not pasted_text:
                raise ValueError("No data provided")
            
            # Convert to StringIO for pandas to read
            data_io = StringIO(pasted_text)
            
            # Try to read as CSV
            try:
                df = pd.read_csv(data_io)
            except Exception as e:
                # Try with different delimiter
                data_io.seek(0)
                try:
                    df = pd.read_csv(data_io, delimiter='\t')
                except:
                    # Try semicolon delimiter
                    data_io.seek(0)
                    df = pd.read_csv(data_io, delimiter=';')
            
            if len(df.columns) < 2:
                raise ValueError("Data must have at least 2 columns. Please check your format.")
            
            if len(df) == 0:
                raise ValueError("No data rows found. Please include data below the headers.")
            
            # Basic data cleaning
            df = self.clean_data(df)
            
            return df
            
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
