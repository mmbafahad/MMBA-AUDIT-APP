import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st

class NLPProcessor:
    """Natural Language Processing for criteria interpretation"""
    
    def __init__(self):
        self.setup_nlp()
    
    @st.cache_resource
    def setup_nlp(_self):
        """Initialize NLP models and download required NLTK data"""
        try:
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            # Try to load spaCy model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                # If model not available, use basic text processing
                nlp = None
            
            return nlp
        except Exception:
            return None
    
    def extract_numbers(self, text):
        """Extract numbers from text"""
        # Look for numbers in various formats
        number_patterns = [
            r'\b\d+\b',  # Simple integers
            r'\b\d+\.\d+\b',  # Decimals
            r'\b\d+,\d+\b',  # Numbers with commas
            r'\b\d+k\b',  # Numbers with k suffix
            r'\b\d+m\b',  # Numbers with m suffix
        ]
        
        numbers = []
        for pattern in number_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    # Convert k and m suffixes
                    if match.endswith('k'):
                        numbers.append(float(match[:-1]) * 1000)
                    elif match.endswith('m'):
                        numbers.append(float(match[:-1]) * 1000000)
                    else:
                        # Remove commas and convert
                        clean_number = match.replace(',', '')
                        numbers.append(float(clean_number))
                except ValueError:
                    continue
        
        return numbers
    
    def extract_keywords(self, text):
        """Extract relevant keywords from criteria text"""
        # Common audit and transaction keywords
        keywords = {
            'actions': ['select', 'find', 'get', 'show', 'extract', 'identify', 'filter'],
            'amounts': ['highest', 'lowest', 'above', 'below', 'greater', 'less', 'over', 'under', 'maximum', 'minimum'],
            'descriptions': ['description', 'desc', 'narrative', 'memo', 'comment'],
            'suspicious': ['suspicious', 'suspense', 'misc', 'miscellaneous', 'various', 'vague', 'unclear', 'doubtful'],
            'duplicates': ['duplicate', 'duplicated', 'same', 'identical', 'repeated'],
            'empty': ['empty', 'blank', 'null', 'missing', 'no description'],
            'dates': ['weekend', 'holiday', 'after hours', 'late', 'early'],
            'patterns': ['round', 'ending', 'starting', 'containing'],
            'credit_specific': ['credit', 'credits', 'credit amount', 'credit amounts', 'credit only', 'credits only'],
            'debit_specific': ['debit', 'debits', 'debit amount', 'debit amounts', 'debit only', 'debits only']
        }
        
        found_keywords = {}
        text_lower = text.lower()
        
        for category, words in keywords.items():
            found_keywords[category] = [word for word in words if word in text_lower]
        
        return found_keywords
    
    def identify_criteria_type(self, text, keywords):
        """Identify the type of sampling criteria"""
        criteria_types = []
        
        # Amount-based criteria
        if keywords['amounts'] or any(word in text.lower() for word in ['value', 'amount', 'dollar', '$']):
            criteria_types.append('amount')
        
        # Description-based criteria
        if (keywords['descriptions'] or keywords['suspicious'] or keywords['empty'] or
            any(word in text.lower() for word in ['description', 'desc', 'narrative', 'memo', 'comment'])):
            criteria_types.append('description')
        
        # Duplicate detection
        if keywords['duplicates']:
            criteria_types.append('duplicates')
        
        # Date-based criteria
        if keywords['dates'] or any(word in text.lower() for word in ['date', 'time', 'day']):
            criteria_types.append('date')
        
        # Pattern-based criteria
        if keywords['patterns']:
            criteria_types.append('patterns')
        
        # Top/bottom selection
        if any(word in text.lower() for word in ['top', 'bottom', 'first', 'last', 'highest', 'lowest']):
            criteria_types.append('ranking')
        
        return criteria_types
    
    def parse_comparison_operators(self, text):
        """Extract comparison operators and values"""
        comparisons = []
        
        # Patterns for different comparison types
        patterns = {
            'greater_than': [r'(?:above|over|greater than|more than|>\s*)(\d+(?:\.\d+)?)',
                           r'(\d+(?:\.\d+)?)\s*(?:and above|or more|or higher)'],
            'less_than': [r'(?:below|under|less than|fewer than|<\s*)(\d+(?:\.\d+)?)',
                         r'(\d+(?:\.\d+)?)\s*(?:and below|or less|or lower)'],
            'equal_to': [r'(?:equal to|equals|=\s*)(\d+(?:\.\d+)?)',
                        r'exactly\s*(\d+(?:\.\d+)?)']
        }
        
        for operator, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text.lower())
                for match in matches:
                    try:
                        value = float(match)
                        comparisons.append({'operator': operator, 'value': value})
                    except ValueError:
                        continue
        
        return comparisons
    
    def extract_sample_size(self, text):
        """Extract the desired sample size from text"""
        # Look for sample size indicators
        patterns = [
            r'(?:select|get|find|show)\s*(\d+)',
            r'(\d+)\s*(?:transactions?|records?|entries?|items?)',
            r'(\d+)\s*(?:highest|lowest|largest|smallest)',
            r'top\s*(\d+)',
            r'first\s*(\d+)',
            r'sample\s*(?:of\s*)?(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    return int(matches[0])
                except ValueError:
                    continue
        
        return None
    
    def extract_text_patterns(self, text):
        """Extract text patterns for description matching"""
        patterns = []
        
        # Look for quoted strings
        quoted_patterns = re.findall(r'"([^"]*)"', text)
        patterns.extend(quoted_patterns)
        
        quoted_patterns = re.findall(r"'([^']*)'", text)
        patterns.extend(quoted_patterns)
        
        # Look for words after "containing", "with", "including", "like"
        containing_patterns = re.findall(r'(?:containing|with|including|like)\s+([a-zA-Z0-9]+)', text.lower())
        patterns.extend(containing_patterns)
        
        # Look for "description of X" or "descriptions of X" patterns
        description_of_patterns = re.findall(r'description(?:s)?\s+of\s+([a-zA-Z0-9]+)', text.lower())
        patterns.extend(description_of_patterns)
        
        # Look for "with description X" patterns  
        with_description_patterns = re.findall(r'with\s+description\s+([a-zA-Z0-9]+)', text.lower())
        patterns.extend(with_description_patterns)
        
        # Look for "contain/contains/containing X" patterns (wider match)
        contain_patterns = re.findall(r'(?:contain|contains|containing)\s+([a-zA-Z0-9]+)', text.lower())
        patterns.extend(contain_patterns)
        

        
        # Look for standalone capitalized words that might be company names (always check)
        # Find capitalized words that are likely company names
        capitalized_words = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,})\b', text)
        # Filter out common words
        common_words = {'Select', 'Find', 'Show', 'Get', 'With', 'Description', 'Transaction', 'Transactions', 'Amount', 'Amounts', 'AND', 'OR'}
        company_names = [word for word in capitalized_words if word not in common_words]
        patterns.extend(company_names)
        
        return patterns
