import re
import pandas as pd
from typing import List, Dict, Any, Union
from .nlp_processor import NLPProcessor
from .audit_analyzer import AuditAnalyzer

class ComplexCriteriaParser:
    """
    Enhanced criteria parser that supports complex logical operations with AND/OR
    and multiple filter types (numerical, text, reference numbers)
    """
    
    def __init__(self):
        self.nlp_processor = NLPProcessor()
        self.audit_analyzer = AuditAnalyzer()
    
    def parse_complex_criteria(self, criteria_text: str, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Parse complex criteria with logical operators
        Examples:
        - "15 highest amount transactions"
        - "transactions with NashTech in description"
        - "transactions with reference 123 OR reference 125"
        - "amount > 50000 AND description contains 'misc'"
        """
        
        # Initialize the parsed structure
        parsed = {
            'logical_structure': None,
            'conditions': [],
            'sample_size': None,
            'sort_by': None,
            'sort_order': 'desc',
            'raw_text': criteria_text
        }
        
        # Check for logical operators
        has_and = ' AND ' in criteria_text.upper() or ' and ' in criteria_text.lower()
        has_or = ' OR ' in criteria_text.upper() or ' or ' in criteria_text.lower()
        
        if has_and or has_or:
            # Parse complex logical structure
            parsed = self._parse_logical_structure(criteria_text, column_mapping)
        else:
            # Parse simple criteria
            parsed = self._parse_simple_criteria(criteria_text, column_mapping)
        
        return parsed
    
    def _parse_logical_structure(self, criteria_text: str, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Parse criteria with AND/OR operators"""
        
        # Split by AND/OR while preserving the operators
        # First handle parentheses for grouping
        criteria_text = self._handle_parentheses(criteria_text)
        
        # Split by logical operators
        parts = re.split(r'\s+(AND|OR|and|or)\s+', criteria_text)
        
        conditions = []
        operators = []
        
        for i, part in enumerate(parts):
            if part.upper() in ['AND', 'OR']:
                operators.append(part.upper())
            else:
                # Parse individual condition
                condition = self._parse_individual_condition(part.strip(), column_mapping)
                if condition:
                    conditions.append(condition)
        
        # Determine overall sample size and sorting from the first numerical condition
        sample_size = None
        sort_by = None
        sort_order = 'desc'
        
        for condition in conditions:
            if condition.get('sample_size') and not sample_size:
                sample_size = condition['sample_size']
            if condition.get('sort_by') and not sort_by:
                sort_by = condition['sort_by']
                sort_order = condition.get('sort_order', 'desc')
        
        return {
            'logical_structure': {
                'conditions': conditions,
                'operators': operators
            },
            'conditions': conditions,
            'sample_size': sample_size,
            'sort_by': sort_by,
            'sort_order': sort_order,
            'raw_text': criteria_text
        }
    
    def _parse_simple_criteria(self, criteria_text: str, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Parse simple criteria without logical operators"""
        condition = self._parse_individual_condition(criteria_text, column_mapping)
        
        return {
            'logical_structure': None,
            'conditions': [condition] if condition else [],
            'sample_size': condition.get('sample_size') if condition else None,
            'sort_by': condition.get('sort_by') if condition else None,
            'sort_order': condition.get('sort_order', 'desc') if condition else 'desc',
            'raw_text': criteria_text
        }
    
    def _parse_individual_condition(self, condition_text: str, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Parse a single condition"""
        condition = {
            'type': None,
            'filters': [],
            'sample_size': None,
            'sort_by': None,
            'sort_order': 'desc',
            'text': condition_text
        }
        
        # Extract numbers and keywords
        numbers = self.nlp_processor.extract_numbers(condition_text)
        keywords = self.nlp_processor.extract_keywords(condition_text)
        
        # Detect condition type and parse accordingly
        condition_type = self._detect_condition_type(condition_text, keywords)
        condition['type'] = condition_type
        
        if condition_type == 'numerical':
            condition.update(self._parse_numerical_condition(condition_text, numbers, keywords))
        elif condition_type == 'text':
            condition.update(self._parse_text_condition(condition_text, keywords, column_mapping))
        elif condition_type == 'reference':
            condition.update(self._parse_reference_condition(condition_text, numbers))
        elif condition_type == 'description':
            condition.update(self._parse_description_condition(condition_text, keywords))
        
        return condition
    
    def _detect_condition_type(self, text: str, keywords: Dict[str, List[str]]) -> str:
        """Detect the type of condition"""
        text_lower = text.lower()
        
        # Reference number patterns
        if any(word in text_lower for word in ['reference', 'ref', 'ref#', 'reference#']):
            return 'reference'
            
        # Numerical patterns
        if (keywords['amounts'] or 
            any(word in text_lower for word in ['amount', 'value', 'highest', 'lowest', 'top', 'bottom', '>', '<', 'above', 'below'])):
            return 'numerical'
            
        # Text/description patterns
        if (any(word in text_lower for word in ['description', 'desc', 'contains', 'with', 'in description']) or
            keywords['descriptions'] or keywords['suspicious'] or keywords['empty']):
            return 'text'
            
        # Default to description if contains specific terms
        return 'description'
    
    def _parse_numerical_condition(self, text: str, numbers: List[float], keywords: Dict[str, List[str]]) -> Dict[str, Any]:
        """Parse numerical filtering conditions"""
        filters = []
        sample_size = None
        sort_by = 'amount'
        sort_order = 'desc'
        
        # Extract sample size
        sample_size = self.nlp_processor.extract_sample_size(text)
        
        # Check for credit/debit specific
        if keywords['credit_specific']:
            sort_by = 'credit_amount_only'
        elif keywords['debit_specific']:
            sort_by = 'debit_amount_only'
        
        # Parse comparison operators
        comparisons = self.nlp_processor.parse_comparison_operators(text)
        filters.extend(comparisons)
        
        # Determine sort order
        if any(word in text.lower() for word in ['highest', 'top', 'maximum', 'largest']):
            sort_order = 'desc'
        elif any(word in text.lower() for word in ['lowest', 'bottom', 'minimum', 'smallest']):
            sort_order = 'asc'
        
        return {
            'filters': filters,
            'sample_size': sample_size,
            'sort_by': sort_by,
            'sort_order': sort_order
        }
    
    def _parse_text_condition(self, text: str, keywords: Dict[str, List[str]], column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Parse text matching conditions"""
        filters = []
        text_patterns = self.nlp_processor.extract_text_patterns(text)
        
        # Add text pattern filters
        for pattern in text_patterns:
            filters.append({
                'type': 'text_contains',
                'value': pattern,
                'field': 'description'
            })
        
        # Add suspicious description filters
        if keywords['suspicious']:
            filters.append({'type': 'suspicious_description'})
        
        # Add empty description filters
        if keywords['empty']:
            filters.append({'type': 'empty_description'})
        
        return {
            'filters': filters,
            'text_patterns': text_patterns
        }
    
    def _parse_reference_condition(self, text: str, numbers: List[float]) -> Dict[str, Any]:
        """Parse reference number filtering conditions"""
        filters = []
        
        # Extract reference numbers
        ref_patterns = [
            r'reference\s+(\d+)',
            r'ref\s+(\d+)',
            r'ref#\s*(\d+)',
            r'reference#\s*(\d+)'
        ]
        
        reference_numbers = []
        for pattern in ref_patterns:
            matches = re.findall(pattern, text.lower())
            reference_numbers.extend([int(match) for match in matches])
        
        # Also add any standalone numbers found
        reference_numbers.extend([int(num) for num in numbers if num == int(num)])
        
        # Remove duplicates
        reference_numbers = list(set(reference_numbers))
        
        for ref_num in reference_numbers:
            filters.append({
                'type': 'reference_equals',
                'value': ref_num,
                'field': 'reference'
            })
        
        return {
            'filters': filters,
            'reference_numbers': reference_numbers
        }
    
    def _parse_description_condition(self, text: str, keywords: Dict[str, List[str]]) -> Dict[str, Any]:
        """Parse general description conditions"""
        filters = []
        
        # Extract text patterns
        text_patterns = self.nlp_processor.extract_text_patterns(text)
        
        for pattern in text_patterns:
            filters.append({
                'type': 'text_contains',
                'value': pattern,
                'field': 'description'
            })
        
        return {
            'filters': filters,
            'text_patterns': text_patterns
        }
    
    def _handle_parentheses(self, text: str) -> str:
        """Handle parentheses in criteria (basic implementation)"""
        # For now, just remove parentheses - can be enhanced later for proper grouping
        return text.replace('(', '').replace(')', '')
    
    def apply_complex_criteria(self, df: pd.DataFrame, parsed_criteria: Dict[str, Any], column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply complex criteria to dataframe"""
        
        if not parsed_criteria['conditions']:
            return df
        
        if parsed_criteria['logical_structure']:
            # Handle complex logical structure
            return self._apply_logical_structure(df, parsed_criteria, column_mapping)
        else:
            # Handle simple criteria
            return self._apply_simple_condition(df, parsed_criteria['conditions'][0], column_mapping)
    
    def _apply_logical_structure(self, df: pd.DataFrame, parsed_criteria: Dict[str, Any], column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply logical structure with AND/OR operations"""
        conditions = parsed_criteria['logical_structure']['conditions']
        operators = parsed_criteria['logical_structure']['operators']
        
        if not conditions:
            return df
        
        # Apply first condition
        result_df = self._apply_simple_condition(df, conditions[0], column_mapping)
        
        # Apply subsequent conditions with operators
        for i, condition in enumerate(conditions[1:]):
            condition_result = self._apply_simple_condition(df, condition, column_mapping)
            
            if i < len(operators):
                operator = operators[i]
                if operator == 'AND':
                    # Intersection of results
                    common_indices = result_df.index.intersection(condition_result.index)
                    result_df = result_df.loc[common_indices]
                elif operator == 'OR':
                    # Union of results
                    result_df = pd.concat([result_df, condition_result]).drop_duplicates()
        
        # Apply overall sorting and sample size
        if parsed_criteria['sort_by'] and len(result_df) > 0:
            sort_column = self._get_sort_column(parsed_criteria['sort_by'], column_mapping, result_df)
            if sort_column and sort_column in result_df.columns:
                ascending = parsed_criteria['sort_order'] == 'asc'
                result_df = result_df.sort_values(sort_column, ascending=ascending)
        
        # Apply sample size
        if parsed_criteria['sample_size'] and len(result_df) > parsed_criteria['sample_size']:
            result_df = result_df.head(parsed_criteria['sample_size'])
        
        return result_df
    
    def _apply_simple_condition(self, df: pd.DataFrame, condition: Dict[str, Any], column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply a single condition to dataframe"""
        result_df = df.copy()
        
        for filter_item in condition.get('filters', []):
            result_df = self._apply_single_filter(result_df, filter_item, column_mapping)
        
        # Apply sorting for this condition
        if condition.get('sort_by') and len(result_df) > 0:
            sort_column = self._get_sort_column(condition['sort_by'], column_mapping, result_df)
            if sort_column and sort_column in result_df.columns:
                ascending = condition.get('sort_order', 'desc') == 'asc'
                result_df = result_df.sort_values(sort_column, ascending=ascending)
        
        # Apply sample size for this condition
        if condition.get('sample_size') and len(result_df) > condition['sample_size']:
            result_df = result_df.head(condition['sample_size'])
        
        return result_df
    
    def _apply_single_filter(self, df: pd.DataFrame, filter_item: Dict[str, Any], column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply a single filter to dataframe"""
        
        if filter_item['type'] == 'text_contains':
            field = filter_item.get('field', 'description')
            if field in column_mapping and column_mapping[field] in df.columns:
                column = column_mapping[field]
                pattern = filter_item['value']
                mask = df[column].astype(str).str.lower().str.contains(pattern.lower(), na=False, regex=False)
                return df[mask]
        
        elif filter_item['type'] == 'reference_equals':
            field = filter_item.get('field', 'reference')
            if field in column_mapping and column_mapping[field] in df.columns:
                column = column_mapping[field]
                value = filter_item['value']
                mask = df[column].astype(str).str.contains(str(value), na=False)
                return df[mask]
        
        elif filter_item['type'] in ['greater_than', 'less_than', 'equal_to']:
            # Numerical filters
            amount_column = column_mapping.get('amount', 'amount')
            if amount_column in df.columns:
                numeric_df = df.copy()
                # Ensure numeric column
                numeric_df[amount_column] = pd.to_numeric(numeric_df[amount_column], errors='coerce')
                
                value = filter_item['value']
                if filter_item['type'] == 'greater_than':
                    mask = numeric_df[amount_column] > value
                elif filter_item['type'] == 'less_than':
                    mask = numeric_df[amount_column] < value
                elif filter_item['type'] == 'equal_to':
                    mask = numeric_df[amount_column] == value
                
                return df[mask]
        
        elif filter_item['type'] == 'suspicious_description':
            # Apply suspicious description analysis
            description_column = column_mapping.get('description', 'description')
            if description_column in df.columns:
                analyzed_df = self.audit_analyzer.analyze_descriptions(df, description_column)
                if 'audit_risk_score' in analyzed_df.columns:
                    return analyzed_df[analyzed_df['audit_risk_score'] >= 0.4]
        
        elif filter_item['type'] == 'empty_description':
            description_column = column_mapping.get('description', 'description')
            if description_column in df.columns:
                mask = (df[description_column].isna() | 
                       (df[description_column].astype(str).str.strip() == '') |
                       (df[description_column].astype(str).str.lower().isin(['nan', 'null', 'none'])))
                return df[mask]
        
        return df
    
    def _get_sort_column(self, sort_by: str, column_mapping: Dict[str, str], df: pd.DataFrame) -> str:
        """Get the actual column name for sorting"""
        if sort_by == 'credit_amount_only' and 'credit_amount_only' in df.columns:
            return 'credit_amount_only'
        elif sort_by == 'debit_amount_only' and 'debit_amount_only' in df.columns:
            return 'debit_amount_only'
        elif sort_by in column_mapping:
            return column_mapping[sort_by]
        elif sort_by in df.columns:
            return sort_by
        return None