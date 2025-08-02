import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import json
from utils.data_processor import DataProcessor
from utils.nlp_processor import NLPProcessor
from utils.audit_analyzer import AuditAnalyzer
from utils.criteria_interpreter import CriteriaInterpreter
from utils.complex_criteria_parser import ComplexCriteriaParser
from utils.prebuilt_criteria import PrebuiltCriteriaProcessor

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'nlp_processor' not in st.session_state:
    st.session_state.nlp_processor = NLPProcessor()
if 'audit_analyzer' not in st.session_state:
    st.session_state.audit_analyzer = AuditAnalyzer()
if 'criteria_interpreter' not in st.session_state:
    st.session_state.criteria_interpreter = CriteriaInterpreter()
if 'complex_parser' not in st.session_state:
    st.session_state.complex_parser = ComplexCriteriaParser()
if 'prebuilt_processor' not in st.session_state:
    st.session_state.prebuilt_processor = PrebuiltCriteriaProcessor(st.session_state.audit_analyzer)

def main():
    st.set_page_config(
        page_title="Audit Transaction Sampling Tool",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Audit Transaction Sampling Tool")
    st.markdown("Upload transaction data and use natural language to define sampling criteria")
    
    # Sidebar for data input and column mapping
    with st.sidebar:
        st.header("📊 Data Input")
        
        # Data input methods
        input_method = st.radio(
            "Choose input method:",
            ["Paste Transaction Data", "Upload File"],
            help="Paste data directly or upload CSV/Excel file"
        )
        
        if input_method == "Paste Transaction Data":
            st.subheader("📋 Paste Your Transaction Data")
            st.markdown("""
            **Instructions:**
            1. Copy data directly from Excel, accounting software, or CSV file
            2. Include column headers in the first row
            3. The tool supports tab, comma, or space-separated data
            """)
            
            with st.expander("💡 Supported Data Formats"):
                st.code("""
Tab-separated (Excel copy-paste):
Date    Amount  Description     Account
2024-01-01      1000.00 Office Supplies Cash
2024-01-02      2500.00 Suspense Entry  Bank

Comma-separated:
Date,Amount,Description,Account
2024-01-01,1000.00,Office Supplies,Cash
2024-01-02,2500.00,Suspense Entry,Bank

Fixed-width (accounting reports):
Date       Amount    Description        Account
2024-01-01  1000.00  Office Supplies   Cash
2024-01-02  2500.00  Suspense Entry    Bank
                """)
            
            pasted_data = st.text_area(
                "Transaction Data:",
                height=300,
                placeholder="Paste your transaction data here...\n\nTip: Copy directly from Excel or your accounting system",
                help="Paste data with headers - supports multiple formats"
            )
            
            if st.button("📥 Process Pasted Data") and pasted_data.strip():
                try:
                    data_processor = DataProcessor()
                    st.session_state.data = data_processor.load_pasted_data(pasted_data)
                    st.success(f"✅ Loaded {len(st.session_state.data):,} transactions with {len(st.session_state.data.columns)} columns")
                    
                    # Show a preview of the loaded data
                    st.subheader("📊 Data Preview")
                    st.dataframe(st.session_state.data.head(), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
                    
                    # Show debug information
                    with st.expander("🔍 Debug Information"):
                        lines = pasted_data.split('\n')[:5]  # Show first 5 lines
                        st.text("First few lines of your data:")
                        for i, line in enumerate(lines):
                            st.code(f"Line {i+1}: {repr(line)}")
                        
                        st.markdown("""
                        **Troubleshooting Tips:**
                        - Ensure your data has column headers in the first row
                        - Check that columns are separated by tabs, commas, or consistent spacing
                        - Try copying smaller sections if the data is very large
                        - Remove any extra formatting or merged cells from Excel before copying
                        """)
        
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload transaction data with 10,000+ records"
            )
            
            if uploaded_file is not None:
                try:
                    data_processor = DataProcessor()
                    st.session_state.data = data_processor.load_file(uploaded_file)
                    st.success(f"✅ Loaded {len(st.session_state.data):,} transactions")
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        # Column mapping section (show only if data is loaded)
        if st.session_state.data is not None:
            st.header("🗂️ Column Mapping")
            st.markdown("Map your columns to standard transaction fields:")
            
            columns = st.session_state.data.columns.tolist()
            
            # Standard fields mapping
            standard_fields = {
                'amount': 'Transaction Amount (or leave None if using Debit/Credit)',
                'debit': 'Debit Amount',
                'credit': 'Credit Amount', 
                'description': 'Transaction Description',
                'date': 'Transaction Date',
                'account': 'Account',
                'reference': 'Reference Number',
                'type': 'Transaction Type'
            }
            
            st.info("💡 Tip: For debit/credit data, map both columns. You can then specify 'credit amounts only' or 'debit amounts only' in your criteria.")
            
            for field, label in standard_fields.items():
                st.session_state.column_mapping[field] = st.selectbox(
                    label,
                    options=['None'] + columns,
                    key=f"mapping_{field}",
                    index=0
                )
            
            # Remove 'None' mappings
            st.session_state.column_mapping = {
                k: v for k, v in st.session_state.column_mapping.items() 
                if v != 'None'
            }
    
    # Main content area
    if st.session_state.data is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📊 Data Preview")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
            # Criteria selection mode
            st.header("🗣️ Sampling Criteria")
            
            criteria_mode = st.radio(
                "Choose criteria mode:",
                ["Prebuilt Audit Criteria", "Custom Natural Language Criteria"],
                help="Use prebuilt criteria for standard audit checks or enter custom criteria"
            )
            
            if criteria_mode == "Prebuilt Audit Criteria":
                display_prebuilt_criteria_ui()
            else:
                display_custom_criteria_ui()
        
        with col2:
            st.header("📈 Data Summary")
            
            # Basic statistics
            total_records = len(st.session_state.data)
            st.metric("Total Records", f"{total_records:,}")
            
            if 'amount' in st.session_state.column_mapping:
                amount_col = st.session_state.column_mapping['amount']
                try:
                    numeric_amounts = pd.to_numeric(
                        st.session_state.data[amount_col], 
                        errors='coerce'
                    ).dropna()
                    
                    if len(numeric_amounts) > 0:
                        st.metric("Total Amount", f"${numeric_amounts.sum():,.2f}")
                        st.metric("Average Amount", f"${numeric_amounts.mean():.2f}")
                        st.metric("Max Amount", f"${numeric_amounts.max():,.2f}")
                    else:
                        st.info("Amount column needs numeric values")
                except Exception:
                    st.info("Amount column needs numeric values")
            
            # Column mapping status
            st.subheader("🗂️ Mapped Fields")
            for field, column in st.session_state.column_mapping.items():
                st.text(f"{field.title()}: {column}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Audit Transaction Sampling Tool
        
        This tool helps auditors efficiently sample transaction data using natural language criteria.
        
        ### Features:
        - 📋 **Paste transaction data directly** or upload CSV/Excel files
        - 🗂️ Flexible column mapping for different data formats
        - 🗣️ **Complex natural language criteria** with AND/OR logical operators
        - 🔢 **Numerical filtering** (e.g., "15 highest amount transactions")
        - 📝 **Text matching** (e.g., "transactions with NashTech in description") 
        - 🔗 **Reference number filtering** (e.g., "transactions with reference 123, 125")
        - 🔍 Intelligent description analysis for audit red flags
        - 📊 Smart sampling based on ISA and general audit rules
        - 📤 Export functionality for selected samples
        
        ### Get Started:
        1. **Paste your transaction data** in the sidebar or upload a file
        2. Map your columns to standard transaction fields  
        3. **Enter your sampling criteria in natural language**
        4. Review and export your selected samples
        
        ### Example Natural Language Criteria:
        - "Select 10 highest value transactions"
        - "Find transactions with suspicious descriptions"
        - "Show entries with no description or vague descriptions"
        - "Get duplicate transactions"
        - "Select transactions above 50,000"
        """)

def display_prebuilt_criteria_ui():
    """Display the prebuilt criteria configuration UI"""
    st.markdown("Configure standard audit criteria automatically applied to your data:")
    
    # Initialize criteria config in session state
    if 'criteria_config' not in st.session_state:
        st.session_state.criteria_config = {
            'high_value_enabled': True,
            'missing_desc_enabled': True,
            'suspicious_desc_enabled': True,
            'duplicate_enabled': True,
            'round_amount_enabled': True,
            'date_check_enabled': False,
            'selected_accounts': [],
        }
    
    # Account Selection (only if account column is mapped)
    if 'account' in st.session_state.column_mapping:
        st.subheader("🏢 Account Filter")
        
        # Get unique accounts from the data
        unique_accounts = st.session_state.prebuilt_processor.get_unique_accounts(
            st.session_state.data, st.session_state.column_mapping
        )
        
        if unique_accounts:
            selected_accounts = st.multiselect(
                "Select specific accounts to analyze (leave empty to analyze all accounts):",
                options=unique_accounts,
                default=st.session_state.criteria_config.get('selected_accounts', []),
                help=f"Found {len(unique_accounts)} unique accounts in your data"
            )
            st.session_state.criteria_config['selected_accounts'] = selected_accounts
            
            if selected_accounts:
                st.info(f"Analysis will be limited to {len(selected_accounts)} selected account(s)")
            else:
                st.info("Analysis will be applied to all accounts in the dataset")
        else:
            st.warning("No account data found in the mapped account column")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 High-Value Transactions")
        st.session_state.criteria_config['high_value_enabled'] = st.checkbox(
            "Enable high-value transaction check", 
            value=st.session_state.criteria_config['high_value_enabled']
        )
        
        if st.session_state.criteria_config['high_value_enabled']:
            num_transactions = st.number_input("Number of transactions:", min_value=1, max_value=100, value=10)
            transaction_type = st.selectbox("Transaction type:", ["both", "credit", "debit"])
            threshold_amount = st.number_input("Threshold amount (optional):", min_value=0.0, value=0.0, step=1000.0)
            
            st.session_state.criteria_config['high_value_config'] = {
                'num_transactions': num_transactions,
                'transaction_type': transaction_type,
                'threshold_amount': threshold_amount if threshold_amount > 0 else None
            }
        
        st.subheader("📝 Description Checks")
        st.session_state.criteria_config['missing_desc_enabled'] = st.checkbox(
            "Missing descriptions", 
            value=st.session_state.criteria_config['missing_desc_enabled'],
            help="Automatically detect transactions with empty descriptions"
        )
        
        st.session_state.criteria_config['suspicious_desc_enabled'] = st.checkbox(
            "Suspicious descriptions", 
            value=st.session_state.criteria_config['suspicious_desc_enabled'],
            help="Scan for suspicious keywords like 'adjustment', 'correction', 'reversal', etc."
        )
        
        st.subheader("📅 Date Range Check")
        st.session_state.criteria_config['date_check_enabled'] = st.checkbox(
            "Enable date range check", 
            value=st.session_state.criteria_config['date_check_enabled']
        )
        
        if st.session_state.criteria_config['date_check_enabled']:
            expected_start = st.date_input("Expected start date:")
            expected_end = st.date_input("Expected end date:")
            
            st.session_state.criteria_config['date_config'] = {
                'expected_start_date': expected_start,
                'expected_end_date': expected_end
            }
    
    with col2:
        st.subheader("🔄 Duplicate Transaction Check")
        st.session_state.criteria_config['duplicate_enabled'] = st.checkbox(
            "Enable duplicate check", 
            value=st.session_state.criteria_config['duplicate_enabled']
        )
        
        if st.session_state.criteria_config['duplicate_enabled']:
            duplicate_options = st.multiselect(
                "Check duplicates by:",
                ["reference", "amount"],
                default=["reference", "amount"],
                help="Choose which fields to use for duplicate detection"
            )
            
            st.session_state.criteria_config['duplicate_config'] = {
                'duplicate_by': duplicate_options
            }
        
        st.subheader("⭕ Round Amount Check")
        st.session_state.criteria_config['round_amount_enabled'] = st.checkbox(
            "Enable round amount check", 
            value=st.session_state.criteria_config['round_amount_enabled']
        )
        
        if st.session_state.criteria_config['round_amount_enabled']:
            roundness_type = st.selectbox("Check for multiples of:", ["100", "1000", "custom"])
            
            if roundness_type == "custom":
                custom_threshold = st.number_input("Custom threshold:", min_value=1, value=500)
                st.session_state.criteria_config['round_amount_config'] = {
                    'roundness_type': 'custom',
                    'custom_threshold': custom_threshold
                }
            else:
                st.session_state.criteria_config['round_amount_config'] = {
                    'roundness_type': roundness_type
                }
    
    # Apply button for prebuilt criteria
    if st.button("🔍 Apply Prebuilt Criteria", type="primary"):
        apply_prebuilt_criteria()
    
    # Add custom criteria option
    st.markdown("---")
    st.subheader("➕ Additional Custom Criteria")
    st.markdown("Add custom criteria in addition to the prebuilt ones:")
    
    additional_criteria = st.text_area(
        "Additional criteria (optional):",
        height=80,
        placeholder="e.g., transactions with 'transfer' in description OR amount > 75000"
    )
    
    if st.button("🔍 Apply Prebuilt + Custom Criteria", type="secondary"):
        apply_prebuilt_criteria(additional_criteria.strip() if additional_criteria.strip() else None)

def display_custom_criteria_ui():
    """Display the custom natural language criteria UI"""
    st.markdown("Enter your sampling criteria in natural language:")
    
    # Example criteria
    with st.expander("💡 Example Complex Criteria"):
        st.markdown("**Simple Criteria:**")
        st.code("Select 15 highest amount transactions")
        st.code("Find transactions with NashTech in description")
        st.code("Show transactions with reference 123")
        
        st.markdown("**Complex Criteria with AND/OR:**")
        st.code("15 highest amount transactions AND transactions with NashTech in description")
        st.code("transactions with reference 123 OR reference 125")
        st.code("amount > 50000 AND description contains 'misc'")
        st.code("credit amounts above 25000 OR debit amounts above 30000")
        
        st.markdown("**Advanced Examples:**")
        st.code("10 highest credit amounts AND (description contains 'payment' OR description contains 'transfer')")
        st.code("transactions with suspicious descriptions OR amount > 100000")
        st.code("reference 456 OR reference 789 AND amount > 10000")
    
    criteria_text = st.text_area(
        "Enter criteria:",
        height=100,
        placeholder="e.g., Select 15 transactions with highest amounts and suspicious descriptions"
    )
    
    if st.button("🔍 Apply Custom Criteria", type="primary"):
        if criteria_text.strip():
            apply_sampling_criteria(criteria_text)
        else:
            st.warning("Please enter sampling criteria")

def apply_prebuilt_criteria(additional_criteria=None):
    """Apply prebuilt audit criteria with optional additional custom criteria"""
    try:
        with st.spinner("Processing prebuilt audit criteria..."):
            # Process debit/credit columns if mapped
            processed_data = st.session_state.data.copy()
            working_column_mapping = st.session_state.column_mapping.copy()
            
            if 'debit' in st.session_state.column_mapping and 'credit' in st.session_state.column_mapping:
                data_processor = DataProcessor()
                processed_data, net_amount_col = data_processor.process_debit_credit_columns(
                    processed_data,
                    st.session_state.column_mapping['debit'],
                    st.session_state.column_mapping['credit']
                )
                if net_amount_col:
                    working_column_mapping['amount'] = net_amount_col
                    st.info("Using combined debit/credit amounts for analysis")
            
            # Apply prebuilt criteria
            prebuilt_results, criteria_stats = st.session_state.prebuilt_processor.apply_all_prebuilt_criteria(
                processed_data, working_column_mapping, st.session_state.criteria_config
            )
            
            # Apply additional custom criteria if provided
            if additional_criteria:
                st.info(f"Processing additional custom criteria: {additional_criteria}")
                
                # Determine if additional criteria is complex
                is_complex = any(op in additional_criteria.upper() for op in [' AND ', ' OR '])
                
                if is_complex:
                    parsed_criteria = st.session_state.complex_parser.parse_complex_criteria(
                        additional_criteria, working_column_mapping
                    )
                    custom_results = st.session_state.complex_parser.apply_complex_criteria(
                        processed_data, parsed_criteria, working_column_mapping
                    )
                else:
                    parsed_criteria = st.session_state.criteria_interpreter.parse_criteria(
                        additional_criteria, working_column_mapping
                    )
                    custom_results = st.session_state.criteria_interpreter.apply_criteria(
                        processed_data, parsed_criteria, working_column_mapping
                    )
                
                # Combine prebuilt and custom results
                if len(prebuilt_results) > 0 and len(custom_results) > 0:
                    # Add selection reason for custom results if not present
                    if 'selection_reason' not in custom_results.columns:
                        custom_results['selection_reason'] = f"Custom Criteria: {additional_criteria}"
                    
                    # Combine and remove duplicates while preserving reasons
                    combined_results = pd.concat([prebuilt_results, custom_results], ignore_index=True)
                    
                    # Group by all columns except selection_reason and combine reasons
                    group_columns = [col for col in combined_results.columns if col != 'selection_reason']
                    def combine_reasons(group):
                        reasons = group['selection_reason'].unique()
                        return ' | '.join(reasons)
                    
                    final_results = combined_results.groupby(group_columns, as_index=False).agg({
                        'selection_reason': combine_reasons
                    })
                elif len(custom_results) > 0:
                    final_results = custom_results
                    if 'selection_reason' not in final_results.columns:
                        final_results['selection_reason'] = f"Custom Criteria: {additional_criteria}"
                else:
                    final_results = prebuilt_results
            else:
                final_results = prebuilt_results
            
            # Display results
            display_results(final_results, criteria_stats, "Prebuilt Audit Criteria")
            
    except Exception as e:
        st.error(f"Error applying prebuilt criteria: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

def display_results(sampled_data, criteria_stats=None, criteria_type="Custom Criteria"):
    """Display sampling results with statistics and export options"""
    if len(sampled_data) == 0:
        st.warning("No transactions match your criteria")
        return
    
    # Results header with statistics
    st.success(f"Found {len(sampled_data):,} transactions matching your criteria")
    
    # Criteria interpretation info
    if criteria_stats:
        st.subheader("Criteria Breakdown")
        for criteria_name, count in criteria_stats.items():
            st.metric(criteria_name, f"{count:,} transactions")
    
    # Display results
    st.subheader("Selected Transactions")
    
    # Show selection reasons if available
    if 'selection_reason' in sampled_data.columns:
        # Group by selection reason to show breakdown
        reason_counts = sampled_data['selection_reason'].value_counts()
        st.subheader("Selection Reasons")
        for reason, count in reason_counts.items():
            st.write(f"• **{reason}**: {count:,} transactions")
    
    # Display data with enhanced formatting
    display_columns = [col for col in sampled_data.columns if col != 'selection_reason']
    if 'selection_reason' in sampled_data.columns:
        display_columns.append('selection_reason')
    
    st.dataframe(
        sampled_data[display_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Export functionality
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_data = sampled_data.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"audit_sample_{criteria_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export for detailed analysis
        json_data = sampled_data.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"audit_sample_{criteria_type.lower().replace(' ', '_')}.json",
            mime="application/json"
        )

def apply_sampling_criteria(criteria_text):
    """Apply the natural language sampling criteria to the data"""
    try:
        with st.spinner("🔄 Processing criteria and analyzing transactions..."):
            # Process debit/credit columns if mapped
            processed_data = st.session_state.data.copy()
            working_column_mapping = st.session_state.column_mapping.copy()
            
            if 'debit' in st.session_state.column_mapping and 'credit' in st.session_state.column_mapping:
                data_processor = DataProcessor()
                processed_data, net_amount_col = data_processor.process_debit_credit_columns(
                    processed_data,
                    st.session_state.column_mapping['debit'],
                    st.session_state.column_mapping['credit']
                )
                if net_amount_col:
                    working_column_mapping['amount'] = net_amount_col
                    st.info("💡 Using combined debit/credit amounts for analysis")
            
            # Determine if this is a complex criteria (contains AND/OR)
            is_complex = any(op in criteria_text.upper() for op in [' AND ', ' OR '])
            
            if is_complex:
                # Use complex criteria parser
                st.info(f"🔄 Processing complex criteria with logical operators...")
                parsed_criteria = st.session_state.complex_parser.parse_complex_criteria(
                    criteria_text, working_column_mapping
                )
                
                # Debug output for complex queries
                if parsed_criteria.get('logical_structure'):
                    conditions_count = len(parsed_criteria['logical_structure']['conditions'])
                    operators = ', '.join(parsed_criteria['logical_structure']['operators'])
                    st.info(f"🔍 Complex query detected: {conditions_count} conditions with operators: {operators}")
                    
                    # Show individual conditions for debugging
                    for i, condition in enumerate(parsed_criteria['logical_structure']['conditions']):
                        st.info(f"Condition {i+1}: Type={condition.get('type')}, Filters={len(condition.get('filters', []))}, Sample Size={condition.get('sample_size')}, Text={condition.get('text', '')[:50]}...")
                
                # Apply complex criteria
                sampled_data = st.session_state.complex_parser.apply_complex_criteria(
                    processed_data, parsed_criteria, working_column_mapping
                )
            else:
                # Use simple criteria parser
                parsed_criteria = st.session_state.criteria_interpreter.parse_criteria(
                    criteria_text, working_column_mapping
                )
                
                # Debug output for simple queries
                if parsed_criteria.get('transaction_type_filter') == 'credit':
                    st.info(f"🔍 Credit-only query. Sort by: {parsed_criteria.get('sort_by')}, Sample size: {parsed_criteria.get('sample_size')}")
                
                if parsed_criteria.get('text_patterns'):
                    st.info(f"🔍 Text patterns found: {parsed_criteria.get('text_patterns')}")
                
                # Apply simple criteria
                sampled_data = st.session_state.criteria_interpreter.apply_criteria(
                    processed_data, parsed_criteria, working_column_mapping
                )
            
            # Analyze descriptions for audit red flags (conditional based on criteria type)
            if 'description' in working_column_mapping:
                description_col = working_column_mapping['description'] 
                # Only analyze if we have description-related criteria or general analysis is needed
                if (not is_complex or 
                    any('description' in str(cond.get('type', '')) or 'text' in str(cond.get('type', '')) 
                        for cond in parsed_criteria.get('conditions', []))):
                    sampled_data = st.session_state.audit_analyzer.analyze_descriptions(
                        sampled_data, description_col
                    )
        
        # Add selection reason if not present
        if 'selection_reason' not in sampled_data.columns and len(sampled_data) > 0:
            sampled_data['selection_reason'] = f"Custom Criteria: {criteria_text}"
        
        # Display results using standardized function
        st.header("🎯 Sampling Results")
        display_results(sampled_data, None, "Custom Criteria")
        
        # Show criteria interpretation
        if len(sampled_data) > 0:
            st.subheader("🧠 Criteria Interpretation")
            interpretation_text = st.session_state.criteria_interpreter.get_interpretation_summary(parsed_criteria)
            st.info(interpretation_text)
            
    except Exception as e:
        st.error(f"❌ Error processing criteria: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.error(f"❌ Error processing criteria: {str(e)}")
        st.error("Please check your criteria format and try again.")

if __name__ == "__main__":
    main()
