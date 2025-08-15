import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import json
import base64
import requests
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


def load_logo_from_url():
    """Load the MMBA logo from URL and convert to base64"""
    try:
        response = requests.get("https://cdn-ljgdn.nitrocdn.com/YpcpPwpYltqokViveZpKXQvSMrMvdYLR/assets/images/optimized/rev-84fe550/www.mmba.co.uk/wp-content/uploads/2023/08/Logo-MMBA-WP.png")
        if response.status_code == 200:
            return base64.b64encode(response.content).decode()
    except:
        pass
    return None

def display_header():
    """Display the MMBA header with logo and welcome message"""
    logo_base64 = load_logo_from_url()
    
    # Custom CSS for modern design
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            padding: 2rem 1rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem;
        }
        .welcome-text {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle-text {
            font-size: 1.2rem;
            opacity: 0.9;
            font-weight: 300;
        }
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border-left: 4px solid #2a5298;
        }
        .section-header {
            background: linear-gradient(90deg, #f8f9fa, #e9ecef);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
            margin-bottom: 1rem;
        }
        .data-input-section {
            background: #f8f9fa;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            border: 1px solid #dee2e6;
        }
        .criteria-section {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }
        .step-number {
            background: #2a5298;
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo
    header_html = f"""
    <div class="main-header">
        <div class="logo-container">
            {f'<img src="data:image/png;base64,{logo_base64}" style="height: 80px; margin-right: 20px;">' if logo_base64 else ''}
        </div>
        <div class="welcome-text">Welcome to MMBA AUDIT SAMPLING TOOL</div>
        <div class="subtitle-text">Advanced AI-Powered Transaction Analysis & Audit Sampling</div>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)

def display_data_input_section():
    """Display the data input section at the top"""
    
    # Data input in main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="data-input-section">', unsafe_allow_html=True)
        
        # Data input methods
        input_method = st.radio(
            "Choose input method:",
            ["📋 Paste Transaction Data", "📁 Upload File"],
            horizontal=True,
            help="Paste data directly or upload CSV/Excel file"
        )
        
        if input_method == "📋 Paste Transaction Data":
            st.markdown("**Paste Your Transaction Data**")
            st.info("""
            📌 **Instructions:**
            • Copy data directly from Excel, accounting software, or CSV file
            • Include column headers in the first row  
            • Ensure data includes transaction amounts, descriptions, and dates
            """)
            
            pasted_data = st.text_area(
                "Paste your transaction data here:",
                height=150,
                placeholder="Date,Description,Amount,Reference\n2024-01-01,Payment to supplier,1500.00,INV001\n2024-01-02,Office supplies,-250.00,EXP002"
            )
            
            if st.button("📥 Process Pasted Data", type="primary") and pasted_data.strip():
                try:
                    data_processor = DataProcessor()
                    st.session_state.data = data_processor.load_pasted_data(pasted_data)
                    st.success(f"✅ Loaded {len(st.session_state.data):,} transactions with {len(st.session_state.data.columns)} columns")
                    st.dataframe(st.session_state.data.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
                    
        else:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel files"
            )
            
            if uploaded_file:
                try:
                    data_processor = DataProcessor()
                    st.session_state.data = data_processor.load_file(uploaded_file)
                    st.success(f"✅ Loaded {len(st.session_state.data):,} transactions with {len(st.session_state.data.columns)} columns")
                    st.dataframe(st.session_state.data.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.data is not None:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Data Summary</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Total Transactions", f"{len(st.session_state.data):,}")
            st.metric("Columns", len(st.session_state.data.columns))
            
            # Show column types
            st.markdown("**Column Types:**")
            for col in st.session_state.data.columns[:5]:  # Show first 5 columns
                st.text(f"• {col}")

def display_column_mapping_section():
    """Display column mapping section"""
    if st.session_state.data is None:
        return
    
    st.markdown('<div class="criteria-section">', unsafe_allow_html=True)
    
    available_columns = list(st.session_state.data.columns)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.column_mapping['date'] = st.selectbox(
            "📅 Date Column", 
            [""] + available_columns,
            index=0 if 'date' not in st.session_state.column_mapping else 
                  available_columns.index(st.session_state.column_mapping['date']) + 1 
                  if st.session_state.column_mapping.get('date') in available_columns else 0
        )
        
        st.session_state.column_mapping['amount'] = st.selectbox(
            "💰 Amount Column", 
            [""] + available_columns,
            index=0 if 'amount' not in st.session_state.column_mapping else 
                  available_columns.index(st.session_state.column_mapping['amount']) + 1 
                  if st.session_state.column_mapping.get('amount') in available_columns else 0
        )
        
    with col2:
        st.session_state.column_mapping['description'] = st.selectbox(
            "📝 Description Column", 
            [""] + available_columns,
            index=0 if 'description' not in st.session_state.column_mapping else 
                  available_columns.index(st.session_state.column_mapping['description']) + 1 
                  if st.session_state.column_mapping.get('description') in available_columns else 0
        )
        
        st.session_state.column_mapping['reference'] = st.selectbox(
            "🔗 Reference Column", 
            [""] + available_columns,
            index=0 if 'reference' not in st.session_state.column_mapping else 
                  available_columns.index(st.session_state.column_mapping['reference']) + 1 
                  if st.session_state.column_mapping.get('reference') in available_columns else 0
        )
    
    with col3:
        st.session_state.column_mapping['account'] = st.selectbox(
            "🏢 Account Column", 
            [""] + available_columns,
            index=0 if 'account' not in st.session_state.column_mapping else 
                  available_columns.index(st.session_state.column_mapping['account']) + 1 
                  if st.session_state.column_mapping.get('account') in available_columns else 0
        )
        
        # Optional: Separate debit/credit columns
        with st.expander("Separate Debit/Credit Columns"):
            st.session_state.column_mapping['debit'] = st.selectbox(
                "Debit Column (Optional)", 
                [""] + available_columns,
                key="debit_col"
            )
            st.session_state.column_mapping['credit'] = st.selectbox(
                "Credit Column (Optional)", 
                [""] + available_columns,
                key="credit_col"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_criteria_selection():
    """Display criteria selection with modern design"""
    if not st.session_state.column_mapping or st.session_state.data is None:
        return
    
    # Criteria mode selection
    criteria_mode = st.radio(
        "Select your analysis approach:",
        ["🏛️ Standard Audit Criteria", "🤖 Custom Natural Language"],
        horizontal=True,
        help="Standard criteria provide ISA-compliant audit checks, while custom allows flexible natural language queries"
    )
    
    if criteria_mode == "🏛️ Standard Audit Criteria":
        display_prebuilt_criteria_ui()
    else:
        display_custom_criteria_ui()

def display_prebuilt_criteria_ui():
    """Display the prebuilt criteria configuration UI"""
    st.markdown('<div class="criteria-section">', unsafe_allow_html=True)
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
    if st.button("🔍 Apply Standard Audit Criteria", type="primary"):
        apply_prebuilt_criteria()
    
    # Add custom criteria option
    st.markdown("---")
    st.subheader("➕ Additional Custom Criteria")
    st.markdown("Add custom criteria in addition to the standard ones:")
    
    additional_criteria = st.text_area(
        "Additional criteria (optional):",
        height=80,
        placeholder="e.g., transactions with 'transfer' in description OR amount > 75000"
    )
    
    if st.button("🔍 Apply Standard + Custom Criteria", type="secondary"):
        apply_prebuilt_criteria(additional_criteria.strip() if additional_criteria.strip() else None)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_custom_criteria_ui():
    """Display the custom natural language criteria UI"""
    st.markdown('<div class="criteria-section">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)

# Include all other existing functions from the original app.py
# (apply_prebuilt_criteria, apply_sampling_criteria, display_results, etc.)

def apply_prebuilt_criteria(additional_criteria=None):
    """Apply prebuilt audit criteria with optional additional custom criteria"""
    try:
        with st.spinner("Processing standard audit criteria..."):
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
            display_results(final_results, criteria_stats, "Standard Audit Criteria")
            
    except Exception as e:
        st.error(f"Error applying standard criteria: {str(e)}")
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
            file_name=f"mmba_audit_sample_{criteria_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON export for detailed analysis
        json_data = sampled_data.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"mmba_audit_sample_{criteria_type.lower().replace(' ', '_')}.json",
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
                
                # Apply complex criteria
                sampled_data = st.session_state.complex_parser.apply_complex_criteria(
                    processed_data, parsed_criteria, working_column_mapping
                )
            else:
                # Use simple criteria parser
                parsed_criteria = st.session_state.criteria_interpreter.parse_criteria(
                    criteria_text, working_column_mapping
                )
                
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

def main():
    st.set_page_config(
        page_title="MMBA Audit Sampling Tool",
        page_icon="🏢",
        layout="wide"
    )
    
    # Display MMBA header
    display_header()
    
    # Data input section at the top
    display_data_input_section()
    
    # Column mapping section
    display_column_mapping_section()
    
    # Criteria selection and analysis
    display_criteria_selection()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-top: 3rem; border-top: 1px solid #dee2e6; color: #666; font-size: 0.9rem;">
        <p>Copyright MMBA ACCOUNTANT</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()