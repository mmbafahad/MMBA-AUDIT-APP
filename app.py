import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import json
from utils.data_processor import DataProcessor
from utils.nlp_processor import NLPProcessor
from utils.audit_analyzer import AuditAnalyzer
from utils.criteria_interpreter import CriteriaInterpreter

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

def main():
    st.set_page_config(
        page_title="Audit Transaction Sampling Tool",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Audit Transaction Sampling Tool")
    st.markdown("Upload transaction data and use natural language to define sampling criteria")
    
    # Sidebar for file upload and column mapping
    with st.sidebar:
        st.header("📁 Data Upload")
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
                
                # Column mapping section
                st.header("🗂️ Column Mapping")
                st.markdown("Map your columns to standard transaction fields:")
                
                columns = st.session_state.data.columns.tolist()
                
                # Standard fields mapping
                standard_fields = {
                    'amount': 'Transaction Amount',
                    'description': 'Transaction Description',
                    'date': 'Transaction Date',
                    'account': 'Account',
                    'reference': 'Reference Number',
                    'type': 'Transaction Type'
                }
                
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
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Main content area
    if st.session_state.data is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📊 Data Preview")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
            # Natural language criteria input
            st.header("🗣️ Sampling Criteria")
            st.markdown("Enter your sampling criteria in natural language:")
            
            # Example criteria
            with st.expander("💡 Example Criteria"):
                examples = [
                    "Select 10 highest value transactions",
                    "Find transactions with no description",
                    "Show suspicious or vague descriptions",
                    "Get duplicate transactions",
                    "Select transactions above 50000",
                    "Find entries with words like 'suspense', 'misc', or 'various'",
                    "Show transactions on weekends",
                    "Find round number amounts ending in 000"
                ]
                for example in examples:
                    st.code(example)
            
            criteria_text = st.text_area(
                "Enter criteria:",
                height=100,
                placeholder="e.g., Select 15 transactions with highest amounts and suspicious descriptions"
            )
            
            if st.button("🔍 Apply Sampling Criteria", type="primary"):
                if criteria_text.strip():
                    apply_sampling_criteria(criteria_text)
                else:
                    st.warning("⚠️ Please enter sampling criteria")
        
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
                    )
                    st.metric("Total Amount", f"${numeric_amounts.sum():,.2f}")
                    st.metric("Average Amount", f"${numeric_amounts.mean():.2f}")
                    st.metric("Max Amount", f"${numeric_amounts.max():,.2f}")
                except:
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
        - 📁 Upload CSV/Excel files with large transaction datasets
        - 🗂️ Flexible column mapping for different data formats
        - 🗣️ Natural language criteria input
        - 🔍 Intelligent description analysis for audit red flags
        - 📊 Smart sampling based on ISA and general audit rules
        - 📤 Export functionality for selected samples
        
        ### Get Started:
        1. Upload your transaction file using the sidebar
        2. Map your columns to standard transaction fields
        3. Enter your sampling criteria in plain English
        4. Review and export your selected samples
        """)

def apply_sampling_criteria(criteria_text):
    """Apply the natural language sampling criteria to the data"""
    try:
        with st.spinner("🔄 Processing criteria and analyzing transactions..."):
            # Parse criteria using NLP
            parsed_criteria = st.session_state.criteria_interpreter.parse_criteria(
                criteria_text, st.session_state.column_mapping
            )
            
            # Apply criteria to data
            sampled_data = st.session_state.criteria_interpreter.apply_criteria(
                st.session_state.data, 
                parsed_criteria, 
                st.session_state.column_mapping
            )
            
            # Analyze descriptions for audit red flags
            if 'description' in st.session_state.column_mapping:
                description_col = st.session_state.column_mapping['description']
                sampled_data = st.session_state.audit_analyzer.analyze_descriptions(
                    sampled_data, description_col
                )
        
        # Display results
        st.header("🎯 Sampling Results")
        
        if len(sampled_data) > 0:
            st.success(f"✅ Found {len(sampled_data)} transactions matching your criteria")
            
            # Show criteria interpretation
            st.subheader("🧠 Criteria Interpretation")
            interpretation_text = st.session_state.criteria_interpreter.get_interpretation_summary(parsed_criteria)
            st.info(interpretation_text)
            
            # Display sampled data
            st.subheader("📋 Selected Transactions")
            
            # Add reasoning column if available
            display_data = sampled_data.copy()
            if 'selection_reason' in display_data.columns:
                st.dataframe(display_data, use_container_width=True)
            else:
                st.dataframe(display_data, use_container_width=True)
            
            # Export functionality
            col1, col2 = st.columns(2)
            with col1:
                csv = display_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="audit_sample.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Convert to Excel
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    display_data.to_excel(writer, sheet_name='Audit_Sample', index=False)
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name="audit_sample.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Analysis summary
            if 'audit_risk_score' in display_data.columns:
                st.subheader("⚠️ Risk Analysis Summary")
                high_risk = len(display_data[display_data['audit_risk_score'] >= 0.7])
                medium_risk = len(display_data[(display_data['audit_risk_score'] >= 0.4) & (display_data['audit_risk_score'] < 0.7)])
                low_risk = len(display_data[display_data['audit_risk_score'] < 0.4])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔴 High Risk", high_risk)
                with col2:
                    st.metric("🟡 Medium Risk", medium_risk)
                with col3:
                    st.metric("🟢 Low Risk", low_risk)
        else:
            st.warning("⚠️ No transactions found matching your criteria. Try adjusting your criteria.")
            
    except Exception as e:
        st.error(f"❌ Error processing criteria: {str(e)}")
        st.error("Please check your criteria format and try again.")

if __name__ == "__main__":
    main()
