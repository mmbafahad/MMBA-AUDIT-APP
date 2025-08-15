# Overview

This is an Audit Transaction Sampling Tool built with Streamlit that allows auditors to input transaction data (via direct paste or file upload) and define sampling criteria using natural language. The application processes transaction data and applies intelligent filtering based on user-defined criteria to identify suspicious transactions or specific patterns for audit purposes. The tool incorporates NLP capabilities to interpret natural language queries and audit-specific analysis to detect red flags in transaction data.

## Recent Changes (August 15, 2025)
- **MAJOR UI REDESIGN**: Complete frontend overhaul with modern MMBA branding
- **NEW**: Added MMBA logo from official URL with professional gradient header design
- **NEW**: "Welcome to MMBA AUDIT SAMPLING TOOL" prominent branding message
- **RESTRUCTURED**: Moved data input section from sidebar to main area at top
- **ENHANCED**: Modern card-based UI with professional color scheme (#2a5298 blue theme)
- **NEW**: Step-by-step workflow with numbered sections (Data Input → Column Mapping → Criteria)
- **IMPROVED**: Horizontal radio buttons and enhanced visual hierarchy
- **ADDED**: Account filtering feature in prebuilt criteria system
- **ENHANCED**: Professional styling with gradients, shadows, and modern typography

## Previous Changes (August 2, 2025)
- Added direct data pasting functionality as primary input method
- Modified interface to support both pasted data and file upload
- Enhanced error handling for data processing
- Updated welcome screen to emphasize natural language criteria input
- **NEW**: Implemented complex criteria parser with AND/OR logical operators
- **NEW**: Added support for numerical filtering (e.g., "15 highest amount transactions")
- **NEW**: Enhanced text matching capabilities (e.g., "transactions with NashTech in description")
- **NEW**: Added reference number filtering (e.g., "transactions with reference 123, 125")
- Fixed credit-only and debit-only transaction filtering
- Enhanced suspicious description detection using ISA audit standards
- **FIXED**: OR operation duplicate handling - transactions matching multiple conditions now appear only once
- **FIXED**: Sample size extraction for "X highest/lowest" patterns now works correctly
- **FIXED**: Complex parser now returns all required fields for compatibility with main application
- **MAJOR UPDATE**: Added comprehensive prebuilt audit criteria system with:
  - High-Value Transactions (configurable number, type, threshold)
  - Missing Description detection
  - Suspicious Description scanning (ISA keywords)
  - Duplicate Transaction checks (by reference, amount, or both)
  - Round Amount detection (multiples of 100, 1000, or custom)
  - Backdated/Future-Dated transaction detection
- **NEW**: Dual criteria mode - users can choose between prebuilt criteria or custom natural language
- **NEW**: Selection reason tracking - each transaction shows why it was selected
- **NEW**: Combined criteria support - users can apply prebuilt + additional custom criteria
- **NEW**: Enhanced result display with criteria breakdown and statistics

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit web application with modern professional UI design
- **Branding**: MMBA-branded interface with official logo and corporate color scheme
- **Layout**: Wide layout with data input at top, step-by-step workflow design
- **Design System**: Professional card-based layout with gradients, shadows, and modern typography
- **Color Scheme**: Corporate blue theme (#2a5298) with professional gradients and styling
- **User Experience**: Step-numbered sections (Data Input → Column Mapping → Criteria Selection)
- **State Management**: Streamlit session state to persist uploaded data, column mappings, and processor instances across user interactions
- **File Upload**: Support for CSV and Excel files with automatic format detection and encoding handling

## Backend Architecture
- **Modular Design**: Utility classes organized in separate modules for specific functionalities
- **Data Processing Pipeline**: Sequential processing from file upload → data cleaning → column mapping → criteria interpretation → analysis
- **Core Components**:
  - `DataProcessor`: Handles file loading, format detection, and basic data cleaning
  - `NLPProcessor`: Natural language processing for criteria interpretation using NLTK and spaCy
  - `AuditAnalyzer`: Audit-specific analysis including red flag detection and suspicious pattern identification
  - `CriteriaInterpreter`: Orchestrates NLP processing and applies business logic for sampling criteria
  - **NEW**: `ComplexCriteriaParser`: Advanced parser for complex criteria with AND/OR logical operators, numerical filtering, text matching, and reference number filtering
  - **NEW**: `PrebuiltCriteriaProcessor`: Handles standard audit criteria including high-value transactions, duplicate detection, suspicious descriptions, round amounts, and date range validation

## Data Processing Strategy
- **File Format Support**: CSV (with multiple encoding fallbacks) and Excel formats
- **Data Cleaning**: Automatic removal of empty rows, whitespace trimming, and null value standardization
- **Column Mapping**: Dynamic mapping system to standardize transaction fields across different data sources
- **Memory Management**: Pandas DataFrames for efficient data manipulation of large transaction datasets

## NLP and Analysis Engine
- **Text Processing**: NLTK for tokenization, stopword removal, and basic text analysis
- **Advanced NLP**: Optional spaCy integration for more sophisticated language understanding
- **Pattern Recognition**: Regex-based pattern matching for audit red flags and suspicious transaction descriptions
- **Criteria Parsing**: Natural language interpretation to convert user queries into structured filtering parameters

# External Dependencies

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis library for transaction processing
- **NumPy**: Numerical computing support for data operations

## NLP Libraries
- **NLTK**: Natural Language Toolkit for text processing, tokenization, and stopword removal
- **spaCy**: Advanced NLP library for language understanding (optional fallback implementation)

## File Processing
- **openpyxl/xlrd**: Excel file reading capabilities through pandas
- **csv**: Built-in CSV processing with encoding detection

## Python Standard Library
- **re**: Regular expressions for pattern matching in transaction descriptions
- **io**: String and file I/O operations for data processing
- **json**: Data serialization for configuration and results

## Development Dependencies
- Standard Python libraries for string manipulation, file handling, and data structures
- No external databases or APIs required - operates entirely on uploaded file data