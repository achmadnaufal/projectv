import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

def set_page_config():
    st.set_page_config(
        page_title="Coupon Optimization System",
        page_icon="🎯",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stHeader {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .metric-card {
            padding: 1rem;
            background-color: #ffffff;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        .info-box {
            padding: 1rem;
            background-color: #e8f0fe;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def create_system_overview():
    st.title("🎯 Coupon Distribution Optimization System")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        ## System Overview
        This advanced optimization system combines machine learning and linear programming 
        to maximize coupon acceptance rates while respecting business constraints.
        
        ### Key Features
        - Dynamic constraint management for venues, times, and weather
        - Customer segmentation with 567 unique profiles
        - Real-time optimization with multiple business rules
        - Contextual analysis across 121 different scenarios
        """)
        
        st.markdown("""
             ## Data Context & Assumptions
            
            This analysis is based on a comprehensive survey conducted through Amazon Mechanical Turk, featuring:
            - 652 verified respondents with 95%+ Turker rating
            - 20 unique scenarios per respondent
            - Total of 12,684 decision points collected
            
            ### Key Assumptions:
            1. The data represents actual decision-making scenarios as respondents were:
               • Carefully selected based on their high Turker rating (95%+)
               • Asked about specific, realistic driving scenarios
               • Required to provide consistent responses across scenarios
            
            2. Survey Quality Controls:
               • Each respondent answered varied scenarios to avoid bias
               • Responses were validated for consistency and completeness
               • Survey design captured real-world decision factors
            
            3. Data Validity:
               • Responses reflect genuine preferences and behaviors
               • Scenario coverage is comprehensive and representative
               • Sample size is statistically significant for analysis
        """)
    
    with col2:
        # Key metrics display
        metrics = {
            "Customer Groups": "567",
            "Scenario Types": "121",
            "Coupon Types": "10",
            "Model Accuracy": "75.1%"
        }
        
        for metric, value in metrics.items():
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>{metric}</h4>
                    <h2>{value}</h2>
                </div>
            """, unsafe_allow_html=True)

def create_model_performance_section():
    st.header("Model Performance Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overall F1 Score",
            value="0.74",
            delta="Balanced Performance"
        )
        
    with col2:
        st.metric(
            label="Acceptance Rate F1",
            value="0.78",
            delta="Strong Positive Detection"
        )
        
    with col3:
        st.metric(
            label="Rejection Rate F1",
            value="0.68",
            delta="Good Negative Detection"
        )

def create_optimization_features():
    st.header("Optimization Features")
    
    tab1, tab2, tab3 = st.tabs(["Venue Distribution", "Time Windows", "Weather Conditions"])
    
    with tab1:
        st.markdown("""
        ### Venue-based Optimization
        - Separate allocation limits for bars, restaurants, and coffee shops
        - Dynamic adjustment based on business capacity
        - Balance between different venue types
        - Customizable distribution targets
        """)
        
    with tab2:
        st.markdown("""
        ### Time-based Distribution
        - Morning (7AM-11AM)
        - Afternoon (2PM)
        - Evening (6PM-10PM)
        - Customizable time window allocations
        - Peak hour optimization
        """)
        
    with tab3:
        st.markdown("""
        ### Weather-based Targeting
        - Sunny conditions optimization
        - Rainy weather adjustments
        - Snow condition handling
        - Weather-specific targeting rules
        - Seasonal adaptations
        """)

def create_technical_details():
    st.header("Technical Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Machine Learning Component
        - Random Forest Classifier
        - Best Parameters:
          - max_depth: 20
          - min_samples_leaf: 1
          - min_samples_split: 2
          - n_estimators: 200
        - Cross-validation accuracy: 75.1%
        """)
        
    with col2:
        st.markdown("""
        ### Optimization Engine
        - PuLP linear programming solver
        - Multi-constraint optimization
        - Real-time constraint updates
        - Weighted acceptance score maximization
        - Population distribution constraints
        """)

def create_usage_instructions():
    st.header("Using the System")
    
    st.markdown("""
    1. **Data Loading**
       - Upload matrix data or use preloaded dataset
       - Automatic validation and preprocessing
    
    2. **Constraint Configuration**
       - Set total voucher limit
       - Configure venue-specific constraints
       - Adjust time and weather distributions
    
    3. **Run Optimization**
       - Click "Run Optimization" to start
       - Review results and visualizations
       - Export optimized allocations
    """)
    
    st.info("Navigate to the Simulation page to start optimizing your coupon distribution strategy.")

def main():
    set_page_config()
    create_system_overview()
    create_model_performance_section()
    create_optimization_features()
    create_technical_details()
    create_usage_instructions()

if __name__ == "__main__":
    main()
