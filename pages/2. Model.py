import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

def set_page_config():
    st.set_page_config(
        page_title="Model Analysis",
        page_icon="🤖",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .metric-row {
            padding: 1rem;
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def display_model_overview():
    st.title("🤖 Model Analysis Results")
    
    st.markdown("""
    ## Model Overview
    Our Random Forest model has been optimized using grid search cross-validation. 
    Below are the detailed results of our model evaluation.
    """)
    
    # Display best parameters
    st.subheader("Best Model Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Depth", "20")
    with col2:
        st.metric("Min Samples Leaf", "1")
    with col3:
        st.metric("Min Samples Split", "2")
    with col4:
        st.metric("N Estimators", "200")

def display_performance_metrics():
    st.header("Performance Metrics")
    
    # Overall metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Cross-validation Score",
            "0.751",
            "±0.010 std"
        )
    with col2:
        st.metric(
            "ROC AUC",
            "0.809",
            "Good Discrimination"
        )
    with col3:
        st.metric(
            "PR AUC",
            "0.834",
            "Strong Precision-Recall"
        )

def display_confusion_matrix():
    st.subheader("Confusion Matrix")
    
    cm = np.array([
        [688, 407],
        [253, 1189]
    ])
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        colorscale='Viridis',
        showscale=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_feature_importance():
    st.header("Feature Importance Analysis")
    
    # Feature importance data from results
    feature_importance = pd.DataFrame({
        'feature': [
            'coupon', 'occupation', 'income', 'age', 'CoffeeHouse',
            'time', 'Bar', 'education', 'CarryAway', 'RestaurantLessThan20'
        ],
        'importance': [
            0.107294, 0.085538, 0.074175, 0.064533, 0.064357,
            0.056346, 0.053024, 0.051886, 0.049802, 0.048409
        ]
    })
    
    fig = px.bar(
        feature_importance.sort_values('importance', ascending=True),
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features'
    )
    
    fig.update_layout(
        height=400,
        yaxis_title="",
        xaxis_title="Relative Importance"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Key Feature Insights
    1. **Coupon Type** (10.7%): Most influential feature in predicting acceptance
    2. **Occupation** (8.6%): Second most important predictor
    3. **Income** (7.4%): Strong influence on coupon acceptance
    4. **Age & Coffee House Frequency** (~6.4% each): Similar level of importance
    5. **Time** (5.6%): Moderate influence on acceptance decisions
    """)

def display_classification_report():
    st.header("Classification Performance")
    
    # Create DataFrame for classification metrics
    report_data = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'No Acceptance (0)': ['0.73', '0.63', '0.68'],
        'Acceptance (1)': ['0.74', '0.82', '0.78'],
        'Weighted Avg': ['0.74', '0.74', '0.74']
    }).set_index('Metric')
    
    st.dataframe(report_data, use_container_width=True)

def display_insights():
    st.header("Model Insights & Key Findings")
    
    st.markdown("""
    ### Strengths
    1. **Strong Overall Performance**
       - 75.1% cross-validation accuracy with low variance (±0.010)
       - High ROC AUC (0.809) indicating good discrimination
       - Strong PR AUC (0.834) showing reliable predictions
    
    2. **Class Performance**
       - Better at predicting acceptances (F1 = 0.78)
       - Reasonable performance on rejections (F1 = 0.68)
       - Well-balanced precision across classes
    
    3. **Feature Insights**
       - Coupon type is the dominant predictor
       - Demographic factors (occupation, income, age) are highly influential
       - Behavioral patterns (CoffeeHouse frequency) show significant impact
    
    ### Areas for Consideration
    1. **Class Balance**
       - Lower recall for non-acceptance (0.63)
       - Some false positives (407 cases)
       - Potential for targeted improvement in rejection prediction
    
    2. **Feature Utilization**
       - Consider feature engineering for temporal patterns
       - Explore interaction effects between top features
       - Investigate combining related features for stronger signals
    """)

def main():
    set_page_config()
    display_model_overview()
    display_performance_metrics()
    
    col1, col2 = st.columns(2)
    with col1:
        display_confusion_matrix()
    with col2:
        display_classification_report()
    
    display_feature_importance()
    display_insights()

if __name__ == "__main__":
    main()
