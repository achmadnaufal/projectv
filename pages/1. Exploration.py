import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import numpy as np

def set_page_config():
    st.set_page_config(
        page_title="Coupon Acceptance Analysis",
        page_icon="🎟️",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        .stPlotlyChart {
            background-color: white;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .filter-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the dataset"""
    df = pd.read_csv('invehiclecouponrecommendation.csv')
    return df

def get_unique_values(df: pd.DataFrame, column: str) -> List:
    """Get unique values for a column"""
    return sorted(df[column].unique().tolist())

def filter_dataframe(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    for column, values in filters.items():
        if values:  # Only apply filter if values are selected
            filtered_df = filtered_df[filtered_df[column].isin(values)]
    
    return filtered_df

def calculate_acceptance_rates(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Calculate acceptance rates for a given feature"""
    grouped = df.groupby(feature)['Y'].agg([
        ('count', 'count'),
        ('accepted', 'sum')
    ]).reset_index()
    
    grouped['acceptance_rate'] = (grouped['accepted'] / grouped['count'] * 100).round(2)
    grouped = grouped.sort_values('acceptance_rate', ascending=False)
    
    # Add percentage of total
    total_samples = grouped['count'].sum()
    grouped['percentage_of_total'] = (grouped['count'] / total_samples * 100).round(2)
    
    return grouped

def create_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    show_percentage: bool = True
) -> go.Figure:
    """Create an enhanced bar chart using plotly"""
    fig = go.Figure()
    
    # Add acceptance rate bars
    fig.add_trace(go.Bar(
        x=data[x],
        y=data[y],
        name='Acceptance Rate',
        text=data[y].round(1).astype(str) + '%',
        textposition='outside',
        marker_color='#1f77b4'
    ))
    
    if show_percentage:
        # Add percentage of total line
        fig.add_trace(go.Scatter(
            x=data[x],
            y=data['percentage_of_total'],
            name='% of Total',
            yaxis='y2',
            line=dict(color='#ff7f0e', width=2),
            text=data['percentage_of_total'].round(1).astype(str) + '%',
            textposition='top center'
        ))
    
    fig.update_layout(
        title=title,
        height=500,
        yaxis=dict(
            title='Acceptance Rate (%)',
            range=[0, max(data[y]) * 1.1]
        ),
        yaxis2=dict(
            title='% of Total',
            overlaying='y',
            side='right',
            range=[0, max(data['percentage_of_total']) * 1.1]
        ),
        xaxis_title=x.capitalize(),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=50)
    )
    
    return fig

def analyze_combinations(
    df: pd.DataFrame,
    features: List[str],
    min_samples: int = 50
) -> pd.DataFrame:
    """Analyze feature combinations with enhanced metrics"""
    grouped = df.groupby(features)['Y'].agg([
        ('count', 'count'),
        ('accepted', 'sum')
    ]).reset_index()
    
    grouped['acceptance_rate'] = (grouped['accepted'] / grouped['count'] * 100).round(2)
    
    # Add percentage of total and filter by minimum samples
    total_samples = grouped['count'].sum()
    grouped['percentage_of_total'] = (grouped['count'] / total_samples * 100).round(2)
    grouped = grouped[grouped['count'] >= min_samples]
    
    grouped = grouped.sort_values('acceptance_rate', ascending=False)
    return grouped

def create_filters_section(df: pd.DataFrame) -> Dict:
    """Create unified filters section"""
    st.markdown("<div class='filter-container'>", unsafe_allow_html=True)
    st.header("🎯 Analysis Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Selection")
        selected_features = st.multiselect(
            "Select features to analyze",
            options=['destination', 'coupon', 'time', 'weather', 'expiration', 'passanger'],
            default=['destination', 'coupon', 'time'],
            help="Choose features to analyze individually"
        )
        
        min_samples = st.slider(
            "Minimum samples for combination analysis",
            min_value=10,
            max_value=200,
            value=50,
            help="Filter out combinations with fewer samples"
        )
    
    with col2:
        st.subheader("Data Filters")
        filters = {}
        
        # Create filters for selected features
        for feature in selected_features:
            unique_values = get_unique_values(df, feature)
            filters[feature] = st.multiselect(
                f"Filter by {feature}",
                options=unique_values,
                default=[],
                help=f"Select specific {feature} values to include"
            )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return {
        'selected_features': selected_features,
        'min_samples': min_samples,
        'data_filters': filters
    }

def main():
    set_page_config()
    
    st.title("🎟️ Coupon Acceptance Analysis")
    st.write("Analyze and visualize coupon acceptance patterns across different features")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Please upload the dataset file 'invehiclecouponrecommendation.csv'")
        return

    # Unified filters section
    filters = create_filters_section(df)
    
    # Apply filters to dataframe
    filtered_df = filter_dataframe(df, filters['data_filters'])
    
    # Display current filter summary
    if any(filters['data_filters'].values()):
        st.info(f"Analyzing {len(filtered_df):,} samples after applying filters (from {len(df):,} total)")

    # Individual feature analysis
    st.header("📊 Individual Feature Analysis")
    
    for idx, feature in enumerate(filters['selected_features']):
        rates = calculate_acceptance_rates(filtered_df, feature)
        fig = create_bar_chart(
            rates,
            x=feature,
            y='acceptance_rate',
            title=f'Acceptance Rates by {feature.capitalize()}'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed statistics
        with st.expander(f"Detailed Statistics for {feature.capitalize()}"):
            stats_df = rates[[feature, 'count', 'accepted', 'acceptance_rate', 'percentage_of_total']]
            stats_df.columns = [
                feature.capitalize(),
                'Total Samples',
                'Accepted',
                'Acceptance Rate (%)',
                '% of Total'
            ]
            st.dataframe(stats_df, use_container_width=True)

    # Combination analysis
    if len(filters['selected_features']) > 1:
        st.header("🔍 Feature Combination Analysis")
        
        combo_rates = analyze_combinations(
            filtered_df,
            filters['selected_features'],
            filters['min_samples']
        )
        
        # Create readable combination labels
        combo_rates['combination'] = combo_rates.apply(
            lambda x: ' | '.join([f"{f}: {x[f]}" for f in filters['selected_features']]),
            axis=1
        )
        
        # Create combination visualization
        fig = px.bar(
            combo_rates.head(10),
            x='acceptance_rate',
            y='combination',
            orientation='h',
            title=f'Top 10 Feature Combinations (min. {filters["min_samples"]} samples)',
            text='acceptance_rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=600, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

        # Show detailed combination statistics
        with st.expander("View Detailed Combination Statistics"):
            stats_df = combo_rates[[
                'combination', 'count', 'accepted',
                'acceptance_rate', 'percentage_of_total'
            ]].head(10)
            stats_df.columns = [
                'Combination',
                'Total Samples',
                'Accepted',
                'Acceptance Rate (%)',
                '% of Total'
            ]
            st.dataframe(stats_df)

if __name__ == "__main__":
    main()
