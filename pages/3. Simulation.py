import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import io
import requests



class GDriveLoader:
    """Handles loading data from Google Drive with support for large files"""
    
    @staticmethod
    def get_file_id_from_url(url: str) -> str:
        """Extract file ID from Google Drive URL"""
        if 'drive.google.com/file/d/' in url:
            file_id = url.split('drive.google.com/file/d/')[1].split('/')[0]
        elif 'drive.google.com/open?id=' in url:
            file_id = url.split('drive.google.com/open?id=')[1]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        else:
            raise ValueError("Invalid Google Drive URL format")
        return file_id

    @staticmethod
    def load_from_gdrive(file_id: str) -> Optional[pd.DataFrame]:
        """Load large CSV from Google Drive with direct download"""
        try:
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            
            with st.spinner("Downloading file from Google Drive..."):
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with io.BytesIO() as temp_file:
                    progress_bar = st.progress(0)
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024 * 1024  # 1MB chunks
                    downloaded = 0
                    
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            temp_file.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                progress = downloaded / total_size
                                progress_bar.progress(min(progress, 1.0))
                    
                    temp_file.seek(0)
                    return pd.read_csv(temp_file)
                        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None

def load_or_get_data():
    """Load data from Google Drive or return cached version"""
    
    # If data is already loaded, return it
    if 'optimization_data' in st.session_state:
        return st.session_state.optimization_data
        
    # First try loading from secrets
    gdrive_url = "https://drive.google.com/file/d/1QgCiApzXN-iFgnwq8DeRH2FkQ83FUN-E/view?usp=drive_link"
    if not gdrive_url:
        gdrive_url = st.text_input(
            "Enter Google Drive sharing URL for optimization_matrix.csv",
            help="Make sure the file is shared with 'Anyone with the link'"
        )
        
        if not gdrive_url:
            st.info("Please enter the Google Drive sharing URL to load the data.")
            return None
    
    try:
        file_id = GDriveLoader.get_file_id_from_url(gdrive_url)
        data = GDriveLoader.load_from_gdrive(file_id)
        
        if data is not None and len(data.columns) > 0 and len(data) > 0:
            # Store in session state
            st.session_state.optimization_data = data
            st.success("Successfully loaded data from Google Drive!")
            
            # Show data overview
            st.write("Data Overview:")
            st.write(f"Rows: {len(data)}")
            st.write(f"Columns: {', '.join(data.columns)}")
            
            return data
            
        else:
            st.error("The loaded data appears to be empty or invalid.")
            return None
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please check that the URL is correct and the file is shared properly.")
        return None

def clear_data():
    """Clear the cached data"""
    if 'optimization_data' in st.session_state:
        del st.session_state.optimization_data
        st.success("Data cache cleared!")
        
def load_matrix_data():
    """Load the optimization matrix from Google Drive"""
    
    # First try loading from secrets
    gdrive_url = "https://drive.google.com/file/d/1QgCiApzXN-iFgnwq8DeRH2FkQ83FUN-E/view?usp=drive_link"
    
    if not gdrive_url:
        # If no URL in secrets, show input for URL
        gdrive_url = st.text_input(
            "Enter Google Drive sharing URL for optimization_matrix.csv",
            help="Make sure the file is shared with 'Anyone with the link'"
        )
        
        if not gdrive_url:
            st.info("Please enter the Google Drive sharing URL to load the data.")
            return None
    
    try:
        # Extract file ID from URL
        file_id = GDriveLoader.get_file_id_from_url(gdrive_url)
        
        # Load the data
        data = GDriveLoader.load_from_gdrive(file_id)
        
        if data is not None:
            st.success("Successfully loaded data from Google Drive!")
            
            # Show data overview
            st.write("Data Overview:")
            st.write(f"Rows: {len(data)}")
            st.write(f"Columns: {', '.join(data.columns)}")
            
            return data
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please check that the URL is correct and the file is shared properly.")
        return None

# Category mappings for decoding
CATEGORY_MAPPINGS = {
    'coupon': {
        0: 'Restaurant(<20)',
        1: 'Coffee House',
        2: 'Carry out & Take away',
        3: 'Bar',
        4: 'Restaurant(20-50)'
    },
    'time': {
        0: '7AM',
        1: '10AM',
        2: '2PM',
        3: '6PM',
        4: '10PM'
    },
    'weather': {
        0: 'Sunny',
        1: 'Rainy',
        2: 'Snowy'
    },
    'temperature': {
        0: '30°F',
        1: '55°F',
        2: '80°F'
    }
}

# Time period groupings
TIME_PERIODS = {
    'morning': ['7AM', '10AM'],
    'afternoon': ['2PM'],
    'evening': ['6PM', '10PM']
}

# Reverse mappings for constraints
REVERSE_MAPPINGS = {
    category: {v: k for k, v in mapping.items()}
    for category, mapping in CATEGORY_MAPPINGS.items()
}

# Mapping for venue types to encoded values
VENUE_TYPE_MAPPING = {
    'bar': 3,  # Encoded value for Bar
    'coffee': 1,  # Encoded value for Coffee House
    'restaurant_low': 0,  # Encoded value for Restaurant(<20)
    'restaurant_high': 4,  # Encoded value for Restaurant(20-50)
    'carryout': 2  # Encoded value for Carry out
}
class CouponOptimizer:
    def __init__(self):
        self.optimization_matrix = None
    
    def update_status(self, message):
        """Update status in Streamlit UI if placeholder exists"""
        if self.status_placeholder is not None:
            self.status_placeholder.info(message)

    def set_optimization_matrix(self, matrix):
        """Set the optimization matrix"""
        self.optimization_matrix = matrix

    def optimize_with_constraints(self, constraints: Dict) -> pd.DataFrame:
        """Run optimization with dynamic constraints and a time limit"""
        if self.optimization_matrix is None:
            raise ValueError("Optimization matrix not set")

        matrix = self.optimization_matrix.copy()
        matrix = self._prefilter_matrix(matrix, constraints)

        problem = LpProblem("Coupon_Distribution", LpMaximize)

        matrix['decision_var'] = [
            LpVariable(f"x_{i}", cat="Binary")
            for i in range(len(matrix))
        ]

        # Objective function using weighted acceptance score
        problem += lpSum(
            var * row['weighted_acceptance_score']
            for var, (_, row) in zip(matrix['decision_var'], matrix.iterrows())
        )

        # Add constraints
        self._add_basic_constraints(problem, matrix, constraints)

        if constraints['venueDistribution']['enabled']:
            self._add_venue_constraints(problem, matrix, constraints)

        if constraints['timeDistribution']['enabled']:
            self._add_time_constraints(problem, matrix, constraints)

        if constraints['weatherDistribution']['enabled']:
            self._add_weather_constraints(problem, matrix, constraints)

        # Solve with a time limit of 15 seconds
        problem.solve(PULP_CBC_CMD(msg=True, timeLimit=5))

        matrix['allocated'] = [var.value() for var in matrix['decision_var']]
        allocated = matrix[matrix['allocated'] == 1]

        # Add decoded values for visualization
        for col, mapping in CATEGORY_MAPPINGS.items():
            if col in allocated.columns:
                allocated[f'{col}_decoded'] = allocated[col].map(mapping)

        return allocated

    def _prefilter_matrix(self, matrix: pd.DataFrame, constraints: Dict) -> pd.DataFrame:
        """Pre-filter matrix based on constraints"""
        filtered = matrix.copy()

        filtered = filtered[
            filtered['num_observations'] <= constraints['totalVouchers']
        ]

        if constraints['venueDistribution']['enabled']:
            venue_masks = []
            for venue_type, limit in constraints['venueDistribution'].items():
                if venue_type == 'enabled':
                    continue  # Skip the 'enabled' key
                if isinstance(limit, (int, float)) and limit > 0:
                    encoded_venue = VENUE_TYPE_MAPPING.get(venue_type)
                    if encoded_venue is not None:
                        venue_masks.append(
                            (filtered['coupon'] == encoded_venue) &
                            (filtered['num_observations'] <= limit)
                        )
            if venue_masks:
                filtered = filtered[pd.concat(venue_masks, axis=1).any(axis=1)]

        return filtered

    def _add_basic_constraints(self, problem, matrix, constraints):
        """Add basic optimization constraints"""
        # One voucher per customer group
        for group in matrix['CustomerGroup_ID'].unique():
            group_rows = matrix[matrix['CustomerGroup_ID'] == group]
            problem += lpSum(group_rows['decision_var']) <= 1

        # Total voucher limit
        problem += lpSum(
            var * row['num_observations']
            for var, (_, row) in zip(matrix['decision_var'], matrix.iterrows())
        ) <= constraints['totalVouchers']

    def _add_venue_constraints(self, problem, matrix, constraints):
        """Add venue-specific constraints"""
        venue_limits = constraints['venueDistribution']
        for venue_type, limit in venue_limits.items():
            if venue_type == 'enabled':
                continue  # Skip the 'enabled' key
            if isinstance(limit, (int, float)) and limit >= 0:
                encoded_venue = VENUE_TYPE_MAPPING.get(venue_type)
                if encoded_venue is not None:
                    venue_rows = matrix[matrix['coupon'] == encoded_venue]
                    problem += lpSum(
                        var * row['num_observations']
                        for var, (_, row) in zip(venue_rows['decision_var'], venue_rows.iterrows())
                    ) <= limit

    def _add_time_constraints(self, problem, matrix, constraints):
        """Add time-based constraints"""
        time_limits = constraints['timeDistribution']

        for period, limit in time_limits.items():
            if period == 'enabled':
                continue  # Skip the 'enabled' key
            if isinstance(limit, (int, float)) and limit >= 0:
                time_values = [REVERSE_MAPPINGS['time'][t] for t in TIME_PERIODS[period]]
                time_rows = matrix[matrix['time'].isin(time_values)]
                problem += lpSum(
                    var * row['num_observations']
                    for var, (_, row) in zip(time_rows['decision_var'], time_rows.iterrows())
                ) <= limit

    def _add_weather_constraints(self, problem, matrix, constraints):
        """Add weather-based constraints"""
        weather_limits = constraints['weatherDistribution']
        for weather_type, limit in weather_limits.items():
            if weather_type == 'enabled':
                continue  # Skip the 'enabled' key
            if isinstance(limit, (int, float)) and limit >= 0:
                encoded_weather = REVERSE_MAPPINGS['weather'].get(weather_type.capitalize())
                if encoded_weather is not None:
                    weather_rows = matrix[matrix['weather'] == encoded_weather]
                    problem += lpSum(
                        var * row['num_observations']
                        for var, (_, row) in zip(weather_rows['decision_var'], weather_rows.iterrows())
                    ) <= limit


def create_distribution_chart(data: pd.DataFrame, 
                            column: str, 
                            title: str,
                            color_discrete_sequence: Optional[List[str]] = None) -> go.Figure:
    """Create a distribution chart for optimization results using decoded values"""
    decoded_col = f'{column}_decoded'
    if decoded_col not in data.columns:
        decoded_col = column
        
    grouped = data.groupby(decoded_col).agg({
        'num_observations': 'sum',
        'weighted_acceptance_score': 'sum'
    }).reset_index()
    
    total_observations = grouped['num_observations'].sum()
    grouped['percentage'] = (grouped['num_observations'] / total_observations * 100).round(2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=grouped[decoded_col],
        y=grouped['percentage'],
        text=grouped['percentage'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        marker_color=color_discrete_sequence,
        name='Distribution',
        hovertemplate=(
            '<b>%{x}</b><br>'  # Label
            'Num Observations: %{customdata}<br>'  # Display num_observations in tooltip
            'Percentage: %{y:.1f}%<extra></extra>'  # Display percentage in tooltip
        ),
        customdata=grouped['num_observations']  # Provide num_observations for tooltip
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=column.capitalize().replace('_', ' '),
        yaxis_title='Percentage (%)',
        showlegend=False,
        height=400
    )
    
    return fig
    
def create_group_distribution_table(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create distribution tables for custom group distributions based on predefined group columns.
    Displays group IDs alongside concatenated column values with inline decoding.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing relevant data.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of DataFrames for each group type.
    """
    # Define the grouping columns
    group_columns = {
        'CustomerGroup_ID': [
            'gender', 'age', 'maritalStatus', 'has_children', 'education',
            'occupation', 'income', 'Bar', 'CoffeeHouse', 'CarryAway',
            'RestaurantLessThan20', 'Restaurant20To50'
        ],
        'SituationGroup_ID': [
            'destination', 'passanger', 'weather', 'temperature',
            'time', 'toCoupon_GEQ5min', 'toCoupon_GEQ15min',
            'toCoupon_GEQ25min', 'direction_same', 'direction_opp'
        ],
        'CouponGroup_ID': ['coupon', 'expiration']
    }

    def decode_value(column, value):
        decoding_rules = {
            # Time and Weather
            'time': {0: '7AM', 1: '10AM', 2: '2PM', 3: '6PM', 4: '10PM'},
            'weather': {0: 'Sunny', 1: 'Rainy', 2: 'Snowy'},
            'temperature': {0: '30°F', 1: '55°F', 2: '80°F'},
            'expiration': {0: '2h', 1: '1d'},
            
            # Demographics
            'age': {
                0: 'below21', 1: '21', 2: '26', 3: '31',
                4: '36', 5: '41', 6: '46', 7: '50plus'
            },
            'gender': {0: 'Female', 1: 'Male'},
            'maritalStatus': {
                0: 'Single', 1: 'Married partner', 2: 'Unmarried partner',
                3: 'Widowed', 4: 'Divorced'
            },
            'education': {
                0: 'Some High School',
                1: 'High School Graduate',
                2: 'Some college - no degree',
                3: 'Associates degree',
                4: 'Bachelors degree',
                5: 'Graduate degree (Masters or Doctorate)'
            },
            'occupation': {
                0: 'Unemployed',
                1: 'Architecture & Engineering',
                2: 'Student',
                3: 'Education&Training&Library',
                4: 'Healthcare Support',
                5: 'Healthcare Practitioners & Technical',
                6: 'Sales & Related',
                7: 'Management',
                8: 'Arts Design Entertainment Sports & Media',
                9: 'Computer & Mathematical',
                10: 'Life Physical Social Science',
                11: 'Personal Care & Service',
                12: 'Community & Social Services',
                13: 'Office & Administrative Support',
                14: 'Construction & Extraction',
                15: 'Legal',
                16: 'Retired',
                17: 'Installation Maintenance & Repair',
                18: 'Transportation & Material Moving',
                19: 'Business & Financial',
                20: 'Protective Service',
                21: 'Food Preparation & Serving Related',
                22: 'Production Occupations',
                23: 'Building & Grounds Cleaning & Maintenance',
                24: 'Farming Fishing & Forestry'
            },
            'income': {
                0: 'Less than $12500',
                1: '$12500 - $24999',
                2: '$25000 - $37499',
                3: '$37500 - $49999',
                4: '$50000 - $62499',
                5: '$62500 - $74999',
                6: '$75000 - $87499',
                7: '$87500 - $99999',
                8: '$100000 or More'
            },
            
            # Location and Context
            'destination': {0: 'No Urgent Place', 1: 'Home', 2: 'Work'},
            'passanger': {0: 'Alone', 1: 'Friend(s)', 2: 'Kid(s)', 3: 'Partner'},
            'coupon': {
                0: 'Restaurant(<20)',
                1: 'Coffee House',
                2: 'Carry out & Take away',
                3: 'Bar',
                4: 'Restaurant(20-50)'
            },
            
            # Frequency Variables
            'Bar': {0: 'never', 1: 'less1', 2: '1~3', 3: '4~8', 4: 'gt8'},
            'CoffeeHouse': {0: 'never', 1: 'less1', 2: '1~3', 3: '4~8', 4: 'gt8'},
            'CarryAway': {0: 'never', 1: 'less1', 2: '1~3', 3: '4~8', 4: 'gt8'},
            'RestaurantLessThan20': {0: 'never', 1: 'less1', 2: '1~3', 3: '4~8', 4: 'gt8'},
            'Restaurant20To50': {0: 'never', 1: 'less1', 2: '1~3', 3: '4~8', 4: 'gt8'}
        }
        
        # Return the decoded value, default to the original if not found
        return decoding_rules.get(column, {}).get(value, value)

    # Prepare a dictionary for group distributions
    distribution_tables = {}

    for group, columns in group_columns.items():
        # Create a decoded version of the columns
        decoded_columns = []
        for col in columns:
            if col in data.columns:
                decoded_col = f"{col}_decoded"
                data[decoded_col] = data[col].apply(lambda x: decode_value(col, x))
                decoded_columns.append(decoded_col)
            else:
                decoded_columns.append(col)  # Use original column if no decoding logic exists

        # Create a concatenated column for decoded values
        data[f'{group}_concat'] = data[decoded_columns].apply(
            lambda row: ', '.join(row.astype(str)), axis=1
        )

        # Group by the group ID and calculate statistics
        grouped = data.groupby(group).agg({
            'num_observations': 'sum',
            f'{group}_concat': lambda x: x.iloc[0]  # Take the first concatenated value per group
        }).reset_index()

        # Dynamically name the concatenated column
        concatenated_column_name = " + ".join(columns).replace('_', ' ').capitalize()
        grouped.rename(columns={
            group: f'{group}',
            'num_observations': 'Total Observations',
            f'{group}_concat': concatenated_column_name
        }, inplace=True)

        # Save to the dictionary
        distribution_tables[group] = grouped

    return distribution_tables


def main():
    st.set_page_config(page_title="Coupon Optimization", layout="wide")
    
    st.title("🎟️ Dynamic Coupon Distribution Optimizer")
    
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = CouponOptimizer()
    
    # Option to choose data source with cache control
    col1, col2 = st.columns([3, 1])
    with col1:
        data_source = st.radio(
            "Choose data source",
            ["Load from Google Drive", "Upload custom matrix"]
        )
    with col2:
        if st.button("Clear Cached Data"):
            clear_data()
            st.rerun()
    
    if data_source == "Load from Google Drive":
        data = load_or_get_data()
        if data is None:
            return
    else:
        uploaded_file = st.file_uploader("Upload optimization matrix (CSV)", type=['csv'])
        if uploaded_file is None:
            st.info("Please upload your optimization matrix CSV file to begin.")
            return
        data = pd.read_csv(uploaded_file)
        st.session_state.optimization_data = data  # Cache uploaded data too

        
    if data is not None:
        try:
                       
            required_columns = [
                'CustomerGroup_ID', 'num_observations', 'weighted_acceptance_score',
                'coupon', 'time', 'weather'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
                
            st.session_state.optimizer.set_optimization_matrix(data)
            
            st.success("Matrix loaded successfully!")
            
            # Data Overview
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customer Groups", data['CustomerGroup_ID'].nunique())
            with col2:
                st.metric("Total Observations", data[(data['SituationGroup_ID'] == 1) & (data['CouponGroup_ID'] == 1)]['num_observations'].sum())
            with col3:
                st.metric("Average Acceptance Score", 
                         f"{0.56:.2f}")
            
            # Group Information
            with st.expander("View Group Information"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("Customer Groups")
                    st.dataframe(data.groupby('CustomerGroup_ID')['num_observations'].sum())
                with col2:
                    st.write("Situation Groups")
                    st.dataframe(data.groupby('SituationGroup_ID')['num_observations'].sum())
                with col3:
                    st.write("Coupon Groups")
                    st.dataframe(data.groupby('CouponGroup_ID')['num_observations'].sum())
            
            # Constraints Setup
            st.sidebar.header("Constraint Settings")
            
            total_vouchers = st.sidebar.slider(
                "Total Vouchers",
                min_value=50,
                max_value=500,
                value=100,
                step=10
            )
            
            # Venue distribution
            st.sidebar.subheader("Venue Distribution")
            venue_enabled = st.sidebar.checkbox("Enable Venue Constraints", value=True)
            venue_constraints = {}
            
            if venue_enabled:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    venue_constraints['bar'] = st.number_input("Bar Limit", 0, 200, 50)
                    venue_constraints['restaurant_low'] = st.number_input("Restaurant(<$20) Limit", 0, 200, 50)
                    venue_constraints['carryout'] = st.number_input("Carry out & Take away", 0, 200, 50)
                with col2:
                    venue_constraints['coffee'] = st.number_input("Coffee Shop Limit", 0, 200, 50)
                    venue_constraints['restaurant_high'] = st.number_input("Restaurant($20-50) Limit", 0, 200, 50)
            
            # Time distribution
            st.sidebar.subheader("Time Distribution")
            time_enabled = st.sidebar.checkbox("Enable Time Constraints")
            time_constraints = {}
            
            if time_enabled:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    time_constraints['morning'] = st.number_input("Morning (7AM-11AM)", 0, 200, 30)
                    time_constraints['afternoon'] = st.number_input("Afternoon (2PM)", 0, 200, 40)
                with col2:
                    time_constraints['evening'] = st.number_input("Evening (6PM-10PM)", 0, 200, 30)
            
            # Weather distribution
            st.sidebar.subheader("Weather Distribution")
            weather_enabled = st.sidebar.checkbox("Enable Weather Constraints")
            weather_constraints = {}
            
            if weather_enabled:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    weather_constraints['sunny'] = st.number_input("Sunny Weather", 0, 200, 40)
                    weather_constraints['rainy'] = st.number_input("Rainy Weather", 0, 200, 30)
                with col2:
                    weather_constraints['snowy'] = st.number_input("Snowy Weather", 0, 200, 30)
            
            constraints = {
                'totalVouchers': total_vouchers,
                'venueDistribution': {
                    'enabled': venue_enabled,
                    **venue_constraints
                },
                'timeDistribution': {
                    'enabled': time_enabled,
                    **time_constraints
                },
                'weatherDistribution': {
                    'enabled': weather_enabled,
                    **weather_constraints
                }
            }
            
            if st.button("Run Optimization", type="primary"):
                with st.spinner("Running optimization..."):
                    try:
                        results = st.session_state.optimizer.optimize_with_constraints(constraints)
                        
                        st.success(f"Successfully allocated {len(results)} customer group")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            venue_fig = create_distribution_chart(
                                results,
                                'coupon',
                                'Venue Distribution',
                                px.colors.qualitative.Set3
                            )
                            st.plotly_chart(venue_fig, use_container_width=True)
                        with col2:
                            time_fig = create_distribution_chart(
                                    results,
                                    'time',
                                    'Time Distribution',
                                    px.colors.qualitative.Set2
                                )
                            st.plotly_chart(time_fig, use_container_width=True)    
                        with col3:
                            weather_fig = create_distribution_chart(
                                results,
                                'weather',
                                'Weather Distribution',
                                px.colors.qualitative.Set1
                            )
                            st.plotly_chart(weather_fig, use_container_width=True)
                            
                        # Add summary metrics
                        st.subheader("Optimization Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric(
                                "Total Allocated Customers",
                                results['num_observations'].sum()
                            )
                        with col2:
                            st.metric(
                                "Total Expected Acceptance",
                                f"{results['weighted_acceptance_score'].sum():.2f}"
                            )
                        with col4:
                            st.metric(
                                "Number of Situation Groups",
                                results['SituationGroup_ID'].nunique()
                            )
                        with col3:
                            st.metric(
                                "Number of Customer Groups",
                                results['CustomerGroup_ID'].nunique()
                            )
                        with col5:
                            st.metric(
                                "Number of Coupon Groups",
                                results['CouponGroup_ID'].nunique()
                            )
                        
                        # Generate distribution tables
                        distribution_tables = create_group_distribution_table(results)

                        # Display group statistics
                        with st.expander("View Group Statistics"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write("Customer Group Distribution")
                                st.dataframe(distribution_tables['CustomerGroup_ID'])
                            with col2:
                                st.write("Situation Group Distribution")
                                st.dataframe(distribution_tables['SituationGroup_ID'])
                            with col3:
                                st.write("Coupon Group Distribution")
                                st.dataframe(distribution_tables['CouponGroup_ID'])

                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name="optimization_results.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during optimization: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("Please upload your optimization matrix CSV file to begin.")
        
        # Show example format
        st.subheader("Required CSV Format")
        example_data = pd.DataFrame({
            'CustomerGroup_ID': [1, 1, 2],
            'Demographic_Behavior': ['0-0-0-0-1-0-0', '0-0-0-0-1-0-0', '1-0-0-1-0-0-0'],
            'num_observations': [20, 25, 30],
            'weighted_acceptance_score': [0.8, 0.7, 0.9],
            'SituationGroup_ID': [1, 2, 1],
            'Situation': ['0-0-0-0-1', '1-0-0-0-1', '0-1-0-0-1'],
            'CouponGroup_ID': [1, 2, 3],
            'coupon': [0, 1, 2],
            'time': [1, 2, 3],
            'weather': [0, 1, 0]
        })
        st.dataframe(example_data)

if __name__ == "__main__":
    main()
