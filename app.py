import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import streamlit as st
import io

# Set page config with better styling
st.set_page_config(
    page_title="NYC Green Taxi Analysis",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styling */
    .stApp header {
        background: rgba(255,255,255,0.9) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Title styling */
    h1 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subheader styling */
    h2, h3, h4 {
        color: #34495e !important;
        font-weight: 600 !important;
    }
    
    /* Card styling */
    .card {
        background: rgba(255,255,255,0.85);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 20px;
        border-radius: 8px;
        background-color: rgba(255,255,255,0.6);
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.8);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2e86de !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(46,134,222,0.3);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2e86de;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #2575c7;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Input controls */
    .stSlider, .stSelectbox, .stTextInput {
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .stMetric {
        background: rgba(255,255,255,0.8);
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2e86de;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Chart styling */
    .stPlotlyChart {
        border-radius: 12px;
        background: white;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Custom classes */
    .highlight-box {
        background: rgba(46,134,222,0.1);
        border-left: 4px solid #2e86de;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
    }
    
    .prediction-result {
        font-size: 24px;
        font-weight: 700;
        color: #2e86de;
        margin: 10px 0;
        text-align: center;
    }
    
    .feature-importance {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the NYC Green Taxi data for January 2020"""
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-01.parquet"
    df = pd.read_parquet(url)
    return df

def preprocess_data(df):
    """Preprocess the data according to assignment requirements"""
    if 'ehail_fee' in df.columns:
        df.drop('ehail_fee', axis=1, inplace=True)
    
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['weekday'] = df['lpep_dropoff_datetime'].dt.weekday
    df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

def create_payment_trip_chart(df):
    """Create payment and trip type distribution charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors1 = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    df['payment_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, colors=colors1, 
                                             wedgeprops=dict(width=0.4, edgecolor='w'), 
                                             textprops={'color':'#34495e', 'fontsize':10})
    ax1.set_title('Payment Type Distribution', fontsize=12, pad=20, color='#34495e')
    
    colors2 = ['#3498db', '#2ecc71', '#e74c3c']
    df['trip_type'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, colors=colors2,
                                          wedgeprops=dict(width=0.4, edgecolor='w'),
                                          textprops={'color':'#34495e', 'fontsize':10})
    ax2.set_title('Trip Type Distribution', fontsize=12, pad=20, color='#34495e')
    
    fig.patch.set_facecolor('none')
    return fig

def create_amount_distribution_chart(df):
    """Create amount distribution charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(df['total_amount'], bins=50, kde=True, ax=ax1, color='#3498db')
    ax1.set_xlabel('Total Amount ($)', color='#34495e')
    ax1.set_ylabel('Count', color='#34495e')
    ax1.set_title('Amount Distribution', fontsize=12, color='#34495e')
    ax1.tick_params(colors='#7f8c8d')
    
    sns.boxplot(y=df['total_amount'], ax=ax2, color='#2ecc71')
    ax2.set_ylabel('Total Amount ($)', color='#34495e')
    ax2.set_title('Amount Spread', fontsize=12, color='#34495e')
    ax2.tick_params(colors='#7f8c8d')
    
    fig.patch.set_facecolor('none')
    return fig

def create_correlation_matrix(df):
    """Create correlation matrix visualization"""
    numeric_vars = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
                   'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 
                   'trip_duration', 'passenger_count', 'total_amount']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_vars].corr(), annot=True, cmap='coolwarm', center=0, ax=ax,
               annot_kws={"size": 9, "color": "#2c3e50"}, fmt=".2f", linewidths=0.5,
               cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, pad=20, color='#34495e')
    fig.patch.set_facecolor('none')
    return fig

def run_analysis(df):
    """Run the complete analysis and display results in Streamlit"""
    st.sidebar.markdown("""
    <div class='card'>
        <h3>NYC Green Taxi</h3>
        <p>Analyze January 2020 trip data with interactive visualizations and machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "📈 Exploratory Analysis", 
        "🔍 Statistical Insights", 
        "🤖 Model Training",
        "🔮 Fare Prediction"
    ])
    
    with tab1:
        st.markdown("""
        <div class='highlight-box'>
            <h3>Dataset Overview</h3>
            <p>Explore the structure and characteristics of the NYC Green Taxi dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("📋 Dataset Information", expanded=True):
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
            
            with st.expander("🔍 Missing Values Analysis"):
                missing_data = df.isnull().sum().rename("Missing Values")
                st.dataframe(missing_data[missing_data > 0].sort_values(ascending=False))
        
        with col2:
            with st.expander("📅 Temporal Distributions"):
                st.markdown("**Weekday Distribution**")
                st.bar_chart(df['weekday'].value_counts().sort_index())
                
                st.markdown("**Hour of Day Distribution**")
                st.bar_chart(df['hourofday'].value_counts().sort_index())
    
    with tab2:
        st.markdown("""
        <div class='highlight-box'>
            <h3>Exploratory Data Analysis</h3>
            <p>Visual exploration of patterns and relationships in the data</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Payment and Trip Types")
        st.pyplot(create_payment_trip_chart(df))
        
        st.markdown("#### Fare Amount Distribution")
        st.pyplot(create_amount_distribution_chart(df))
        
        st.markdown("#### Feature Correlations")
        st.pyplot(create_correlation_matrix(df))
    
    with tab3:
        st.markdown("""
        <div class='highlight-box'>
            <h3>Statistical Insights</h3>
            <p>Statistical tests and group comparisons to uncover significant patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ANOVA Tests")
            st.markdown("**Total Amount by Trip Type**")
            trip_type_groups = [df[df['trip_type'] == i]['total_amount'] for i in df['trip_type'].unique()]
            f_val, p_val = stats.f_oneway(*trip_type_groups)
            col1a, col1b = st.columns(2)
            col1a.metric("F-statistic", f"{f_val:.2f}")
            col1b.metric("p-value", f"{p_val:.4f}")
            
            st.markdown("**Total Amount by Weekday**")
            weekday_groups = [df[df['weekday'] == i]['total_amount'] for i in range(7)]
            f_val, p_val = stats.f_oneway(*weekday_groups)
            col1c, col1d = st.columns(2)
            col1c.metric("F-statistic", f"{f_val:.2f}")
            col1d.metric("p-value", f"{p_val:.4f}")
        
        with col2:
            st.markdown("##### Chi-square Test")
            st.markdown("**Trip Type vs Payment Type**")
            contingency_table = pd.crosstab(df['trip_type'], df['payment_type'])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            col2a, col2b = st.columns(2)
            col2a.metric("Chi-square", f"{chi2:.2f}")
            col2b.metric("p-value", f"{p:.4f}")
            
            with st.expander("📊 Group Averages"):
                st.markdown("**Average Amounts**")
                avg_amount = df.groupby('weekday')['total_amount'].mean().rename("Avg Total Amount")
                avg_tip = df.groupby('weekday')['tip_amount'].mean().rename("Avg Tip Amount")
                st.dataframe(pd.concat([avg_amount, avg_tip], axis=1))
    
    with tab4:
        st.markdown("""
        <div class='highlight-box'>
            <h3>Model Training</h3>
            <p>Train and evaluate different machine learning models for fare prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        categorical_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type', 'weekday', 'hourofday']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        X = df_encoded.drop(['total_amount', 'lpep_pickup_datetime', 'lpep_dropoff_datetime', 'VendorID'], axis=1, errors='ignore')
        y = df_encoded['total_amount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_option = st.selectbox(
            "Select Model Type:",
            ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
            key="model_select"
        )
        
        if st.button("🚀 Train Model", key="train_button"):
            with st.spinner(f"Training {model_option} model..."):
                if model_option == "Linear Regression":
                    model = LinearRegression()
                elif model_option == "Decision Tree":
                    model = DecisionTreeRegressor(random_state=42)
                elif model_option == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.session_state.model = model
                st.session_state.feature_names = X.columns
                st.session_state.model_type = model_option
                
                st.success("Model training completed successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                with col2:
                    st.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
                
                if hasattr(model, 'feature_importances_'):
                    st.markdown("##### Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='Blues_d', ax=ax)
                    ax.set_title('Top 10 Important Features', fontsize=12, color='#34495e')
                    ax.set_xlabel('Importance Score', color='#34495e')
                    ax.set_ylabel('Feature', color='#34495e')
                    ax.tick_params(colors='#7f8c8d')
                    fig.patch.set_facecolor('none')
                    st.pyplot(fig)
    
    with tab5:
        st.markdown("""
        <div class='highlight-box'>
            <h3>Fare Prediction</h3>
            <p>Predict total fare amount based on trip characteristics</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'model' not in st.session_state:
            st.warning("Please train a model first in the Model Training tab")
            st.stop()
        
        st.markdown("#### Trip Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trip_distance = st.slider('Trip Distance (miles)', 0.0, 50.0, 2.5, 0.1)
            fare_amount = st.slider('Base Fare ($)', 0.0, 100.0, 10.0, 0.5)
            passenger_count = st.slider('Passengers', 1, 6, 1)
        
        with col2:
            trip_duration = st.slider('Duration (min)', 0.0, 120.0, 15.0, 1.0)
            payment_type = st.selectbox('Payment Type', sorted(df['payment_type'].unique()))
            trip_type = st.selectbox('Trip Type', sorted(df['trip_type'].unique()))
        
        col3, col4 = st.columns(2)
        with col3:
            weekday = st.selectbox('Weekday', range(7), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        with col4:
            hourofday = st.selectbox('Hour of Day', range(24))
        
        if st.button("🔮 Predict Fare", key="predict_button"):
            input_data = pd.DataFrame(0, index=[0], columns=st.session_state.feature_names)
            
            # Set numerical features
            input_data['trip_distance'] = trip_distance
            input_data['fare_amount'] = fare_amount
            input_data['passenger_count'] = passenger_count
            input_data['trip_duration'] = trip_duration
            
            # Set default values for other numerical features
            numeric_features = ['extra', 'mta_tax', 'tip_amount', 'tolls_amount', 
                              'improvement_surcharge', 'congestion_surcharge']
            for feature in numeric_features:
                if feature in input_data.columns:
                    input_data[feature] = float(df[feature].median())
            
            # Set categorical features
            payment_col = f"payment_type_{payment_type}"
            if payment_col in input_data.columns:
                input_data[payment_col] = 1
            
            trip_col = f"trip_type_{trip_type}"
            if trip_col in input_data.columns:
                input_data[trip_col] = 1
            
            weekday_col = f"weekday_{weekday}"
            if weekday_col in input_data.columns:
                input_data[weekday_col] = 1
            
            hour_col = f"hourofday_{hourofday}"
            if hour_col in input_data.columns:
                input_data[hour_col] = 1
            
            try:
                prediction = st.session_state.model.predict(input_data)
                
                st.markdown(f"""
                <div class='card' style='text-align: center;'>
                    <h4>Predicted Fare Amount</h4>
                    <p class='prediction-result'>${prediction[0]:.2f}</p>
                    <p>Using {st.session_state.model_type} model</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def main():
    st.title("🚕 NYC Green Taxi Fare Analysis")
    st.markdown("""
    <style>
        .title-wrapper {
            display: flex;
            align-items: center;
            gap: 15px;
        }
    </style>
    <div class='title-wrapper'>
        <div>January 2020 Trip Data Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_data()
    df = preprocess_data(df)
    run_analysis(df)

if __name__ == "__main__":
    main()