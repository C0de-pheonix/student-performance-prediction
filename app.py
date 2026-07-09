"""
Student Performance Prediction System
========================================
Main Streamlit Dashboard Application (Module 4)

A multi-page analytics dashboard with:
  🏠 Home - Overview & key metrics
  📊 EDA Dashboard - Interactive visualizations
  🤖 Model Performance - ML model evaluation
  🎯 Predict - Single student prediction
  📈 Student Comparison - Side-by-side analysis
  💡 Recommendations - Performance improvement suggestions

Run: streamlit run app.py
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import load_data, handle_missing_values, remove_duplicates, encode_categorical, preprocess_data
from recommender import generate_recommendations, get_strengths, get_summary_stats

# ============================
# PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS FOR PREMIUM UI
# ============================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid rgba(255,255,255,0.6);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: #2d3436;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #636e72;
        margin-top: 0.3rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Colored metric cards */
    .metric-green { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); }
    .metric-blue { background: linear-gradient(135deg, #0984e3 0%, #6c5ce7 100%); }
    .metric-orange { background: linear-gradient(135deg, #e17055 0%, #fdcb6e 100%); }
    .metric-purple { background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); }
    .metric-green .metric-value, .metric-blue .metric-value,
    .metric-orange .metric-value, .metric-purple .metric-value {
        color: white;
    }
    .metric-green .metric-label, .metric-blue .metric-label,
    .metric-orange .metric-label, .metric-purple .metric-label {
        color: rgba(255,255,255,0.85);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3436;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #6c5ce7;
        display: inline-block;
    }
    
    /* Recommendation cards */
    .rec-critical { border-left: 5px solid #d63031; background: #fff5f5; }
    .rec-high { border-left: 5px solid #e17055; background: #ffeaa7; }
    .rec-medium { border-left: 5px solid #fdcb6e; background: #ffeaa710; }
    .rec-low { border-left: 5px solid #00b894; background: #f0fff4; }
    .rec-card {
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
    }
    
    /* Strength badge */
    .strength-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3436 0%, #000000 100%);
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebarFooter"] {display: none !important;}
    a[href*="streamlit.io"] {display: none !important;}
    
    /* Tables */
    .styled-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ============================
# DATA LOADING & CACHING
# ============================
@st.cache_data
def load_and_clean_data():
    """Load and preprocess data, cached for performance."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'student_data.csv')
    if not os.path.exists(data_path):
        st.error("❌ Dataset not found! Please run `python src/generate_dataset.py` first.")
        st.stop()
    
    df = pd.read_csv(data_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    return df


@st.cache_data
def get_preprocessed_data():
    """Get fully preprocessed data for model predictions."""
    return preprocess_data()


@st.cache_resource
def load_models():
    """Train ML models on the fly instead of loading from disk."""
    from custom_ml import CustomLinearRegression, CustomDecisionTreeClassifier, CustomRandomForestClassifier
    
    try:
        prep_data = get_preprocessed_data()
    except Exception as e:
        return {}
        
    lr = CustomLinearRegression()
    lr.fit(prep_data['X_train_reg'], prep_data['y_train_reg'])
    
    dt = CustomDecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)
    dt.fit(prep_data['X_train_clf'], prep_data['y_train_clf'])
    
    rf = CustomRandomForestClassifier(n_estimators=10, max_depth=6, min_samples_split=5, min_samples_leaf=2, random_state=42)
    rf.fit(prep_data['X_train_clf'], prep_data['y_train_clf'])
    
    return {
        'linear_regression': lr,
        'decision_tree': dt,
        'random_forest': rf
    }


# ============================
# SIDEBAR NAVIGATION
# ============================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: white; font-size: 1.5rem; margin: 0;">🎓</h1>
        <h2 style="color: white; font-size: 1.1rem; margin: 0.3rem 0; font-weight: 700;">Student Performance</h2>
        <p style="color: #b2bec3; font-size: 0.8rem; margin: 0;">Prediction System</p>
    </div>
    <hr style="border-color: #636e72; margin: 0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home",
         "📊 EDA Dashboard",
         "🤖 Model Performance",
         "🎯 Predict",
         "📈 Student Comparison",
         "💡 Recommendations"],
        label_visibility="collapsed"
    )
    



# ============================
# LOAD DATA
# ============================
df = load_and_clean_data()


# ============================
# PAGE: HOME
# ============================
if page == "🏠 Home":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Student Performance Prediction System</h1>
        <p>Analyze academic data • Predict performance • Generate actionable insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-blue">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Students</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = df['final_exam_score'].mean()
        st.markdown(f"""
        <div class="metric-card metric-purple">
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Avg. Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_pct = (df['performance_level'] == 'High').mean() * 100
        st.markdown(f"""
        <div class="metric-card metric-green">
            <div class="metric-value">{high_pct:.1f}%</div>
            <div class="metric-label">High Performers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        low_pct = (df['performance_level'] == 'Low').mean() * 100
        st.markdown(f"""
        <div class="metric-card metric-orange">
            <div class="metric-value">{low_pct:.1f}%</div>
            <div class="metric-label">Low Performers</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="section-header">📊 Performance Distribution</div>', unsafe_allow_html=True)
        perf_counts = df['performance_level'].value_counts().reindex(['High', 'Average', 'Low'])
        fig = px.pie(
            values=perf_counts.values,
            names=perf_counts.index,
            color=perf_counts.index,
            color_discrete_map={'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'},
            hole=0.45
        )
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          textfont_size=14)
        fig.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            font=dict(family="Inter")
        )
        st.plotly_chart(fig, width="stretch")
    
    with col_right:
        st.markdown('<div class="section-header">📈 Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x='final_exam_score', nbins=30,
            color='performance_level',
            color_discrete_map={'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'},
            opacity=0.8,
            barmode='overlay'
        )
        fig.update_layout(
            xaxis_title="Final Exam Score",
            yaxis_title="Count",
            margin=dict(t=20, b=40, l=40, r=20),
            height=350,
            font=dict(family="Inter"),
            legend_title="Performance"
        )
        st.plotly_chart(fig, width="stretch")
    
    # Dataset Preview
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(15), width="stretch", height=400)
    
    # Dataset stats
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**📐 Dataset Shape:**")
        st.info(f"{df.shape[0]} rows × {df.shape[1]} columns")
    with col_b:
        st.markdown("**📊 Feature Types:**")
        st.info(f"{len(df.select_dtypes(include=[np.number]).columns)} numerical, {len(df.select_dtypes(include='object').columns)} categorical")


# ============================
# PAGE: EDA DASHBOARD
# ============================
elif page == "📊 EDA Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Interactive visualizations revealing key patterns in student performance data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters
    with st.expander("🔍 **Filter Data**", expanded=False):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            gender_filter = st.multiselect("Gender", df['gender'].unique(), default=list(df['gender'].unique()))
        with fcol2:
            income_filter = st.multiselect("Family Income", df['family_income'].unique(), default=list(df['family_income'].unique()))
        with fcol3:
            edu_filter = st.multiselect("Parent Education", df['parent_education'].unique(), default=list(df['parent_education'].unique()))
        with fcol4:
            perf_filter = st.multiselect("Performance Level", df['performance_level'].unique(), default=list(df['performance_level'].unique()))
    
    # Apply filters
    df_filtered = df[
        (df['gender'].isin(gender_filter)) &
        (df['family_income'].isin(income_filter)) &
        (df['parent_education'].isin(edu_filter)) &
        (df['performance_level'].isin(perf_filter))
    ]
    
    st.caption(f"Showing **{len(df_filtered)}** of {len(df)} students")
    
    # Row 1: Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📚 Study Hours vs Score</div>', unsafe_allow_html=True)
        fig = px.scatter(
            df_filtered, x='study_hours_per_week', y='final_exam_score',
            color='performance_level',
            color_discrete_map={'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'},
            trendline='ols', opacity=0.6,
            hover_data=['attendance_percentage', 'internal_marks']
        )
        fig.update_layout(height=400, margin=dict(t=20, b=40, l=40, r=20),
                          font=dict(family="Inter"),
                          xaxis_title="Study Hours/Week", yaxis_title="Final Score")
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown('<div class="section-header">📅 Attendance vs Score</div>', unsafe_allow_html=True)
        fig = px.scatter(
            df_filtered, x='attendance_percentage', y='final_exam_score',
            color='performance_level',
            color_discrete_map={'High': '#00B894', 'Average': '#FDCB6E', 'Low': '#E17055'},
            trendline='ols', opacity=0.6,
            hover_data=['study_hours_per_week', 'internal_marks']
        )
        fig.update_layout(height=400, margin=dict(t=20, b=40, l=40, r=20),
                          font=dict(family="Inter"),
                          xaxis_title="Attendance %", yaxis_title="Final Score")
        fig.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="75% threshold")
        st.plotly_chart(fig, width="stretch")
    
    # Row 2: Correlation heatmap
    st.markdown('<div class="section-header">🔗 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    numerical_cols = ['age', 'attendance_percentage', 'study_hours_per_week', 'previous_cgpa',
                      'internal_marks', 'participation_score', 'sleep_hours', 'final_exam_score']
    numerical_cols = [c for c in numerical_cols if c in df_filtered.columns]
    corr = df_filtered[numerical_cols].corr()
    
    fig = px.imshow(
        corr, text_auto='.2f', aspect='auto',
        color_continuous_scale='RdBu_r', zmin=-1, zmax=1
    )
    fig.update_layout(height=500, margin=dict(t=20, b=20, l=20, r=20),
                      font=dict(family="Inter"))
    st.plotly_chart(fig, width="stretch")
    
    # Row 3: Box plots
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="section-header">👤 Score by Gender</div>', unsafe_allow_html=True)
        fig = px.box(df_filtered, x='gender', y='final_exam_score',
                     color='gender', color_discrete_map={'Male': '#6C5CE7', 'Female': '#FD79A8'})
        fig.update_layout(height=350, showlegend=False, margin=dict(t=20, b=40, l=40, r=20),
                          font=dict(family="Inter"))
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown('<div class="section-header">🎓 Score by Parent Edu</div>', unsafe_allow_html=True)
        fig = px.box(df_filtered, x='parent_education', y='final_exam_score',
                     color='parent_education',
                     category_orders={'parent_education': ['High School', 'Bachelor', 'Master', 'PhD']})
        fig.update_layout(height=350, showlegend=False, margin=dict(t=20, b=40, l=40, r=20),
                          font=dict(family="Inter"))
        st.plotly_chart(fig, width="stretch")
    
    with col3:
        st.markdown('<div class="section-header">💰 Score by Income</div>', unsafe_allow_html=True)
        fig = px.box(df_filtered, x='family_income', y='final_exam_score',
                     color='family_income',
                     color_discrete_map={'Low': '#E17055', 'Medium': '#FDCB6E', 'High': '#00B894'},
                     category_orders={'family_income': ['Low', 'Medium', 'High']})
        fig.update_layout(height=350, showlegend=False, margin=dict(t=20, b=40, l=40, r=20),
                          font=dict(family="Inter"))
        st.plotly_chart(fig, width="stretch")
    
    # Row 4: Mean scores by category
    st.markdown('<div class="section-header">📊 Mean Scores by Category</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        means = df_filtered.groupby('test_preparation')['final_exam_score'].mean().reset_index()
        fig = px.bar(means, x='test_preparation', y='final_exam_score',
                     color='test_preparation',
                     color_discrete_map={'Completed': '#00B894', 'Not Completed': '#E17055'},
                     text=means['final_exam_score'].round(1))
        fig.update_layout(height=300, showlegend=False, margin=dict(t=20, b=40, l=40, r=20),
                          font=dict(family="Inter"),
                          xaxis_title="Test Preparation", yaxis_title="Mean Score")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        means = df_filtered.groupby('internet_access')['final_exam_score'].mean().reset_index()
        fig = px.bar(means, x='internet_access', y='final_exam_score',
                     color='internet_access',
                     color_discrete_map={'Yes': '#0984E3', 'No': '#FDCB6E'},
                     text=means['final_exam_score'].round(1))
        fig.update_layout(height=300, showlegend=False, margin=dict(t=20, b=40, l=40, r=20),
                          font=dict(family="Inter"),
                          xaxis_title="Internet Access", yaxis_title="Mean Score")
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, width="stretch")


# ============================
# PAGE: MODEL PERFORMANCE
# ============================
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Machine Learning Model Performance</h1>
        <p>Comparison of trained models — Linear Regression, Decision Tree, and Random Forest</p>
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    
    if not models:
        st.warning("⚠️ No trained models found. Please run model training first:")
        st.code("python -c \"import sys; sys.path.insert(0,'src'); from data_preprocessing import preprocess_data; from model import train_all_models; train_all_models(preprocess_data())\"", language="bash")
        st.stop()
    
    # Load preprocessed data for evaluation
    try:
        prep_data = get_preprocessed_data()
    except Exception as e:
        st.error(f"Error loading preprocessed data: {e}")
        st.stop()
    
    # Evaluate models
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 confusion_matrix, r2_score, mean_absolute_error, mean_squared_error)
    
    st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
    
    # Regression metrics
    if 'linear_regression' in models:
        lr = models['linear_regression']
        y_pred_reg = lr.predict(prep_data['X_test_reg'])
        lr_r2 = r2_score(prep_data['y_test_reg'], y_pred_reg)
        lr_mae = mean_absolute_error(prep_data['y_test_reg'], y_pred_reg)
        lr_rmse = np.sqrt(mean_squared_error(prep_data['y_test_reg'], y_pred_reg))
        
        st.markdown("### 📈 Linear Regression (Score Prediction)")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{lr_r2:.4f}")
        col2.metric("MAE", f"{lr_mae:.2f}")
        col3.metric("RMSE", f"{lr_rmse:.2f}")
        
        # Actual vs Predicted plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prep_data['y_test_reg'], y=y_pred_reg,
                                 mode='markers', marker=dict(color='#6C5CE7', opacity=0.5, size=6),
                                 name='Predictions'))
        # Perfect prediction line
        min_val = min(prep_data['y_test_reg'].min(), y_pred_reg.min())
        max_val = max(prep_data['y_test_reg'].max(), y_pred_reg.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', line=dict(color='red', dash='dash'),
                                 name='Perfect Prediction'))
        fig.update_layout(
            xaxis_title="Actual Score", yaxis_title="Predicted Score",
            height=400, margin=dict(t=30, b=40, l=40, r=20),
            font=dict(family="Inter"),
            title="Actual vs Predicted Scores"
        )
        st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    # Classification metrics
    classification_results = []
    
    for model_name, display_name in [('decision_tree', 'Decision Tree'), ('random_forest', 'Random Forest')]:
        if model_name in models:
            model = models[model_name]
            y_pred = model.predict(prep_data['X_test_clf'])
            
            acc = accuracy_score(prep_data['y_test_clf'], y_pred)
            prec = precision_score(prep_data['y_test_clf'], y_pred, average='weighted', zero_division=0)
            rec = recall_score(prep_data['y_test_clf'], y_pred, average='weighted', zero_division=0)
            f1 = f1_score(prep_data['y_test_clf'], y_pred, average='weighted', zero_division=0)
            
            classification_results.append({
                'Model': display_name,
                'Accuracy': f"{acc:.4f}",
                'Precision': f"{prec:.4f}",
                'Recall': f"{rec:.4f}",
                'F1 Score': f"{f1:.4f}"
            })
    
    if classification_results:
        st.markdown("### 🏷️ Classification Models (Performance Level Prediction)")
        st.table(pd.DataFrame(classification_results))
    
    # Confusion matrices
    col1, col2 = st.columns(2)
    labels = ['Low', 'Average', 'High']
    
    for idx, (model_name, display_name) in enumerate([('decision_tree', 'Decision Tree'), ('random_forest', 'Random Forest')]):
        if model_name in models:
            model = models[model_name]
            y_pred = model.predict(prep_data['X_test_clf'])
            cm = confusion_matrix(prep_data['y_test_clf'], y_pred)
            
            with [col1, col2][idx]:
                st.markdown(f"**{display_name} — Confusion Matrix**")
                fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                                color_continuous_scale='Purples',
                                labels=dict(x="Predicted", y="Actual"))
                fig.update_layout(height=350, margin=dict(t=30, b=40, l=40, r=20),
                                  font=dict(family="Inter"))
                st.plotly_chart(fig, width="stretch")
    
    # Feature importance
    if 'random_forest' in models:
        st.markdown('<div class="section-header">🏆 Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        rf = models['random_forest']
        importances = rf.feature_importances_
        feature_names = prep_data['feature_names']
        
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(height=450, margin=dict(t=20, b=40, l=20, r=20),
                          font=dict(family="Inter"), showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")


# ============================
# PAGE: PREDICT
# ============================
elif page == "🎯 Predict":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Student Performance Prediction</h1>
        <p>Enter student details to predict their academic performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    if not models:
        st.warning("⚠️ Models not trained yet. Please train models first.")
        st.stop()
    
    # Input form
    st.markdown('<div class="section-header">📝 Enter Student Details</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 17, 25, 20)
        attendance = st.slider("Attendance (%)", 40.0, 100.0, 75.0, 0.5)
        study_hours = st.slider("Study Hours/Week", 0.0, 40.0, 12.0, 0.5)
        previous_cgpa = st.slider("Previous CGPA", 2.0, 10.0, 6.5, 0.1)
    
    with col2:
        internal_marks = st.slider("Internal Marks (out of 50)", 0, 50, 30)
        participation = st.slider("Participation Score (0-10)", 0, 10, 5)
        sleep_hours = st.slider("Sleep Hours/Night", 4.0, 10.0, 7.0, 0.5)
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        internet = st.selectbox("Internet Access", ["Yes", "No"])
    
    with col3:
        parent_edu = st.selectbox("Parent Education", ["High School", "Bachelor", "Master", "PhD"])
        family_inc = st.selectbox("Family Income", ["Low", "Medium", "High"])
        test_prep = st.selectbox("Test Preparation", ["Completed", "Not Completed"])
    
    if st.button("🔮 Predict Performance", type="primary", width="stretch"):
        # Encode inputs
        gender_enc = 1 if gender == "Male" else 0
        extra_enc = 1 if extracurricular == "Yes" else 0
        internet_enc = 1 if internet == "Yes" else 0
        test_prep_enc = 1 if test_prep == "Not Completed" else 0  # LabelEncoder order
        parent_edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        income_map = {'Low': 0, 'Medium': 1, 'High': 2}
        
        student_features = {
            'age': age,
            'attendance_percentage': attendance,
            'study_hours_per_week': study_hours,
            'previous_cgpa': previous_cgpa,
            'internal_marks': internal_marks,
            'participation_score': participation,
            'sleep_hours': sleep_hours,
            'gender_encoded': gender_enc,
            'extracurricular_encoded': extra_enc,
            'internet_access_encoded': internet_enc,
            'parent_education_encoded': parent_edu_map[parent_edu],
            'family_income_encoded': income_map[family_inc],
            'test_preparation_encoded': test_prep_enc
        }
        
        # Get preprocessed data for scaler
        try:
            prep_data = get_preprocessed_data()
            scaler = prep_data['scaler']
            feature_names = prep_data['feature_names']
            
            features = []
            for fname in feature_names:
                features.append(student_features.get(fname, 0))
            
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            lr_model = models['linear_regression']
            rf_model = models['random_forest']
            
            pred_score = np.clip(lr_model.predict(X_scaled)[0], 0, 100)
            pred_level = rf_model.predict(X_scaled)[0]
            level_map = {0: 'Low', 1: 'Average', 2: 'High'}
            probs = rf_model.predict_proba(X_scaled)[0]
            
            result = {
                'predicted_score': round(pred_score, 1),
                'predicted_level': level_map.get(pred_level, 'Unknown'),
                'probabilities': {
                    'Low': round(probs[0] * 100, 1),
                    'Average': round(probs[1] * 100, 1),
                    'High': round(probs[2] * 100, 1)
                }
            }
            
            # Display results
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            level_color = {'High': 'metric-green', 'Average': 'metric-orange', 'Low': 'metric-card'}
            
            with col_r1:
                st.markdown(f"""
                <div class="metric-card metric-blue">
                    <div class="metric-value">{result['predicted_score']}</div>
                    <div class="metric-label">Predicted Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r2:
                color_cls = level_color.get(result['predicted_level'], 'metric-card')
                st.markdown(f"""
                <div class="metric-card {color_cls}">
                    <div class="metric-value">{result['predicted_level']}</div>
                    <div class="metric-label">Performance Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_r3:
                max_prob = max(result['probabilities'].values())
                st.markdown(f"""
                <div class="metric-card metric-purple">
                    <div class="metric-value">{max_prob}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Probability gauge
            fig = go.Figure()
            for level, prob in result['probabilities'].items():
                color = {'Low': '#E17055', 'Average': '#FDCB6E', 'High': '#00B894'}[level]
                fig.add_trace(go.Bar(x=[prob], y=[level], orientation='h',
                                     marker_color=color, text=f"{prob}%",
                                     textposition='auto', name=level))
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Probability (%)", yaxis_title="",
                height=200, margin=dict(t=40, b=40, l=20, r=20),
                font=dict(family="Inter"), showlegend=False,
                xaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig, width="stretch")
            
            # Quick recommendations
            student_raw = {
                'attendance_percentage': attendance,
                'study_hours_per_week': study_hours,
                'previous_cgpa': previous_cgpa,
                'internal_marks': internal_marks,
                'participation_score': participation,
                'sleep_hours': sleep_hours,
                'extracurricular': extracurricular,
                'internet_access': internet,
                'test_preparation': test_prep
            }
            
            recs = generate_recommendations(student_raw, result['predicted_score'], result['predicted_level'])
            if recs:
                st.markdown('<div class="section-header">💡 Quick Recommendations</div>', unsafe_allow_html=True)
                for rec in recs[:5]:
                    priority_class = f"rec-{rec['priority'].lower()}"
                    st.markdown(f"""
                    <div class="rec-card {priority_class}">
                        {rec['icon']} <strong>[{rec['priority']}]</strong> {rec['recommendation']}
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Make sure you've generated the dataset and trained the models first.")


# ============================
# PAGE: STUDENT COMPARISON
# ============================
elif page == "📈 Student Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Student Comparison</h1>
        <p>Compare two students side-by-side with radar charts and detailed metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Student A")
        student_a_id = st.selectbox("Select Student A", sorted(df['student_id'].unique()), key='student_a')
    
    with col2:
        st.markdown("### 🟠 Student B")
        student_b_id = st.selectbox("Select Student B", sorted(df['student_id'].unique()), index=1, key='student_b')
    
    student_a = df[df['student_id'] == student_a_id].iloc[0]
    student_b = df[df['student_id'] == student_b_id].iloc[0]
    
    # Comparison table
    st.markdown('<div class="section-header">📋 Side-by-Side Comparison</div>', unsafe_allow_html=True)
    
    compare_cols = ['gender', 'age', 'attendance_percentage', 'study_hours_per_week',
                    'previous_cgpa', 'internal_marks', 'participation_score',
                    'sleep_hours', 'extracurricular', 'internet_access',
                    'test_preparation', 'final_exam_score', 'performance_level']
    
    comparison_data = []
    for col in compare_cols:
        comparison_data.append({
            'Feature': col.replace('_', ' ').title(),
            f'Student A (ID: {student_a_id})': str(student_a[col]),
            f'Student B (ID: {student_b_id})': str(student_b[col])
        })
    
    st.table(pd.DataFrame(comparison_data))
    
    # Radar chart
    st.markdown('<div class="section-header">🕸️ Radar Comparison</div>', unsafe_allow_html=True)
    
    radar_features = ['attendance_percentage', 'study_hours_per_week', 'previous_cgpa',
                      'internal_marks', 'participation_score', 'sleep_hours']
    radar_labels = ['Attendance', 'Study Hours', 'CGPA', 'Internal Marks', 'Participation', 'Sleep']
    
    # Normalize to 0-100 scale for radar
    max_vals = {'attendance_percentage': 100, 'study_hours_per_week': 40, 'previous_cgpa': 10,
                'internal_marks': 50, 'participation_score': 10, 'sleep_hours': 10}
    
    a_vals = [float(student_a[f]) / max_vals[f] * 100 for f in radar_features]
    b_vals = [float(student_b[f]) / max_vals[f] * 100 for f in radar_features]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=a_vals + [a_vals[0]], theta=radar_labels + [radar_labels[0]],
        fill='toself', fillcolor='rgba(108, 92, 231, 0.2)',
        line=dict(color='#6C5CE7', width=2),
        name=f'Student A (ID: {student_a_id})'
    ))
    fig.add_trace(go.Scatterpolar(
        r=b_vals + [b_vals[0]], theta=radar_labels + [radar_labels[0]],
        fill='toself', fillcolor='rgba(225, 112, 85, 0.2)',
        line=dict(color='#E17055', width=2),
        name=f'Student B (ID: {student_b_id})'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=500, margin=dict(t=40, b=40, l=80, r=80),
        font=dict(family="Inter"),
        legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h')
    )
    st.plotly_chart(fig, width="stretch")
    
    # Score comparison bar chart
    st.markdown('<div class="section-header">📊 Metric Comparison</div>', unsafe_allow_html=True)
    
    compare_metrics = ['final_exam_score', 'attendance_percentage', 'study_hours_per_week',
                       'internal_marks', 'previous_cgpa']
    compare_labels = ['Final Score', 'Attendance %', 'Study Hrs/Wk', 'Internal Marks', 'Prev. CGPA']
    
    fig = go.Figure(data=[
        go.Bar(name=f'Student A (ID: {student_a_id})',
               x=compare_labels,
               y=[float(student_a[c]) for c in compare_metrics],
               marker_color='#6C5CE7'),
        go.Bar(name=f'Student B (ID: {student_b_id})',
               x=compare_labels,
               y=[float(student_b[c]) for c in compare_metrics],
               marker_color='#E17055')
    ])
    fig.update_layout(barmode='group', height=400,
                      margin=dict(t=20, b=40, l=40, r=20),
                      font=dict(family="Inter"))
    st.plotly_chart(fig, width="stretch")


# ============================
# PAGE: RECOMMENDATIONS
# ============================
elif page == "💡 Recommendations":
    st.markdown("""
    <div class="main-header">
        <h1>💡 Performance Recommendations</h1>
        <p>AI-powered personalized recommendations to improve student performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Student selector
    student_id = st.selectbox("Select a Student", sorted(df['student_id'].unique()))
    student = df[df['student_id'] == student_id].iloc[0]
    
    # Student overview
    st.markdown('<div class="section-header">👤 Student Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    level = student['performance_level']
    level_color = {'High': 'metric-green', 'Average': 'metric-orange', 'Low': 'metric-card'}
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-blue">
            <div class="metric-value">{student['student_id']}</div>
            <div class="metric-label">Student ID</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-purple">
            <div class="metric-value">{student['final_exam_score']}</div>
            <div class="metric-label">Exam Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color_cls = level_color.get(level, 'metric-card')
        st.markdown(f"""
        <div class="metric-card {color_cls}">
            <div class="metric-value">{level}</div>
            <div class="metric-label">Performance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{student['attendance_percentage']}%</div>
            <div class="metric-label">Attendance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{student['study_hours_per_week']}</div>
            <div class="metric-label">Study Hrs/Wk</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prepare student data
    student_raw = {
        'attendance_percentage': student['attendance_percentage'],
        'study_hours_per_week': student['study_hours_per_week'],
        'previous_cgpa': student['previous_cgpa'],
        'internal_marks': student['internal_marks'],
        'participation_score': student['participation_score'],
        'sleep_hours': student['sleep_hours'],
        'extracurricular': student['extracurricular'],
        'internet_access': student['internet_access'],
        'test_preparation': student['test_preparation']
    }
    
    # Generate recommendations
    recs = generate_recommendations(student_raw, student['final_exam_score'], level)
    stats = get_summary_stats(recs)
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Recommendations", stats['total'])
    with col2:
        st.metric("🚨 Critical", stats['critical'])
    with col3:
        st.metric("⚠️ High Priority", stats['high'])
    with col4:
        st.metric("Avg. Impact Score", stats['avg_impact'])
    
    # Recommendations list
    st.markdown('<div class="section-header">📋 Recommendations</div>', unsafe_allow_html=True)
    
    for rec in recs:
        priority_class = f"rec-{rec['priority'].lower()}"
        st.markdown(f"""
        <div class="rec-card {priority_class}">
            {rec['icon']} <strong>[{rec['priority']}]</strong> {rec['recommendation']}
            <br><small style="color: #636e72;">Impact Score: {'🔴' * min(rec['impact'], 10)} ({rec['impact']}/10)</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Strengths
    st.markdown('<div class="section-header">⭐ Student Strengths</div>', unsafe_allow_html=True)
    strengths = get_strengths(student_raw)
    
    if strengths:
        strength_html = ""
        for icon, s in strengths:
            strength_html += f'<span class="strength-badge">{icon} {s}</span> '
        st.markdown(strength_html, unsafe_allow_html=True)
    else:
        st.info("Keep working on improving your academic habits to develop strengths!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Student metrics radar
    st.markdown('<div class="section-header">📊 Student Profile</div>', unsafe_allow_html=True)
    
    radar_features = ['attendance_percentage', 'study_hours_per_week', 'previous_cgpa',
                      'internal_marks', 'participation_score', 'sleep_hours']
    radar_labels = ['Attendance', 'Study Hours', 'CGPA', 'Internal Marks', 'Participation', 'Sleep']
    max_vals = {'attendance_percentage': 100, 'study_hours_per_week': 40, 'previous_cgpa': 10,
                'internal_marks': 50, 'participation_score': 10, 'sleep_hours': 10}
    
    vals = [float(student[f]) / max_vals[f] * 100 for f in radar_features]
    
    # Class average
    avg_vals = [float(df[f].mean()) / max_vals[f] * 100 for f in radar_features]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=radar_labels + [radar_labels[0]],
        fill='toself', fillcolor='rgba(108, 92, 231, 0.25)',
        line=dict(color='#6C5CE7', width=2.5),
        name=f'Student {student_id}'
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_vals + [avg_vals[0]], theta=radar_labels + [radar_labels[0]],
        fill='toself', fillcolor='rgba(99, 110, 114, 0.1)',
        line=dict(color='#636E72', width=1.5, dash='dash'),
        name='Class Average'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=450, margin=dict(t=40, b=40, l=80, r=80),
        font=dict(family="Inter"),
        legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h')
    )
    st.plotly_chart(fig, width="stretch")
