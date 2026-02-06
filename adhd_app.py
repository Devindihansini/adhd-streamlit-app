
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Page configuration
st.set_page_config(page_title="ADHD Assessment Tool", layout="wide")

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 Data Analysis", "📈 Visualization", "🔮 ADHD Level Prediction", "📋 Model Performance"]
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('ADHD Symptom Self-Assessment (Responses).xlsx')
    return df

df = load_data()

# Load model and encoders
@st.cache_resource
def load_model():
    with open('adhd_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model()

# Get the target column
target_column = encoders.get('target_column', df.columns[-1])

# Create a clean version of the data for training
def preprocess_data(df):
    df_clean = df.copy()
    columns_to_drop = ['Timestamp', 'Email', 'Break', 'Marks']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns], errors='ignore')
    
    # Encode questions
    response_mapping = encoders['response_mapping']
    question_columns = encoders.get('question_columns', df_clean.columns[2:-1])
    
    for col in question_columns:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].map(response_mapping)
    
    # Encode gender and age
    if 'Gender' in df_clean.columns:
        df_clean['Gender'] = encoders['gender_encoder'].transform(df_clean['Gender'])
    if 'Age' in df_clean.columns:
        df_clean['Age'] = encoders['age_encoder'].transform(df_clean['Age'])
    
    return df_clean

df_clean = preprocess_data(df)

# Helper function to make dataframes Streamlit/Arrow compatible
def make_df_displayable(df):
    """Convert all problematic columns to strings for display"""
    if isinstance(df, pd.Series):
        # Handle Series objects
        return df.astype(str)
    
    df_display = df.copy()
    # Convert all columns to string to ensure PyArrow compatibility
    for col in df_display.columns:
        df_display[col] = df_display[col].astype(str)
    return df_display

# Create a cleaner version of the raw data for display (without problematic columns)
df_display = df.drop(columns=['Timestamp', 'Email', 'Break', 'Marks'], errors='ignore')

# Page 1: Home
if page == "🏠 Home":
    st.title("🎯 ADHD Self-Assessment Tool")
    st.markdown("---")
    
    st.subheader("Welcome!")
    st.write("""
    This app is designed for ADHD symptom assessment.
    You can use the following sections:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Data Analysis")
        st.write("Data preview and statistics")
    
    with col2:
        st.markdown("### 📈 Visualization")
        st.write("Visualize data with graphs and charts")
    
    with col3:
        st.markdown("### 🔮 Prediction")
        st.write("Predict ADHD levels")
    
    st.markdown("---")
    st.info("💡 **Note:** This is only a self-assessment tool. Consult with a doctor for scientific diagnosis.")

# Page 2: Data Analysis
elif page == "📊 Data Analysis":
    st.title("📊 Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Statistics", "Filtering"])
    
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(make_df_displayable(df_display.head(10)))
        
        st.subheader("Data Shape")
        st.write(f"Rows: **{df_display.shape[0]}**, Columns: **{df_display.shape[1]}**")
    
    with tab2:
        st.subheader("Basic Statistics")
        st.dataframe(make_df_displayable(df_display.describe()))
        
        st.subheader("Data Types")
        st.dataframe(make_df_displayable(pd.DataFrame({'Column': df_display.columns, 'Type': df_display.dtypes.values})))
    
    with tab3:
        st.subheader("Data Filtering")
        filter_col = st.selectbox("Select column to filter", df_display.columns)
        
        if pd.api.types.is_datetime64_any_dtype(df_display[filter_col]) or df_display[filter_col].dtype == 'object':
            unique_vals = df_display[filter_col].unique()
            selected_val = st.selectbox("Select value", unique_vals)
            filtered_df = df_display[df_display[filter_col] == selected_val]
        else:
            min_val = float(df_display[filter_col].min())
            max_val = float(df_display[filter_col].max())
            selected_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
            filtered_df = df_display[(df_display[filter_col] >= selected_range[0]) & (df_display[filter_col] <= selected_range[1])]
        
        st.dataframe(make_df_displayable(filtered_df))
        st.write(f"Filtered rows: **{len(filtered_df)}**")

# Page 3: Visualization
elif page == "📈 Visualization":
    st.title("📈 Data Visualization")
    
    chart_type = st.selectbox(
        "Select chart type",
        ["Bar Chart", "Histogram", "Scatter Plot", "Box Plot"]
    )
    
    if chart_type == "Bar Chart":
        col = st.selectbox("Select column", df.select_dtypes(include=['object']).columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'{col} - Bar Chart')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        st.pyplot(fig)
    
    elif chart_type == "Histogram":
        col = st.selectbox("Select column", df.select_dtypes(include=[np.number]).columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[col].dropna(), bins=20, color='lightgreen', edgecolor='black')
        ax.set_title(f'{col} - Histogram')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

# Page 4: Prediction
elif page == "🔮 ADHD Level Prediction":
    st.title("🔮 ADHD Level Prediction")
    
    st.write("Please answer the following questions:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender_options = list(encoders['gender_encoder'].classes_)
            age_options = list(encoders['age_encoder'].classes_)
            gender = st.selectbox("Gender", gender_options)
            age = st.selectbox("Age Group", age_options)
        
        with col2:
            # Sample questions
            q1 = st.selectbox("Do you avoid starting difficult tasks?",
                             ["Never", "Rarely", "Sometimes", "Often", "Always"])
            q2 = st.selectbox("Do you do other things when you need to focus?",
                             ["Never", "Rarely", "Sometimes", "Often", "Always"])
        
        submit = st.form_submit_button("Predict")
    
    if submit:
        with st.spinner("Predicting..."):
            # Get question columns from encoders
            question_columns = encoders.get('question_columns', df_clean.columns[2:])
            
            # Prepare input data with actual column names
            input_data = {
                'Gender': encoders['gender_encoder'].transform([gender])[0],
                'Age': encoders['age_encoder'].transform([age])[0],
            }
            
            # Add the first two questions
            if len(question_columns) > 0:
                input_data[question_columns[0]] = encoders['response_mapping'][q1]
            if len(question_columns) > 1:
                input_data[question_columns[1]] = encoders['response_mapping'][q2]
            
            # Fill remaining questions with default value (Sometimes = 2)
            for q_col in question_columns[2:]:
                input_data[q_col] = 2  # 'Sometimes' as default
            
            # Reorder columns to match training data
            feature_columns = ['Gender', 'Age'] + list(question_columns)
            
            # Create DataFrame in correct order
            input_df = pd.DataFrame([input_data])[feature_columns]
            
            # Make prediction
            prediction = model.predict(input_df)
            prediction_label = encoders['target_encoder'].inverse_transform(prediction)[0]
            
            # Show results
            st.success("✅ Prediction complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Gender", gender)
            with col2:
                st.metric("Age Group", age)
            with col3:
                st.metric("Predicted ADHD Level", prediction_label)
            
            # Details
            st.info(f"""
            **Details:** {prediction_label}
            
            **Note:** This is only an AI-based prediction.
            For scientific diagnosis, consult with a doctor.
            """)

# Page 5: Model Performance
elif page == "📋 Model Performance":
    st.title("📋 Model Performance")
    
    # Prepare features and target
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Make predictions on full data
    y_pred = model.predict(X)
    
    # Convert y_pred to original labels for consistent comparison
    y_pred_labels = encoders['target_encoder'].inverse_transform(y_pred)
    
    tab1, tab2 = st.tabs(["Classification Report", "Confusion Matrix"])
    
    with tab1:
        st.subheader("Classification Report")
        report = classification_report(y, y_pred_labels, target_names=list(encoders['target_encoder'].classes_), output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
    
    with tab2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred_labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=encoders['target_encoder'].classes_,
                    yticklabels=encoders['target_encoder'].classes_,
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

# Footer information
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **📁 Project Files:**
    - adhd_model.pkl (Trained model)
    - encoders.pkl (Encoders)
    - ADHD_Symptom_Self_Assessment.xlsx (Dataset)
    """
)
