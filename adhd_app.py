
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(page_title="ADHD Assessment Tool", layout="wide")

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📊 Data Analysis", "📈 Visualization", "🔮 ADHD Level Prediction", "📋 Model Performance"]
)

# Load data - Completely fixed Arrow compatibility
@st.cache_data
def load_data():
    try:
        data_path = os.path.join(SCRIPT_DIR, 'ADHD Symptom Self-Assessment (Responses).xlsx')
        df = pd.read_excel(data_path)
        
        # STRONGLY convert ALL columns to Arrow-compatible types
        for col in df.columns:
            # First try to convert datetime columns properly
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Convert to string with proper format
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass
            
            # For ALL object columns, force convert to string
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str)
                except:
                    # If conversion fails, fill NaN and try again
                    df[col] = df[col].fillna('').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty dataframe with proper structure
        return pd.DataFrame()

df = load_data()

# Load model and encoders
@st.cache_resource
def load_model():
    try:  # added ----
        model_path = os.path.join(SCRIPT_DIR, 'adhd_model.pkl')
        encoders_path = os.path.join(SCRIPT_DIR, 'encoders.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        return model, encoders
    except Exception as e:  # added ----
        st.error(f"Error loading model: {e}")
        return None, None

model, encoders = load_model()

# Get the target column
# added ----
if encoders:
    target_column = encoders.get('target_column', df.columns[-1] if len(df.columns) > 0 else 'Level')
else:
    target_column = df.columns[-1] if len(df.columns) > 0 else 'Level'

# Function to get clean dataframe for display (Arrow compatible)
def get_display_df():
    """Return Arrow-compatible dataframe for display"""
    display_df = df.copy()
    
    # Ensure all columns are strings for Arrow compatibility
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            try:
                display_df[col] = display_df[col].astype(str)
            except:
                display_df[col] = display_df[col].fillna('').astype(str)
    
    return display_df   

# Create a clean version of the data for training
def preprocess_data(df):
    df_clean = df.copy()
    # Remove unnecessary columns
    columns_to_drop = ['Timestamp', 'Email', 'Break', 'Marks']
    for col in columns_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col], errors='ignore')
    
    if df_clean.empty:
        return df_clean
    
    # Encode questions if encoders exist
    if encoders and 'response_mapping' in encoders:
        response_mapping = encoders['response_mapping']
        question_columns = encoders.get('question_columns', [])
        
        # If no question columns in encoders, try to infer them
        if not question_columns:
            # Skip first 2 columns (Gender, Age) and last column (target)
            if len(df_clean.columns) > 3:
                question_columns = df_clean.columns[2:-1]
        
        for col in question_columns:
            if col in df_clean.columns:
                try:
                    df_clean[col] = df_clean[col].map(response_mapping)
                except:
                    pass
    
    # Encode gender and age
    if 'Gender' in df_clean.columns and encoders and 'gender_encoder' in encoders:
        try:
            df_clean['Gender'] = encoders['gender_encoder'].transform(df_clean['Gender'].astype(str))
        except:
            pass
    
    if 'Age' in df_clean.columns and encoders and 'age_encoder' in encoders:
        try:
            df_clean['Age'] = encoders['age_encoder'].transform(df_clean['Age'].astype(str))
        except:
            pass

    # Encode target labels if available so performance metrics use consistent numeric labels
    if target_column in df_clean.columns and encoders and 'target_encoder' in encoders:
        try:
            if df_clean[target_column].dtype == object:
                df_clean[target_column] = encoders['target_encoder'].transform(df_clean[target_column].astype(str))
        except Exception:
            pass
    
    return df_clean

df_clean = preprocess_data(df)

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

        # Use the Arrow-compatible display dataframe
        display_df = get_display_df()
        st.dataframe(display_df.head(10))
        
        st.subheader("Data Shape")
        st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
        
        # Show column info
        st.subheader("Column Information")
        if not df.empty:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)
    
    with tab2:
        st.subheader("Basic Statistics")
        
        if not df.empty:
            # Show statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.write("Numeric Columns Statistics:")
                st.dataframe(df[numeric_cols].describe())
            else:
                st.info("No numeric columns found")
            
            # Show unique value counts for categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                st.subheader("Categorical Columns")
                selected_col = st.selectbox("Select column to view value counts", cat_cols)
                if selected_col:
                    value_counts = df[selected_col].value_counts()
                    st.dataframe(value_counts)
        else:
            st.warning("No data available")
    
    with tab3:
        st.title("🔍 Data Filtering")
        
        if df.empty:
            st.warning("No data available for filtering")
        else:
            # Use display dataframe
            display_df = get_display_df()
            
            # Select column to filtesr
            filter_col = st.selectbox("Select column to filter", display_df.columns)
            
            if filter_col:
                # Get unique values
                unique_vals = display_df[filter_col].dropna().unique()
                
                if len(unique_vals) <= 20:  # For columns with few unique values
                    selected_vals = st.multiselect(
                        f"Select values from '{filter_col}'",
                        options=unique_vals.tolist(),
                        default=unique_vals[:min(3, len(unique_vals))].tolist()
                    )
                    
                    if selected_vals:
                        mask = display_df[filter_col].isin(selected_vals)
                        filtered_data = display_df[mask]
                    else:
                        filtered_data = display_df
                
                else:  # For columns with many values, use search
                    search_term = st.text_input(f"Search in '{filter_col}' (leave empty to show all)")
                    
                    if search_term:
                        try:
                            # Convert to string and search
                            mask = display_df[filter_col].astype(str).str.contains(
                                search_term, case=False, na=False
                            )
                            filtered_data = display_df[mask]
                        except:
                            filtered_data = display_df
                    else:
                        filtered_data = display_df
                
                # Show filtered results
                st.subheader(f"Filtered Results ({len(filtered_data)} rows)")
                
                # Simple display without pagination issues
                if len(filtered_data) > 100:
                    st.write(f"Showing first 100 of {len(filtered_data)} rows")
                    st.dataframe(filtered_data.head(100))
                else:
                    st.dataframe(filtered_data)
                
                # Download option
                if len(filtered_data) > 0:
                    csv = filtered_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download filtered data as CSV",
                        data=csv,
                        file_name="filtered_adhd_data.csv",
                        mime="text/csv"
                    )

# Page 3: Visualization
elif page == "📈 Visualization":
    st.title("📈 Data Visualization")
    
    if df.empty:
        st.warning("No data available for visualization")
    else:
        chart_type = st.selectbox(
            "Select chart type",
            ["Bar Chart", "Histogram", "Pie Chart"]
        )
        
        if chart_type == "Bar Chart":
            # Get categorical columns
            object_cols = [col for col in df.columns if df[col].dtype == 'object']
            
            if object_cols:
                selected_col = st.selectbox("Select categorical column", object_cols)
                
                # Get value counts
                value_counts = df[selected_col].value_counts().head(15)  # Limit to top 15
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title(f'Top Values in {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No categorical columns available for bar chart")
        
        elif chart_type == "Histogram":
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select numeric column", numeric_cols)
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df[selected_col].dropna(), bins=20, color='lightgreen', 
                       edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No numeric columns available for histogram")
        
        elif chart_type == "Pie Chart":
            # Get categorical columns with few unique values
            suitable_cols = []
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() <= 10:
                    suitable_cols.append(col)
            
            if suitable_cols:
                selected_col = st.selectbox("Select column for pie chart", suitable_cols)
                
                # Get value counts
                value_counts = df[selected_col].value_counts()
                
                # Create pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(value_counts.values, labels=value_counts.index, 
                      autopct='%1.1f%%', startangle=90)
                ax.set_title(f'Distribution of {selected_col}')
                ax.axis('equal')  # Equal aspect ratio ensures pie is circular
                st.pyplot(fig)
            else:
                st.info("No suitable columns for pie chart (need columns with ≤10 unique values)")

# Page 4: Prediction
elif page == "🔮 ADHD Level Prediction":
    st.title("🔮 ADHD Level Prediction")
    
    if model is None or encoders is None:
        st.error("Model not loaded. Please check if model files exist.")
        st.info("Required files: 'adhd_model.pkl' and 'encoders.pkl'")
    else:
        st.markdown("<p style='font-size:18px; margin-bottom: 12px;'>Please answer the following questions:</p>", unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            # Get response options
            response_options = ["Never", "Rarely", "Sometimes", "Often", "Always"]
            
            # Get Gender and Age options
            if 'gender_encoder' in encoders:
                gender_options = list(encoders['gender_encoder'].classes_)
            else:
                gender_options = ['Male', 'Female', 'Other']
            
            if 'age_encoder' in encoders:
                age_options = list(encoders['age_encoder'].classes_)
            else:
                age_options = ['Under 18', '18-25', '26-35', '36-45', '46+']
            
            # Demographics section
            st.subheader("📋 Demographics")
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("Gender", gender_options)
            
            with col2:
                age = st.selectbox("Age Group", age_options)
            
            # Get actual question columns from encoders
            question_columns = encoders.get('question_columns', [])
            
            # Questions section
            st.subheader("❓ ADHD Assessment Questions")
            st.write(f"Please answer all {len(question_columns)} questions below:")
            
            # Create a dictionary to store all answers
            answers = {}
            
            # Display questions in 2 columns
            cols = st.columns(2)
            col_idx = 0
            
            if question_columns:
                for idx, q_col in enumerate(question_columns):
                    with cols[col_idx % 2]:
                        # Clean question text for display
                        q_display = q_col.strip() if isinstance(q_col, str) else str(q_col)
                        
                        # Question number
                        st.markdown(
                            f"<div style='font-size:18px; font-weight:600; margin-bottom: 4px;'>Q{idx + 1}. {q_display}</div>",
                            unsafe_allow_html=True
                        )
                        
                        # Response selectbox
                        response = st.selectbox(
                            f"Answer for Q{idx + 1}",
                            response_options,
                            key=f"q_{idx}",
                            label_visibility="collapsed"
                        )
                        answers[q_col] = response
                    
                    col_idx += 1
            else:
                st.warning("No questions found in model encoders!")
            
            # Submit button
            st.markdown("---")
            submit = st.form_submit_button("🔮 Predict ADHD Level", use_container_width=True)
        
        if submit:
            with st.spinner("Analyzing your responses..."):
                try:
                    # Prepare input data
                    input_data = {
                        'Gender': encoders['gender_encoder'].transform([gender])[0],
                        'Age': encoders['age_encoder'].transform([age])[0]
                    }
                    
                    # Add all question responses
                    if question_columns and answers:
                        for q_col in question_columns:
                            if q_col in answers:
                                # Get the response value from mapping
                                response = answers[q_col]
                                input_data[q_col] = encoders['response_mapping'][response]
                    
                    # Create feature array in correct order
                    feature_columns = ['Gender', 'Age'] + question_columns
                    
                    # Create DataFrame
                    input_df = pd.DataFrame([input_data])
                    
                    # Ensure all feature columns exist
                    for col in feature_columns:
                        if col not in input_df.columns:
                            input_df[col] = 2  # Default 'Sometimes'
                    
                    # Reorder columns
                    input_df = input_df[feature_columns]
                    
                    # Make prediction
                    prediction = model.predict(input_df)
                    prediction_label = encoders['target_encoder'].inverse_transform(prediction)[0]
                    
                    # Compute separate AD and HD scores from the answered questions
                    split_index = 16 if len(question_columns) >= 16 else len(question_columns) // 2
                    ad_questions = question_columns[:split_index]
                    hd_questions = question_columns[split_index:]
                    
                    ad_score = sum(
                        encoders['response_mapping'][answers[q]]
                        for q in ad_questions
                        if q in answers
                    )
                    hd_score = sum(
                        encoders['response_mapping'][answers[q]]
                        for q in hd_questions
                        if q in answers
                    )
                    total_score = ad_score + hd_score
                    
                    # Show results
                    st.success("✅ Prediction Complete!")
                    
                    # Display results in metrics
                    score_col1, score_col2, score_col3, score_col4, score_col5 = st.columns(5)
                    
                    with score_col1:
                        st.metric("Gender", gender)
                    with score_col2:
                        st.metric("Age Group", age)
                    with score_col3:
                        st.metric("Predicted ADHD Level", prediction_label)
                    with score_col4:
                        st.metric("Total ADHD Score", total_score)
                    with score_col5:
                        st.metric("HD Score", hd_score)
                    
                    st.markdown("---")
                    st.subheader("📊 Assessment Results")
                    st.info(f"""
                    **Your Predicted ADHD Level:** {prediction_label}
                    
                    **Total ADHD Score:** {total_score}
                    **Attention Deficit (AD) Score:** {ad_score}
                    **Hyperactivity/Impulsivity (HD) Score:** {hd_score}
                    
                    **Details:** Based on your answers to all {len(question_columns)} assessment questions, 
                    the AI model predicts your ADHD level to be **{prediction_label}**.
                    
                    ⚠️ **Important Note:** This is only an AI-based prediction and should not be considered a medical diagnosis. 
                    For a scientific and professional diagnosis, please consult with a qualified healthcare provider or doctor.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("Please make sure you have answered all questions and that the model files are properly trained.")
                    import traceback
                    st.write(traceback.format_exc())

# Page 5: Model Performance
elif page == "📋 Model Performance":
    st.title("📋 Model Performance")
    
    if model is None or encoders is None or df_clean.empty:
        st.warning("Model or data not available for performance analysis")
    else:
        # Get features and target
        if target_column in df_clean.columns:
            X = df_clean.drop(columns=[target_column], errors='ignore')
            y = df_clean[target_column]
            y_true = y
            
            # Normalize target labels to numeric codes if original values are strings
            if y_true.dtype == object:
                try:
                    y_true = encoders['target_encoder'].transform(y_true.astype(str))
                except Exception:
                    pass
            
            if len(X) == 0 or len(y_true) == 0:
                st.warning("No data available for performance analysis")
            else:
                # Make predictions
                y_pred = model.predict(X)
                
                # Ensure both y_true and y_pred are in the same format (strings)
                y_pred = np.array([str(x) for x in y_pred])
                if hasattr(y_true, 'values'):
                    y_true = np.array([str(x) for x in y_true.values])
                else:
                    y_true = np.array([str(x) for x in y_true])
                
                tab1, tab2 = st.tabs(["Classification Report", "Confusion Matrix"])
                
                with tab1:
                    st.subheader("Classification Report")
                    try:
                        # Get unique labels from actual predictions and true values
                        unique_labels = sorted(set(list(y_true) + list(y_pred)))
                        report = classification_report(
                            y_true, 
                            y_pred, 
                            labels=unique_labels,
                            output_dict=True
                        )
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.2f}"))
                    except Exception as e:
                        st.error(f"Error generating classification report: {e}")
                        st.info("Try retraining the model with the training script.")
                
                with tab2:
                    st.subheader("Confusion Matrix")
                    try:
                        # Get unique labels from actual predictions and true values
                        unique_labels = sorted(set(list(y_true) + list(y_pred)))
                        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, 
                            annot=True, 
                            fmt='d', 
                            cmap='Blues',
                            xticklabels=unique_labels,
                            yticklabels=unique_labels,
                            ax=ax
                        )
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error creating confusion matrix: {e}")
                
                # Show feature importance if available
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Top 10 Feature Importances")
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(range(len(feature_importance)), feature_importance['Importance'])
                    ax.set_yticks(range(len(feature_importance)))
                    ax.set_yticklabels(feature_importance['Feature'])
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importance')
                    ax.invert_yaxis()  # Most important at top
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.warning(f"Target column '{target_column}' not found in data")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **📁 Project Files:**
    - adhd_model.pkl (Trained model)
    - encoders.pkl (Encoders)
    - ADHD Symptom Self-Assessment (Responses).xlsx (Dataset)
    """
)
