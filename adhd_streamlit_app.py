# -*- coding: utf-8 -*-

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import os
import warnings
import sys # added -----
warnings.filterwarnings('ignore')

pd.set_option('future.no_silent_downcasting', True)  # added

# added ----
# Try to import openpyxl
try:
    import openpyxl
except ImportError:
    print("Error: 'openpyxl' library is not installed.")
    print("Please install it by running: pip install openpyxl")
    sys.exit(1)
        
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("\nPlease install required packages using:")
    print("pip install pandas numpy matplotlib seaborn scikit-learn openpyxl streamlit")
    sys.exit(1)

# ----

# Set the path to your Excel file (update this to your actual file path)
excel_file_path = 'E:\\Campus\\Final_Project\\Streamlit\\ADHD Symptom Self-Assessment (Responses).xlsx'

# Check if file exists
if not os.path.exists(excel_file_path):
    print(f"Error: File '{excel_file_path}' not found!")
    print("Please make sure the Excel file is in the same directory as this script.")
    print("Current directory:", os.getcwd())

    # ----
    # added
    # List files in current directory
    print("\nFiles in current directory:")
    for file in os.listdir('.'):
        if file.endswith(('.xlsx', '.xls')):
            print(f"  - {file}")
    # ----

    exit(1)

# Load the Excel sheet
try: # added
    df = pd.read_excel(excel_file_path)
    print("✓ Successfully loaded Excel file")  # added
    print("Data shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

# ----
# added
except Exception as e:
    print(f"Error loading Excel file: {e}")
    print("\nPossible solutions:")
    print("1. Make sure the file is not open in Excel")
    print("2. Check if the file is corrupted")
    print("3. Install openpyxl: pip install openpyxl")
    exit(1)
# ----

print("\n" + "="*50)
print("Check data")
print("="*50)

# Details of data
print("Columns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# Check the having 'Level ' row
if 'Level' in df.columns:
    print("\nLevel distribution:")
    print(df['Level'].value_counts())
else:
    print("'Level' column not found. Checking last column...")
    print(df.iloc[:, -1].value_counts())

print("\n" + "="*50)
print("2. Data pre-processing and model training")
print("="*50)

print("Cleaning data...")

# Remove unnecessary rows
columns_to_remove  = ['Timestamp', 'Email', 'Break', 'Marks']
df = df.drop(columns=[c for c in columns_to_remove if c in df.columns], errors='ignore')

# Create a clean copy of the dataframe
df_clean = df.copy()

# Get the final row as a target
target_column = df_clean.columns[-1]
print(f"Target column: {target_column}")

# Assign numerical values to responses
response_mapping = {
    'Never': 0,
    'Rarely': 1,
    'Sometimes': 2,
    'Often': 3,
    'Always': 4
}

# Encode the row of the questions
question_columns = df_clean.columns[2:-1]  # Skip first 2 columns (Gender, Age)
for col in question_columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].map(response_mapping)

# Encode age and gender
le_gender = LabelEncoder()
df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])

le_age = LabelEncoder()
df_clean['Age'] = le_age.fit_transform(df_clean['Age'])

# Encode the target
le_target = LabelEncoder()
df_clean[target_column] = le_target.fit_transform(df_clean[target_column])

print("✓ Data preprocessing complete") # added
print("\nData after preprocessing:") # added
print(df_clean.head())

print("\n" + "="*50)
print("Train the model")
print("="*50)

# Separate features and target
X = df_clean.drop(columns=[target_column])
y = df_clean[target_column]

# Split data without stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# added
print("✓ Model training complete")

# Test performance
y_pred = model.predict(X_test)

# Convert the numerical classes from le_target.classes_ to strings for display purposes
string_class_names = [str(cls) for cls in le_target.classes_]

print("Classification Report:")
print(classification_report(y_test, y_pred, labels=sorted(y.unique()), target_names=string_class_names))

print("\n" + "="*50)
print("Save The model")
print("="*50)

# Save the model
try: # added
    with open('adhd_model.pkl', 'wb') as f:
        pickle.dump(model, f)
# added ----
    print("✓ Model saved as 'adhd_model.pkl'")
except Exception as e:
    print(f"Error saving model: {e}")
# ----

# Save encoders
# added
try:
    encoders = {
        'gender_encoder': le_gender,
        'age_encoder': le_age,
        'target_encoder': le_target,
        'response_mapping': response_mapping,
        'question_columns': question_columns.tolist(),
        'target_column': target_column
    }

    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    # added ----
    print("✓ Encoders saved as 'encoders.pkl'")
except Exception as e:
    print(f"Error saving encoders: {e}")
# ----

print("Model and encoders saved successfully!")

# Create a separate Streamlit app file
app_code = '''
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        df = pd.read_excel('ADHD Symptom Self-Assessment (Responses).xlsx')
        
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
        with open('adhd_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
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
        st.write("Please answer the following questions:")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Get options from encoders
                if 'gender_encoder' in encoders:
                    gender_options = list(encoders['gender_encoder'].classes_)
                else:
                    gender_options = ['Male', 'Female', 'Other']
                
                if 'age_encoder' in encoders:
                    age_options = list(encoders['age_encoder'].classes_)
                else:
                    age_options = ['Under 18', '18-25', '26-35', '36-45', '46+']
                
                gender = st.selectbox("Gender", gender_options)
                age = st.selectbox("Age Group", age_options)
            
            with col2:
                # Use the actual question columns from encoders for sample questions
                response_options = ["Never", "Rarely", "Sometimes", "Often", "Always"]
                
                # Get actual question columns
                question_columns = encoders.get('question_columns', [])
                
                if len(question_columns) >= 2:
                    # Use first two questions
                    q1_text = question_columns[0][:50] + "..." if len(question_columns[0]) > 50 else question_columns[0]
                    q2_text = question_columns[1][:50] + "..." if len(question_columns[1]) > 50 else question_columns[1]
                    
                    q1 = st.selectbox(q1_text, response_options)
                    q2 = st.selectbox(q2_text, response_options)
                else:
                    # Default questions if no question columns
                    q1 = st.selectbox("Do you avoid starting difficult tasks?", response_options)
                    q2 = st.selectbox("Do you do other things when you need to focus?", response_options)
            
            submit = st.form_submit_button("Predict")
        
        if submit:
            with st.spinner("Predicting..."):
                try:
                    # Prepare input data
                    input_data = {
                        'Gender': encoders['gender_encoder'].transform([gender])[0],
                        'Age': encoders['age_encoder'].transform([age])[0]
                    }
                    
                    # Get question columns from encoders
                    question_columns = encoders.get('question_columns', [])
                    
                    # Map the two answered questions
                    if len(question_columns) >= 1:
                        input_data[question_columns[0]] = encoders['response_mapping'][q1]
                    if len(question_columns) >= 2:
                        input_data[question_columns[1]] = encoders['response_mapping'][q2]
                    
                    # Fill remaining questions with default value (Sometimes = 2)
                    for q_col in question_columns[2:]:
                        input_data[q_col] = 2  # 'Sometimes' as default
                    
                    # Create feature array in correct order
                    feature_columns = ['Gender', 'Age'] + question_columns
                    
                    # Create DataFrame ensuring all columns exist
                    input_df = pd.DataFrame([input_data])
                    
                    # Make sure all feature columns exist in input_df
                    for col in feature_columns:
                        if col not in input_df.columns:
                            input_df[col] = 2  # Default value
                    
                    # Reorder columns to match training data
                    input_df = input_df[feature_columns]
                    
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
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("Try answering all questions and make sure the model files are properly trained.")

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
            
            if len(X) == 0 or len(y) == 0:
                st.warning("No data available for performance analysis")
            else:
                # Make predictions
                y_pred = model.predict(X)
                
                tab1, tab2 = st.tabs(["Classification Report", "Confusion Matrix"])
                
                with tab1:
                    st.subheader("Classification Report")
                    try:
                        report = classification_report(
                            y, 
                            y_pred, 
                            target_names=list(encoders['target_encoder'].classes_), 
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
                        cm = confusion_matrix(y, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, 
                            annot=True, 
                            fmt='d', 
                            cmap='Blues',
                            xticklabels=encoders['target_encoder'].classes_,
                            yticklabels=encoders['target_encoder'].classes_,
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
'''

# Create Streamlit app file
with open('adhd_app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)

print("\n" + "="*50)
print("Next Steps:") # added
print("="*50)
print("1. Install Streamlit if not already installed:")
print("   pip install streamlit")
print("\n2. Run the Streamlit app:")                 
print("streamlit run adhd_app.py")
print("\nMake sure all these files arse in the same directory:")
print("- ADHD Symptom Self-Assessment (Responses).xlsx")
print("- adhd_model.pkl")
print("- encoders.pkl")
print("- adhd_app.py")