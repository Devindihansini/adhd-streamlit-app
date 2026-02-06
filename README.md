s# 🧠 ADHD Self-Assessment Tool

A comprehensive Streamlit-based web application for ADHD symptom assessment with machine learning predictions and detailed data analysis.

## 📋 Overview

This project provides:
- **Data Analysis**: Interactive data exploration and visualization
- **ML Model Training**: Random Forest classifier trained on survey responses
- **ADHD Prediction**: Real-time predictions based on user inputs
- **Performance Metrics**: Classification reports and confusion matrices
- **Data Visualization**: Multiple chart types for data insights

## 🎯 Features

### 🏠 Home Page
- Welcome message and feature overview
- Quick navigation to all sections

### 📊 Data Analysis
- **Data Preview**: View raw data with shape information
- **Statistics**: Numeric and categorical data statistics
- **Filtering**: Search and filter data by column values
- **CSV Export**: Download filtered results

### 📈 Visualization
- **Bar Charts**: Categorical data distribution
- **Histograms**: Numeric data distribution
- **Pie Charts**: Proportion visualization
- Interactive column selection

### 🔮 ADHD Level Prediction
- Interactive form-based input
- Gender and age selection
- ADHD symptom question responses
- Real-time ML-based predictions
- Color-coded confidence metrics

### 📋 Model Performance
- Classification report with precision/recall/F1 scores
- Confusion matrix visualization
- Feature importance analysis
- Top 10 most important features

## 📦 Requirements

- Python 3.8 or higher
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- openpyxl

## 🚀 Installation

### 1. Prerequisites
Ensure Python is installed on your system. Download from [python.org](https://www.python.org/)

### 2. Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit openpyxl
```

Or install all at once:

```bash
pip install pandas>=1.0.0 numpy>=1.19.0 scikit-learn>=0.24.0 matplotlib>=3.3.0 seaborn>=0.11.0 streamlit>=1.0.0 openpyxl>=3.0.0
```

### 3. Verify Installation

```bash
pip list | grep -E "pandas|streamlit|scikit-learn"
```

## 📁 Project Structure

```
Streamlit/
├── train_model.py                           # Script to train the ML model
├── adhd_app.py                              # Streamlit web application
├── adhd_streamlit_app.py                    # Alternative app version
├── install_dependencies.py                  # Dependency installation script
├── README.md                               # This file
├── ADHD Symptom Self-Assessment (Responses).xlsx  # Dataset
├── adhd_model.pkl                          # Trained Random Forest model
├── encoders.pkl                            # Encoder objects
└── requirements.txt                        # Python dependencies list
```

## 🎬 Getting Started

### Step 1: Prepare Your Data

Ensure you have the Excel dataset file:
- **File**: `ADHD Symptom Self-Assessment (Responses).xlsx`
- **Location**: Same directory as the scripts
- **Required Columns**: Timestamp, Email, Gender, Age, Survey Questions, Level

### Step 2: Train the Model (First Time Only)

```bash
python train_model.py
```

This will:
1. Load the Excel dataset
2. Preprocess the data
3. Train the Random Forest model
4. Create `adhd_model.pkl`
5. Create `encoders.pkl`
6. Generate `adhd_app.py`

### Step 3: Run the Streamlit App

```bash
streamlit run adhd_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 📊 Data Format

### Expected Excel Columns

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| Timestamp | DateTime | 2024-01-15 10:30:00 | Will be removed |
| Email | Text | user@email.com | Will be removed |
| Gender | Categorical | Male/Female/Other | Encoded for model |
| Age | Categorical | 18-25, 26-35, etc. | Encoded for model |
| Q1-Q32* | Categorical | Never/Rarely/Sometimes/Often/Always | Survey responses |
| Level | Categorical | Mild/Moderate/Severe/None | Target variable |

*Exact number of questions may vary

### Response Mapping

```
Never    → 0
Rarely   → 1
Sometimes → 2
Often    → 3
Always   → 4
```

## 🛠️ Troubleshooting

### Error: "File not found"
```
Error: File 'ADHD Symptom Self-Assessment (Responses).xlsx' not found!
```
**Solution**: Ensure the Excel file is in the same directory as the script.

### Error: "openpyxl not installed"
```
Error: 'openpyxl' library is not installed.
```
**Solution**: 
```bash
pip install openpyxl
```

### Error: "Model not loaded"
```
Error: Model not loaded. Please check if model files exist.
```
**Solution**: Run `python train_model.py` to generate model files.

### Arrow Serialization Errors
If you see PyArrow errors when displaying data:
- The app automatically converts problematic columns to strings
- Ensure all required libraries are up to date:
```bash
pip install --upgrade streamlit pyarrow pandas
```

### Port Already in Use
```
Error: Address already in use
```
**Solution**:
```bash
streamlit run adhd_app.py --server.port 8502
```

## 🔧 Configuration

### Auto-rerun disabled
To enable auto-rerun when files change:
```bash
streamlit run adhd_app.py --logger.level=debug
```

### Adjust page width
Modify `st.set_page_config()` in adhd_app.py:
```python
st.set_page_config(layout="wide")  # or "centered"
```

## 📈 Model Details

### Algorithm: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Random State**: 42 (for reproducibility)
- **Test Size**: 20% (80% training, 20% testing)
- **Features**: Gender, Age, 30+ survey questions
- **Target**: ADHD Level classification

### Performance Metrics
Achieved via sklearn's `classification_report()`:
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples per class

## 💾 Generated Files

After running `train_model.py`:

### 1. `adhd_model.pkl`
Binary pickle file containing the trained Random Forest model.

### 2. `encoders.pkl`
Dictionary containing:
- `gender_encoder`: LabelEncoder for gender values
- `age_encoder`: LabelEncoder for age groups
- `target_encoder`: LabelEncoder for ADHD levels
- `response_mapping`: Dictionary mapping text responses to numbers
- `question_columns`: List of survey question column names
- `target_column`: Name of the target variable

### 3. `adhd_app.py`
Complete Streamlit application, automatically generated from template.

## 🎓 Understanding the Prediction

When you input responses:
1. **Preprocessing**: Responses encoded to numbers
2. **Feature Engineering**: 30+ question responses processed
3. **Default Values**: Unanswered questions set to "Sometimes" (2)
4. **Prediction**: Model predicts ADHD level
5. **Output**: One of the trained classes (e.g., Mild, Moderate, Severe, None)

**Note**: This is an AI-based assessment tool. For clinical diagnosis, consult a healthcare professional.

## 📝 Usage Example

### Prediction Workflow
1. Open app → Select "🔮 ADHD Level Prediction"
2. Choose Gender: Male/Female/Other
3. Choose Age Group: Under 18/18-25/26-35/etc.
4. Answer 2 sample questions
5. Click "Predict"
6. View predicted ADHD level

### Data Analysis Workflow
1. Open app → Select "📊 Data Analysis"
2. Browse "Data Preview" tab to see raw data
3. Check "Statistics" for numeric summaries
4. Use "Filtering" to search specific values
5. Download filtered results as CSV

## 🔐 Data Privacy

This application:
- ✅ Runs locally on your machine
- ✅ Requires no internet connection
- ✅ Stores no user submissions
- ✅ All data remains on your computer

## 📞 Support

### Common Issues

#### Issue: App crashes on startup
**Solution**: Make sure all packages are installed and dataset exists

#### Issue: Predictions seem inaccurate
**Solution**: Model accuracy depends on training data quality. Retrain with more/better data

#### Issue: Performance is slow
**Solution**: Reduce dataset size or close other applications

## 🔄 Retraining the Model

To retrain with new data:

1. Update the Excel file with new survey responses
2. Run: `python train_model.py`
3. Confirm the model files are regenerated
4. Restart the Streamlit app

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| **streamlit** | Web app framework |
| **pandas** | Data manipulation |
| **scikit-learn** | Machine learning |
| **matplotlib** | Static plotting |
| **seaborn** | Statistical visualization |
| **numpy** | Numerical computing |
| **openpyxl** | Excel file handling |
| **pickle** | Object serialization |

## 📄 License

This project is provided as-is for educational purposes.

## ⚠️ Medical Disclaimer

**This tool is NOT a clinical diagnostic instrument.**

The ADHD Self-Assessment Tool:
- Provides educational information only
- Should not replace professional medical evaluation
- Is not approved by any medical authority
- Cannot be used for self-diagnosis

**If you suspect ADHD, please consult a qualified healthcare professional.**

## 🤝 Contributing

To improve this tool:
1. Collect more quality data
2. Tune model hyperparameters
3. Add more visualization features
4. Improve question wording
5. Test with diverse populations

## ✅ Verification Checklist

Before running the app, verify:
- [ ] Python 3.8+ is installed
- [ ] All packages are installed (`pip list`)
- [ ] Excel dataset file exists in the directory
- [ ] `adhd_model.pkl` exists (or you've run `train_model.py`)
- [ ] `encoders.pkl` exists
- [ ] Internet connection (first run only, for potential dependencies)

## 🎯 Next Steps

1. **First Time**:
   ```bash
   python train_model.py
   streamlit run adhd_app.py
   ```

2. **Regular Use**:
   ```bash
   streamlit run adhd_app.py
   ```

3. **Retrain with New Data**:
   ```bash
   python train_model.py
   streamlit run adhd_app.py
   ```

## 📧 Questions?

Review the comments in the code or check these resources:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Pandas Tutorials](https://pandas.pydata.org/docs/)

---

**Created**: February 2026  
**Version**: 1.0  
**Status**: Production Ready
