🚀 Features
📁 1. File Upload
Upload any CSV dataset
Automatically handles:
Duplicate columns

Data type consistency
🔍 2. Data Overview
Preview dataset
Summary statistics
Missing values analysis
Data types inspection

🎛️ 3. Interactive Filtering
Filter numeric columns using sliders
Filter categorical columns using multi-select
Save filtered dataset for further analysis

📈 4. Exploratory Data Analysis (EDA)
Distribution plots (Histogram + Boxplot)
Scatter plots with correlation values
Correlation heatmap
Categorical value counts visualization

🧹 5. Data Cleaning
Missing value handling:
Drop
Mean
Median
Mode
Column removal
Outlier detection using IQR method
Outlier handling:
Remove outliers
Cap (Winsorization)

🔄 6. Data Transformation
Scaling:
Min-Max Scaling
Standard Scaling
Encoding:
Label Encoding
One-Hot Encoding

📉 7. Dimensionality Reduction
PCA (Principal Component Analysis)
Select number of components interactively

🔁 8. Data Evolution Tracking
Compare:
Original dataset
Processed dataset

⬇️ 9. Download
Export processed dataset as CSV

🛠️ Tech Stack
Python
Streamlit
Pandas
NumPy
Plotly
Scikit-learn
📦 Installation
pip install streamlit pandas numpy plotly scikit-learn
▶️ How to Run
python -m streamlit run app.py
📌 Project Structure

📁 project-folder
│── app.py
│── README.md

💡 Use Cases
Data preprocessing for Machine Learning
Quick EDA for datasets
Academic projects and demos
Data cleaning pipelines

⚠️ Notes
Works best with clean CSV files
Automatically handles basic preprocessing issues
PCA requires numeric or encoded data

⭐ Acknowledgements
Streamlit community
Scikit-learn documentation
Plotly visualization tools
