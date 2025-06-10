# CSV Data Analyzer Dashboard

A powerful and interactive Streamlit dashboard for data exploration, cleaning, visualization, and basic machine learning modeling on CSV files.

## Features

- **CSV Upload:** Easily upload your own CSV file.
- **Data Validation:** Checks for empty files, missing values, and column consistency.
- **Data Cleaning:** Drop missing values, select columns, and filter rows using pandas query syntax.
- **Data Info:** View data types, missing values, and unique value counts per column.
- **Descriptive Statistics:** Summary statistics for numeric columns.
- **Downloadable Reports:** Download cleaned data and descriptive statistics as CSV files.
- **Visualizations:**
  - Histograms (with adjustable bins)
  - Boxplots (select columns)
  - Correlation heatmaps (select columns)
  - Scatter plots and pairplots (select columns)
  - Bar charts for categorical columns
- **Inferential Section:**  
  Perform basic inferential statistics and hypothesis testing:
  - **Correlation Analysis:** Pearson correlation and p-value for numeric columns.
  - **T-test / ANOVA:** Compare means between groups for a numeric target and categorical feature.
- **Predictive Section:** Build simple regression or classification models interactively.
- **Machine Learning Section:** Try different ML models (Logistic Regression, Random Forest, Linear Regression, Random Forest Regressor) and view metrics.
- **Responsive Layout:** Visualizations and stats are arranged for easy comparison.
- **Help Section:** Expandable instructions for users.

## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/DataAnalysis.git
    cd DataAnalysis
    ```

2. **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

```bash
streamlit run streamlit_csv_analyzer.py
```

Open the provided local URL in your browser.

## Usage

1. Upload a CSV file.
2. Explore, clean, and visualize your data.
3. Download cleaned data or statistics.
4. Try predictive and machine learning models on your dataset.

## Notes

- For best emoji support in plots, ensure your system has an emoji-compatible font (e.g., Segoe UI Emoji on Windows).
- For large files, Streamlit’s caching will speed up repeated operations.
- If you encounter CSV parsing errors, check your file for consistent columns.

## License

MIT License

---

Made with [Streamlit](https://streamlit.io/), Github Copilot, and ❤️ by Grunobuide
