import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import matplotlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


if sys.platform.startswith('win'):
    matplotlib.rcParams['font.family'] = 'Segoe UI Emoji'
elif sys.platform == 'darwin':
    matplotlib.rcParams['font.family'] = 'Apple Color Emoji'
else:
    matplotlib.rcParams['font.family'] = 'Noto Color Emoji'
st.title("CSV Data Analyzer")
with st.expander("ℹ️ How to use this dashboard", expanded=False):
    st.markdown("""
    **Instructions:**
    - **Upload your CSV file** using the uploader above.
    - **Data Cleaning Options:** Drop rows with missing values, select columns, or filter rows using pandas query syntax.
    - **Data Info:** View data types, missing values, and unique value counts for each column.
    - **Descriptive Statistics:** See summary statistics for numeric columns.
    - **Download Data & Reports:** Download the cleaned data or descriptive statistics as CSV files.
    - **Visualizations:** 
        - Adjust histogram bins and select columns for each plot.
        - Use the pairplot and scatter plot for deeper numeric analysis.
        - Bar charts are available for categorical columns.
    - **Tip:** Use the filter box to subset your data (e.g., `age > 30 and gender == "Male"`).
    """)
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def validate_data(df):
    errors = []
    if df.empty:
        errors.append("The uploaded CSV file is empty.")
    if df.isnull().all().all():
        errors.append("All values in the CSV are missing.")
    if df.shape[1] < 2:
        errors.append("The CSV should have at least two columns.")
    return errors

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        errors = validate_data(df)
        if errors:
            st.error("Data validation failed:")
            for err in errors:
                st.write(f"- {err}")
        else:
            # --- Data Cleaning Options ---
            st.subheader("Data Cleaning Options")
            drop_na = st.checkbox("Drop rows with missing values")
            columns = st.multiselect("Select columns to analyze", df.columns, default=list(df.columns))
            filter_query = st.text_input("Filter rows (e.g., `column > 10`)", "")

            # Apply cleaning
            df_clean = df.copy()
            if drop_na:
                df_clean = df_clean.dropna()
            if columns:
                df_clean = df_clean[columns]
            if filter_query:
                try:
                    df_clean = df_clean.query(filter_query)
                except Exception as e:
                    st.warning(f"Filter error: {e}")

            st.subheader("Cleaned Data Preview")
            st.dataframe(df_clean.head())
                        # --- Data Info ---
            st.subheader("Data Info")
            info_df = pd.DataFrame({
                "Data Type": df_clean.dtypes,
                "Missing Values": df_clean.isnull().sum(),
                "Unique Values": df_clean.nunique()
            })
            st.dataframe(info_df)
            st.subheader("Descriptive Statistics")
            st.write(df_clean.describe())
                        # --- Downloadable Reports ---
            st.subheader("Download Data & Reports")
            csv_clean = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Cleaned Data as CSV",
                data=csv_clean,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

            stats_csv = df_clean.describe().to_csv().encode('utf-8')
            st.download_button(
                label="Download Descriptive Statistics as CSV",
                data=stats_csv,
                file_name="descriptive_statistics.csv",
                mime="text/csv"
            )
            numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                st.subheader("Histogram")
                col = st.selectbox("Select column for histogram", numeric_cols)
                bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)
                fig, ax = plt.subplots()
                sns.histplot(df_clean[col], bins=bins, kde=True, ax=ax)
                st.pyplot(fig)

                st.subheader("Boxplot")
                box_col = st.multiselect("Select columns for boxplot", numeric_cols, default=numeric_cols)
                if box_col:
                    fig2, ax2 = plt.subplots()
                    sns.boxplot(data=df_clean[box_col], ax=ax2)
                    st.pyplot(fig2)
                else:
                    st.info("Select at least one column for boxplot.")

                st.subheader("Correlation Heatmap")
                corr_cols = st.multiselect("Columns for correlation", numeric_cols, default=numeric_cols)
                if len(corr_cols) >= 2:
                    fig3, ax3 = plt.subplots()
                    sns.heatmap(df_clean[corr_cols].corr(), annot=True, cmap="coolwarm", ax=ax3)
                    st.pyplot(fig3)
                    st.write("Correlation matrix:")
                    st.dataframe(df_clean[corr_cols].corr())
                else:
                    st.info("Select at least two columns for correlation heatmap.")
                            # --- Additional Visualizations ---
                st.subheader("Scatter Plot")
                if len(numeric_cols) >= 2:
                    scatter_cols = st.multiselect(
                        "Select columns for scatter plot (pick two)", numeric_cols, default=numeric_cols[:2], key="scatter_cols"
                    )
                    if len(scatter_cols) == 2:
                        fig4, ax4 = plt.subplots()
                        sns.scatterplot(data=df_clean, x=scatter_cols[0], y=scatter_cols[1], ax=ax4)
                        st.pyplot(fig4)
                    else:
                        st.info("Select exactly two columns for scatter plot.")
                else:
                    st.info("Need at least two numeric columns for scatter plot.")

                st.subheader("Pairplot")
                if len(numeric_cols) >= 2:
                    pair_cols = st.multiselect(
                        "Select columns for pairplot", numeric_cols, default=numeric_cols, key="pairplot_cols"
                    )
                    if len(pair_cols) >= 2:
                        fig5 = sns.pairplot(df_clean[pair_cols])
                        st.pyplot(fig5)
                    else:
                        st.info("Select at least two columns for pairplot.")
                else:
                    st.info("Need at least two numeric columns for pairplot.")
            else:
                st.info("No numeric columns found for plotting.")
             # --- Predictive Section ---
            st.header("🔮 Predictive Section")
            st.markdown("Select a target column and features to build a simple predictive model.")

            all_cols = df_clean.columns.tolist()
            target_col = st.selectbox("Select target column", all_cols)
            feature_cols = st.multiselect(
                "Select feature columns", [col for col in all_cols if col != target_col], default=[col for col in all_cols if col != target_col]
            )

            if target_col and feature_cols:
                X = df_clean[feature_cols]
                y = df_clean[target_col]

                # Handle categorical variables
                X = pd.get_dummies(X, drop_first=True)
                if y.dtype == 'O':
                    y = pd.factorize(y)[0]

                test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2)
                random_state = 42

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Choose model type
                if df_clean[target_col].dtype in [np.float64, np.int64] and len(np.unique(y)) > 10:
                    st.write("Regression problem detected.")
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write("**R² score:**", r2_score(y_test, y_pred))
                    st.write("**MSE:**", mean_squared_error(y_test, y_pred))
                    st.write("**Predictions (first 10):**")
                    st.write(pd.DataFrame({"Actual": y_test[:10], "Predicted": y_pred[:10]}))
                else:
                    st.write("Classification problem detected.")
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
                    st.write("**Classification report:**")
                    st.text(classification_report(y_test, y_pred))
                    st.write("**Predictions (first 10):**")
                    st.write(pd.DataFrame({"Actual": y_test[:10], "Predicted": y_pred[:10]}))
            else:
                st.info("Select a target and at least one feature column for prediction.")
            # --- Machine Learning Modeling Section ---
            st.header("🤖 Machine Learning Modeling")
            st.markdown("Try different machine learning models on your data. Select a target and features, then choose a model to train and evaluate.")

            ml_target_col = st.selectbox("Select target column for ML", all_cols, key="ml_target")
            ml_feature_cols = st.multiselect(
                "Select feature columns for ML", [col for col in all_cols if col != ml_target_col], default=[col for col in all_cols if col != ml_target_col], key="ml_features"
            )

            ml_model_type = st.selectbox(
                "Choose model type",
                ["Logistic Regression (classification)", "Random Forest (classification)", "Linear Regression (regression)", "Random Forest (regression)"]
            )

            if ml_target_col and ml_feature_cols:
                X_ml = df_clean[ml_feature_cols]
                y_ml = df_clean[ml_target_col]

                # Handle categorical variables
                X_ml = pd.get_dummies(X_ml, drop_first=True)
                if y_ml.dtype == 'O':
                    y_ml = pd.factorize(y_ml)[0]

                test_size_ml = st.slider("Test size (fraction) for ML", 0.1, 0.5, 0.2, key="ml_test_size")
                random_state_ml = 42

                X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=test_size_ml, random_state=random_state_ml)

                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

                if "classification" in ml_model_type:
                    if "Logistic" in ml_model_type:
                        model_ml = LogisticRegression(max_iter=1000)
                    else:
                        model_ml = RandomForestClassifier()
                    model_ml.fit(X_train_ml, y_train_ml)
                    y_pred_ml = model_ml.predict(X_test_ml)
                    st.write("**Accuracy:**", accuracy_score(y_test_ml, y_pred_ml))
                    st.write("**Classification report:**")
                    st.text(classification_report(y_test_ml, y_pred_ml))
                    st.write("**Predictions (first 10):**")
                    st.write(pd.DataFrame({"Actual": y_test_ml[:10], "Predicted": y_pred_ml[:10]}))
                else:
                    if "Linear" in ml_model_type:
                        model_ml = LinearRegression()
                    else:
                        model_ml = RandomForestRegressor()
                    model_ml.fit(X_train_ml, y_train_ml)
                    y_pred_ml = model_ml.predict(X_test_ml)
                    st.write("**R² score:**", r2_score(y_test_ml, y_pred_ml))
                    st.write("**MSE:**", mean_squared_error(y_test_ml, y_pred_ml))
                    st.write("**Predictions (first 10):**")
                    st.write(pd.DataFrame({"Actual": y_test_ml[:10], "Predicted": y_pred_ml[:10]}))
            else:
                st.info("Select a target and at least one feature column for ML modeling.")
    except pd.errors.ParserError as e:
        st.error(f"CSV parsing error: {e}")
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
else:
    st.info("Please upload a CSV file to begin.")