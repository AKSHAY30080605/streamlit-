import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA

st.set_page_config(page_title="Data Preprocessing & EDA Suite", layout="wide")

# Custom CSS for Modern Design (Dark sleek theme)
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Modify sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #0ea5e9;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #0284c7;
        color: #ffffff;
        border: 1px solid #38bdf8;
    }
    
    /* Style st.info and success boxes */
    div.stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Data Preprocessing & EDA Suite")

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    # Read the data and establish persistent state
    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != file.name:
        raw_df = pd.read_csv(file)
        
        # Ensure unique column names to prevent Plotly/Narwhals parsing errors
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
        
        # Ensure strong types to prevent PyArrow serialization issues
        raw_df.columns = raw_df.columns.astype(str)
        for c in raw_df.select_dtypes(include=['object']).columns:
            raw_df[c] = raw_df[c].astype(str)
            
        st.session_state.current_df = raw_df
        st.session_state.original_df = raw_df.copy()
        st.session_state.uploaded_file = file.name

    # Load from persistent state and ALWAYS deduplicate columns
    df = st.session_state.current_df
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = df.columns.astype(str)
    st.session_state.current_df = df
    original_df = st.session_state.original_df
    original_df = original_df.loc[:, ~original_df.columns.duplicated()]

    # =========================
    # SIDEBAR
    # =========================
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("🔄 Reset Data to Original"):
        st.session_state.current_df = st.session_state.original_df.copy()
        st.rerun()
        
    st.sidebar.markdown("---")
    
    step = st.sidebar.radio("Select Suite Step", [
        "Data Overview",
        "Interactive Filtering",
        "EDA Visualization",
        "Data Cleaning",
        "Transformation",
        "PCA"
    ])

    # =========================
    # DATA OVERVIEW
    # =========================
    if step == "Data Overview":
        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Statistics")
        st.write(df.describe())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Data Types")
        st.write(df.dtypes)

    # =========================
    # INTERACTIVE FILTERING
    # =========================
    elif step == "Interactive Filtering":
        st.subheader("Interactive Dataset Filtering")
        filtered_df = df.copy()
        
        col1, col2 = st.columns(2)
        
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        with col1:
            st.markdown("#### Numeric Filters")
            for col in num_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                with st.expander(f"Filter: {col}"):
                    if min_val == max_val:
                        st.info(f"All values in '{col}' are {min_val} — no range to filter.")
                    else:
                        range_val = st.slider(f"Select Range", min_val, max_val, (min_val, max_val), key=f'slider_{col}')
                        filtered_df = filtered_df[(filtered_df[col] >= range_val[0]) & (filtered_df[col] <= range_val[1])]
                    
        with col2:
            st.markdown("#### Categorical Filters")
            for col in cat_cols:
                unique_vals = df[col].unique().tolist()
                with st.expander(f"Filter: {col}"):
                    selected_vals = st.multiselect(f"Select Values", unique_vals, default=unique_vals, key=f'multi_{col}')
                    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
                    
        st.metric(label="Rows Remaining", value=len(filtered_df), delta=f"Initial: {len(df)}", delta_color="off")
        st.dataframe(filtered_df)
        
        if st.button("Save Filtered Data"):
            st.session_state.current_df = filtered_df
            st.rerun()

    # =========================
    # EDA VISUALIZATION
    # =========================
    elif step == "EDA Visualization":
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Plotly toolbar config: only show Download and Autoscale
        chart_cfg = {"modeBarButtonsToKeep": ["toImage", "autoScale2d"], "displaylogo": False}
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribution Plot")
            dist_col = st.selectbox("Select Numeric Column", num_cols, key="dist")
            if st.button("Show Distribution"):
                fig = px.histogram(df, x=dist_col, marginal="box", template="plotly_dark", color_discrete_sequence=['#38bdf8'])
                st.plotly_chart(fig, use_container_width=True, config=chart_cfg)

        with c2:
            st.subheader("Scatter Plot")
            scat_x = st.selectbox("X-axis", num_cols, key="scat_x")
            scat_y = st.selectbox("Y-axis", num_cols, key="scat_y")
            if st.button("Show Scatter"):
                if scat_x == scat_y:
                    st.warning("Please select two different columns for the scatter plot.")
                else:
                    pearson_corr = df[[scat_x, scat_y]].corr().iloc[0, 1]
                    st.metric(f"Pearson Correlation ({scat_x} vs {scat_y})", f"{pearson_corr:.3f}")
                    scatter_df = df[[scat_x, scat_y]].copy()
                    scatter_df.columns = [scat_x, scat_y]
                    fig = px.scatter(scatter_df, x=scat_x, y=scat_y, 
                                     template="plotly_dark", 
                                     color_discrete_sequence=['#a855f7'])
                    st.plotly_chart(fig, use_container_width=True, config=chart_cfg)
                
        st.markdown("---")
        
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Correlation Heatmap")
            if st.button("Show Heatmap"):
                numeric_df = df.select_dtypes(include=np.number).dropna()
                fig = px.imshow(numeric_df.corr(), text_auto=".2f", aspect="auto", template="plotly_dark", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True, config=chart_cfg)
                
        with c4:
            st.subheader("Categorical Value Counts")
            if len(cat_cols) > 0:
                cat_col = st.selectbox("Categorical Column", cat_cols)
                if st.button("Show Counts"):
                    counts = df[cat_col].value_counts().reset_index()
                    counts.columns = ['Value', 'Count']
                    fig = px.bar(counts, x='Value', y='Count', template="plotly_dark", color='Count', color_continuous_scale="Teal")
                    st.plotly_chart(fig, use_container_width=True, config=chart_cfg)


    # =========================
    # DATA CLEANING
    # =========================
    elif step == "Data Cleaning":
        st.subheader("Missing Values")
        method = st.selectbox("Method", ["Drop", "Mean", "Median", "Mode"])

        if st.button("Apply Missing Handling"):
            if method == "Drop":
                df = df.dropna()
            elif method == "Mean":
                df = df.fillna(df.mean(numeric_only=True))
            elif method == "Median":
                df = df.fillna(df.median(numeric_only=True))
            elif method == "Mode":
                df = df.fillna(df.mode().iloc[0])
            st.session_state.current_df = df
            st.rerun()

        st.markdown("---")
        st.subheader("Remove Columns")
        cols_to_remove = st.multiselect("Select Columns to Remove", df.columns.tolist(), key="remove_cols")
        if cols_to_remove:
            if st.button("Drop Selected Columns"):
                df = df.drop(columns=cols_to_remove)
                st.session_state.current_df = df
                st.rerun()

        st.markdown("---")
        st.subheader("Outlier Detection & Removal (IQR Method)")
        
        num_df = df.select_dtypes(include=np.number)
        
        if not num_df.empty:
            st.markdown("#### Outlier Analysis")
            outlier_results = []
            for col in num_df.columns:
                Q1 = num_df[col].quantile(0.25)
                Q3 = num_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = num_df[(num_df[col] < lower_bound) | (num_df[col] > upper_bound)]
                num_outliers = len(outliers)
                outlier_results.append({
                    'Column': col, 
                    'Number of Outliers': num_outliers, 
                    'Percent Outliers (%)': round((num_outliers / len(df)) * 100, 2)
                })
                
            # Highlight the column with the highest number of outliers
            outliers_df = pd.DataFrame(outlier_results)
            st.dataframe(outliers_df.set_index('Column').style.highlight_max(axis=0, subset=['Number of Outliers'], color='#a855f7'))

            st.markdown("#### Visualize Outliers")
            col_outlier_viz = st.selectbox("Select Column to Plot:", num_df.columns)
            if col_outlier_viz:
                fig_outlier = px.box(df, y=col_outlier_viz, template="plotly_dark", color_discrete_sequence=['#f43f5e'], title=f"Outlier Plot: {col_outlier_viz}")
                st.plotly_chart(fig_outlier, use_container_width=True, config={"modeBarButtonsToKeep": ["toImage", "autoScale2d"], "displaylogo": False})

            st.markdown("#### Apply Outlier Handling")
            
            selected_outlier_cols = st.multiselect(
                "Select Columns to Apply Outlier Handling",
                num_df.columns.tolist(),
                default=num_df.columns.tolist(),
                key="outlier_cols"
            )
            
            if selected_outlier_cols:
                selected_num_df = num_df[selected_outlier_cols]
                Q1 = selected_num_df.quantile(0.25)
                Q3 = selected_num_df.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    st.markdown("**Remove Outliers**")
                    st.caption("Drops rows with outliers in selected columns.")
                    if st.button("Remove Outliers", key="remove_outliers"):
                        df = df[~((selected_num_df < lower_bound) | (selected_num_df > upper_bound)).any(axis=1)]
                        st.session_state.current_df = df
                        st.rerun()
                        
                with btn_col2:
                    st.markdown("**Cap Outliers (Winsorize)**")
                    st.caption("Clips extreme values in selected columns to IQR bounds.")
                    if st.button("Cap Outliers", key="cap_outliers"):
                        df_capped = df.copy()
                        for col in selected_outlier_cols:
                            df_capped[col] = df_capped[col].clip(lower=lower_bound[col], upper=upper_bound[col])
                        df = df_capped
                        st.session_state.current_df = df
                        st.rerun()
            else:
                st.warning("Please select at least one column.")
        else:
            st.info("No numerical columns found to analyze for outliers.")

        st.markdown("---")
        st.write("Current Dataset Preview:")
        st.write(df)


    # =========================
    # TRANSFORMATION
    # =========================
    elif step == "Transformation":
        st.subheader("Scaling")
        method = st.selectbox("Scaling Method", ["MinMax", "Standard"])

        if st.button("Apply Scaling"):
            num_cols = df.select_dtypes(include=np.number).columns
            scaler = MinMaxScaler() if method == "MinMax" else StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            st.session_state.current_df = df
            st.rerun()

        st.subheader("Encoding")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(cat_cols) > 0:
            enc = st.selectbox("Encoding Type", ["Label", "OneHot"])
            if st.button("Apply Encoding"):
                if enc == "Label":
                    for col in cat_cols:
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                else:
                    df = pd.get_dummies(df, columns=cat_cols)
                st.session_state.current_df = df
                st.rerun()

        st.write(df)


    # =========================
    # PCA
    # =========================
    elif step == "PCA":
        st.subheader("PCA")
        n = st.slider("Number of Components", 1, len(df.columns), 2)

        if st.button("Apply PCA"):
            df_clean = df.dropna()
            df_numeric = pd.get_dummies(df_clean)
            n_actual = min(n, len(df_numeric.columns))
            pca = PCA(n_components=n_actual)
            df = pd.DataFrame(pca.fit_transform(df_numeric), columns=[f"PC{i+1}" for i in range(n_actual)])
            st.session_state.current_df = df
            st.rerun()

        st.write(df)


    # =========================
    # BEFORE VS AFTER
    # =========================
    st.markdown("---")
    st.subheader("Data Evolution")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Dataset")
        st.write(original_df)

    with col2:
        st.write("Current Processed Dataset")
        st.write(df)

    # =========================
    # DOWNLOAD
    # =========================
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Current Pipeline State", csv, "processed.csv", "text/csv")
