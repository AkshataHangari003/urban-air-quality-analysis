import streamlit as st
import pandas as pd
import plotly.express as px
import os
import streamlit.components.v1 as components
import plotly.graph_objects as go
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="India AQI Analytics",
    layout="wide"
)

st.title("🌍 India Air Quality Analytics System")

# ---------------- DATA SOURCE ----------------
st.sidebar.subheader("📂 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ---------------- LOAD DEFAULT DATA ----------------
@st.cache_data
def load_default():
    path = "INDIA_AQI_COMPLETE_20251126.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df

    return None

# ---------------- LOAD DATA ----------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.sidebar.success("✅ Uploaded file loaded")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        st.stop()

else:

    df = load_default()

    if df is not None:
        st.sidebar.info("Using default dataset")

# ---------------- VALIDATION ----------------
if df is None or df.empty:
    st.error("❌ No data available")
    st.stop()

# ---------------- METRO CITY FILTER ----------------

allowed_cities = [

    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Chennai",
    "Kolkata",
    "Hyderabad"
]

city_like_col = None

for col in df.columns:

    if "city" in col.lower():

        city_like_col = col

        break

if city_like_col is not None and uploaded_file is None:

    df = df[

        df[city_like_col]
        .astype(str)
        .str.strip()
        .isin(allowed_cities)
    ]

# ---------------- COLUMN DETECTION ----------------
def detect_col(columns, keywords):

    for keyword in keywords:

        for col in columns:

            if keyword.lower() in col.lower():

                return col

    return None

city_col = detect_col(

    df.columns,

    [

        "city",
        "location",
        "metro"
    ]
)

aqi_col = detect_col(

    df.columns,

    [

        "aqi",
        "us_aqi",
        "air_quality"
    ]
)

date_col = detect_col(

    df.columns,

    [

        "date",
        "datetime",
        "time"
    ]
)

year_col = detect_col(

    df.columns,

    [

        "year"
    ]
)

# ---------------- FIX CITY COLUMN ----------------
if city_col is None:

    city_col = st.sidebar.selectbox(
        "Select City Column",
        df.columns
    )

# ---------------- FIX AQI COLUMN ----------------
if aqi_col is None:

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:

        aqi_col = st.sidebar.selectbox(
            "Select AQI Column",
            numeric_cols
        )

    else:
        st.error("❌ No numeric AQI column found")
        st.stop()

# ---------------- FIX YEAR COLUMN ----------------
if year_col is None and date_col is not None:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df["Year"] = df[date_col].dt.year

    year_col = "Year"

if year_col is None:

    df["Year"] = 2024
    year_col = "Year"

# ---------------- LOAD ML MODEL ----------------

model = None

if os.path.exists(
    "aqi_prediction_model.pkl"
):

    model = joblib.load(
        "aqi_prediction_model.pkl"
    )

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio(
    "Navigation",
    [
        "📊 AQI Analytics Dashboard",
        "📈 Interactive BI Report",
        "📘 Insights & Methodology"
    ]
)

# ====
# =====================================================

if page == "📊 AQI Analytics Dashboard":

    st.subheader("📊 Interactive Dashboard")

    # ---------------- FILTERS ----------------
    col1, col2 = st.columns(2)

    with col1:

        city = st.selectbox(
            "Select City",
            sorted(df[city_col].dropna().unique())
        )

    with col2:

        year = st.selectbox(
            "Select Year",
            sorted(df[year_col].dropna().unique())
        )


    city_df = df[
        df[city_col] == city
    ]

    filtered = city_df[
        city_df[year_col] == year
    ]

    # ---------------- FILTER DATA ----------------
    city_df = df[
    df[city_col] == city
    ]

    filtered = city_df[
        city_df[year_col] == year
    ]

    if filtered.empty:

        st.warning("No data available")

        st.stop()

    # ---------------- KPI ----------------
    avg_aqi = int(filtered[aqi_col].mean())

    st.metric(
        " Average AQI",
        avg_aqi
    )

    # ---------------- AQI GRAPH ----------------
    st.subheader("📈 AQI Trend Analysis")

    trend_df = city_df.copy()
    fig = px.area(
        trend_df,
        x=year_col,
        y=aqi_col,
        title=f"{city} AQI Trend"
    )

    fig.add_hline(
        y=50,
        line_dash="dash",
        annotation_text="Good",
        line_color="green"
    )

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="Satisfactory",
        line_color="blue"
    )

    fig.add_hline(
        y=200,
        line_dash="dash",
        annotation_text="Moderate",
        line_color="orange"
    )

    fig.add_hline(
        y=300,
        line_dash="dash",
        annotation_text="Poor",
        line_color="red"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- POLLUTANT ANALYSIS ----------------

    pollutant_keywords = [

        "pm",
        "no2",
        "so2",
        "co",
        "o3"
    ]

    pollutant_cols = [

        col for col in df.columns

        if any(

            p in col.lower()

            for p in pollutant_keywords
        )
    ]


    # keep only numeric columns
    pollutant_cols = [

        col for col in pollutant_cols

        if pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]


    if pollutant_cols:

        st.subheader(
            "🧪 Pollutant Breakdown"
        )

        values = filtered[
            pollutant_cols
        ].mean(
            numeric_only=True
        )

        fig2 = px.bar(

            x=pollutant_cols,

            y=values,

            color=values
        )

        st.plotly_chart(

            fig2,

            use_container_width=True
        )

    # ---------------- POLLUTION IMAGES ----------------

    st.subheader(" Major Sources of Air Pollution")

    img1 = os.path.join("images", "xyz.jpg")
    img2 = os.path.join("images", "opi1.jpg")

    c1, c2 = st.columns(2)

    with c1:

        if os.path.exists(img1):

            st.image(
                img1,
                caption=" Pollution",
                use_container_width=True
            )

    with c2:

        if os.path.exists(img2):

            st.image(
                img2,
                caption="🏭 Industrial Pollution",
                use_container_width=True
            )

    # ---------------- TABLE ----------------

    st.subheader("📋 Data Table")

    st.dataframe(filtered)

    # ---------------- WEATHER ANALYSIS ----------------

    weather_keywords = [

        "temp",
        "humid",
        "wind"
    ]

    weather_cols = [

        col for col in df.columns

        if any(

            w in col.lower()

            for w in weather_keywords
        )
    ]

    if len(weather_cols) >= 1:

        st.subheader(
            "🌦 Weather Impact Analysis"
        )

        fig_weather = px.scatter(

            filtered,

            x=weather_cols[0],

            y=aqi_col,

            color=aqi_col,

            title="Weather vs AQI"
        )

        st.plotly_chart(

            fig_weather,

            use_container_width=True
        )

    # ---------------- CORRELATION ----------------

    numeric_df = filtered.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] > 2:

        st.subheader(
            "🔥 Correlation Heatmap"
        )

        corr = numeric_df.corr()

        fig_corr = go.Figure(

            data=go.Heatmap(

                z=corr.values,

                x=corr.columns,

                y=corr.columns
            )
        )

        st.plotly_chart(

            fig_corr,

            use_container_width=True
        )

    # ---------------- AQI PREDICTION ----------------

    if model is not None:

        st.subheader(
            "🤖 AQI Prediction"
        )

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns.tolist()

        selected_features = [

            col for col in numeric_cols

            if col != aqi_col
        ][:len(model.feature_names_in_)]

        user_input = {}

        for col in selected_features:

            user_input[col] = st.number_input(

                col,

                value=float(
                    df[col].mean()
                )
            )

        if st.button(
            "Predict AQI"
        ):

            input_df = pd.DataFrame(
                [user_input]
            )

            prediction = model.predict(
                input_df
            )

            st.success(

                f"Predicted AQI: {round(prediction[0], 2)}"
            )

# =====================================================
# POWER BI
# =====================================================

elif page == "📈 Interactive BI Report":

    st.subheader("📊 Power BI Dashboard")

    html_file = "urban_aqi_powerbi_dashboard.html"

    if os.path.exists(html_file):

        with open(
            html_file,
            "r",
            encoding="utf-8"
        ) as f:

            html_data = f.read()

        components.html(
            html_data,
            height=900,
            scrolling=True
        )

    else:
        st.warning("⚠️ Power BI HTML not found")

# =====================================================
# EXPLANATION
# =====================================================

elif page == "📘 Insights & Methodology":

    st.subheader("📘 Insights & Methodology")

    st.markdown(f"""

### 📊 Dataset Summary

- Total Records: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Total Cities: {df[city_col].nunique()}

### 📌 Key Insights

- Average AQI: {round(df[aqi_col].mean(),2)}
- Highest AQI: {round(df[aqi_col].max(),2)}
- Lowest AQI: {round(df[aqi_col].min(),2)}

""")