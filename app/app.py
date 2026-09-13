import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CASEFILE Investigation Dashboard",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")


# =========================================================
# LOAD DATA
# =========================================================

def load_csv(filename, folder=PROCESSED_DIR):
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()


gps = load_csv("gps_anomalies.csv")
priority = load_csv("search_priority_scores.csv")
route = load_csv("predicted_route.csv")
xai = load_csv("xai_explanation.csv")
xai_factors = load_csv("xai_factor_contributions.csv")

cases = load_csv(
    "missing_person_cases.csv",
    SYNTHETIC_DIR
)


# =========================================================
# TITLE
# =========================================================

st.title("🔎 CASEFILE")
st.subheader(
    "AI-Powered Missing Person Investigation "
    "and Probable Location Prediction System"
)

st.warning(
    "Academic simulation only. Predictions are probabilistic "
    "and must not be used as proof of a person's location."
)


# =========================================================
# CHECK DATA
# =========================================================

if cases.empty:
    st.error(
        "missing_person_cases.csv was not found. "
        "Please check the data/synthetic folder."
    )
    st.stop()


# =========================================================
# CASE SELECTION
# =========================================================

st.sidebar.header("Case Selection")

case_ids = cases["Case_ID"].astype(str).tolist()

selected_case = st.sidebar.selectbox(
    "Select Case",
    case_ids
)

case = cases[
    cases["Case_ID"].astype(str) == selected_case
].iloc[0]


# =========================================================
# CASE INFORMATION
# =========================================================

st.header("📋 Case Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Case ID",
        str(case.get("Case_ID", "N/A"))
    )

with col2:
    st.metric(
        "Age Group",
        str(case.get("Age_Group", "N/A"))
    )

with col3:
    st.metric(
        "Last Seen Time",
        str(case.get("Last_Seen_Time", "N/A"))
    )

with col4:
    st.metric(
        "Day",
        str(case.get("Day", "N/A"))
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write(
        "**Weather:**",
        case.get("Weather", "N/A")
    )

with col2:
    st.write(
        "**Usual Area:**",
        case.get("Usual_Area", "N/A")
    )

with col3:
    st.write(
        "**Previous Area:**",
        case.get("Previous_Area", "N/A")
    )

with col4:
    st.write(
        "**Time Since Last Seen:**",
        case.get("Time_Since_Last_Seen", "N/A")
    )


# =========================================================
# LAST KNOWN LOCATION
# =========================================================

st.header("📍 Last Known Location")

lat = float(case["Last_Latitude"])
lon = float(case["Last_Longitude"])

st.write(
    f"Latitude: **{lat:.6f}**"
)

st.write(
    f"Longitude: **{lon:.6f}**"
)


# =========================================================
# SEARCH PRIORITY
# =========================================================

st.header("🎯 Search Priority Areas")

if not priority.empty:

    priority_display = priority.copy()

    if "Area" in priority_display.columns:

        priority_display = priority_display.sort_values(
            "Priority_Score",
            ascending=False
        )

        top5 = priority_display.head(5)

        st.dataframe(
            top5,
            use_container_width=True
        )

        if "Priority_Score" in top5.columns:

            chart_data = top5[
                ["Area", "Priority_Score"]
            ].set_index("Area")

            st.bar_chart(chart_data)


else:
    st.info(
        "Search priority data not available."
    )


# =========================================================
# TOP PROBABLE LOCATIONS
# =========================================================

st.header("📌 Top Probable Locations")

if not priority.empty:

    cols = [
        c for c in [
            "Area",
            "ML_Score",
            "Historical_Score",
            "Route_Score",
            "Distance_Score",
            "Time_Score",
            "Anomaly_Score",
            "Priority_Score",
            "Priority"
        ]
        if c in priority.columns
    ]

    st.dataframe(
        priority.sort_values(
            "Priority_Score",
            ascending=False
        ).head(5)[cols],
        use_container_width=True
    )

else:
    st.info(
        "Prediction data not available."
    )


# =========================================================
# MOVEMENT ANALYSIS
# =========================================================

st.header("🚶 Movement Analysis")

if not gps.empty:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "GPS Records",
            len(gps)
        )

    with col2:
        if "cluster" in gps.columns:
            clusters = gps[
                gps["cluster"] != -1
            ]["cluster"].nunique()

            st.metric(
                "Movement Areas",
                clusters
            )

    with col3:
        if "is_anomaly" in gps.columns:
            anomalies = int(
                gps["is_anomaly"].sum()
            )

            st.metric(
                "Anomalies",
                anomalies
            )

    with col4:
        if "speed" in gps.columns:
            st.metric(
                "Average Speed",
                f"{gps['speed'].mean():.2f}"
            )


# =========================================================
# ANOMALIES
# =========================================================

st.header("⚠️ Detected Movement Anomalies")

if not gps.empty and "is_anomaly" in gps.columns:

    anomaly_data = gps[
        gps["is_anomaly"] == 1
    ].copy()

    st.write(
        f"Detected anomaly records: **{len(anomaly_data)}**"
    )

    display_cols = [
        c for c in [
            "user_id",
            "timestamp",
            "latitude",
            "longitude",
            "speed",
            "cluster"
        ]
        if c in anomaly_data.columns
    ]

    st.dataframe(
        anomaly_data[display_cols].head(20),
        use_container_width=True
    )

else:
    st.info(
        "Anomaly information not available."
    )


# =========================================================
# ROUTE PREDICTION
# =========================================================

st.header("🛣️ Probable Route")

if not route.empty:

    st.dataframe(
        route,
        use_container_width=True
    )

    if "Area" in route.columns:

        route_list = route["Area"].astype(str).tolist()

        if len(route_list) > 0:

            route_text = " → ".join(
                route_list
            )

            st.success(
                f"Probable Route: {route_text}"
            )

else:

    st.info(
        "Route prediction data not available."
    )


# =========================================================
# EXPLAINABLE AI
# =========================================================

st.header("🧠 Prediction Explanation")

if not xai.empty:

    st.dataframe(
        xai,
        use_container_width=True
    )

else:

    st.info(
        "XAI explanation data not available."
    )


# =========================================================
# FACTOR CONTRIBUTIONS
# =========================================================

if not xai_factors.empty:

    st.subheader(
        "Prediction Factor Contributions"
    )

    st.dataframe(
        xai_factors,
        use_container_width=True
    )


# =========================================================
# INTERACTIVE MAP
# =========================================================

st.header("🗺️ Interactive Investigation Map")

m = folium.Map(
    location=[lat, lon],
    zoom_start=12
)


# =========================================================
# LAST KNOWN LOCATION
# =========================================================

folium.Marker(
    [lat, lon],
    popup="Last Known Location",
    tooltip="Last Known Location"
).add_to(m)


# =========================================================
# FREQUENTLY VISITED AREAS
# =========================================================

if not gps.empty and "cluster" in gps.columns:

    normal_gps = gps[
        gps["cluster"] != -1
    ].copy()

    if not normal_gps.empty:

        centers = (
            normal_gps
            .groupby("cluster")
            [["latitude", "longitude"]]
            .mean()
            .reset_index()
        )

        for _, row in centers.iterrows():

            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"]
                ],
                radius=6,
                popup=f"Movement Area {row['cluster']}",
                tooltip=f"Area {row['cluster']}"
            ).add_to(m)


# =========================================================
# SEARCH PRIORITY AREAS
# =========================================================

if not priority.empty and not gps.empty:

    if "Area" in priority.columns:

        normal_gps = gps[
            gps["cluster"] != -1
        ].copy()

        centers = (
            normal_gps
            .groupby("cluster")
            [["latitude", "longitude"]]
            .mean()
            .reset_index()
        )

        centers["Area"] = centers[
            "cluster"
        ].astype(str)

        top_priority = priority.sort_values(
            "Priority_Score",
            ascending=False
        ).head(5).copy()

        top_priority["Area"] = (
            top_priority["Area"]
            .astype(str)
        )

        map_data = top_priority.merge(
            centers,
            on="Area",
            how="left"
        )

        for _, row in map_data.iterrows():

            if pd.notna(row.get("latitude")):

                folium.Marker(
                    [
                        row["latitude"],
                        row["longitude"]
                    ],
                    popup=(
                        f"Area: {row['Area']}<br>"
                        f"Priority Score: "
                        f"{row['Priority_Score']:.2f}"
                    ),
                    tooltip=(
                        f"Priority Area: {row['Area']}"
                    )
                ).add_to(m)


# =========================================================
# ANOMALOUS LOCATIONS
# =========================================================

if not gps.empty and "is_anomaly" in gps.columns:

    anomaly_points = gps[
        gps["is_anomaly"] == 1
    ].head(200)

    for _, row in anomaly_points.iterrows():

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=3,
            popup="Anomalous Movement"
        ).add_to(m)


# =========================================================
# DISPLAY MAP
# =========================================================

st_folium(
    m,
    width=1200,
    height=600
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CASEFILE — Academic Advanced Machine Learning Project"
)

st.caption(
    "This system provides probabilistic investigation-support "
    "results based on historical and synthetic data."
)