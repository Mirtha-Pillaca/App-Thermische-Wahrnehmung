# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import altair as alt
import pydeck as pdk
import math
from app_projekt import df

# ============================================================
# Seitenkonfigurationen
# ============================================================
st.set_page_config(layout="wide")

# ============================================================
# Funktionen definieren
# ============================================================
# --- Laden der Daten ---

df_original = df.copy()
df["operative_temperature"] = pd.to_numeric(df["operative_temperature"], errors="coerce")
df["thermal_sensation"] = pd.to_numeric(df["thermal_sensation"], errors="coerce")


# --- Mehrfarbige Kreise für Klimazonen und Klimatypen erstellen ---
def create_pie_segments(df, climate_column, radius=1.5):
    # Leere Liste für die Segmente
    segments = []
    # Über jede Zeile des Dataframes iterieren
    for _, row in df.iterrows():
        climate_zones = row[climate_column]
        # Anzahl der Klimazonen bestimmen
        n = len(climate_zones)
        # Winkel pro Segment berechnen
        angle_step = 360 / n

        # Über jede Klimazone iterieren
        for i, climate_zone in enumerate(climate_zones):
            # Anfangswinkel berechnen
            start_angle = i * angle_step
            # Endwinkel berechnen
            end_angle = (i + 1) * angle_step
            # Polygon beginnen
            polygon = [[row["longitude"], row["latitude"]]]

            # Randpunkte erzeugen
            for angle in range(int(start_angle), int(end_angle) + 1, 5):
                # Radius festlegen
                lat_radius = radius
                # Radius für die Länge korrigieren
                lon_radius = radius / math.cos(math.radians(row["latitude"]))
                # Neue Koordinaten berechnen
                lon = (row["longitude"] + lon_radius * math.cos(math.radians(angle)))
                lat = (row["latitude"] + lat_radius * math.sin(math.radians(angle)))
                # Punkte an Polygon anhängen
                polygon.append([lon, lat])

            # Polygon schließen
            polygon.append([row["longitude"], row["latitude"]])

            # Segment speichern
            segments.append({
                "country": row["country"],
                climate_column: climate_zone,
                "polygon": polygon,
                "color": color_mapping.get(climate_zone)
            })

    return pd.DataFrame(segments)

# ============================================================
# SESSION STATE FILTERS (for other tabs)
# ============================================================
for key in [
    "region_filter", "country_filter", "city_filter",
    "season_filter", "climate_filter", "building_filter",
    "cooling_filter", "gender_filter"
]:
    if key not in st.session_state:
        st.session_state[key] = "Alle"

# ============================================================
# Seitentitel
# ============================================================
st.title("Thermische Wahrnehmung – Interaktives Analyse‑Dashboard")

# ============================================================
# VARIABLE MAPPING
# ============================================================
# variables = {
#     "Lufttemperatur": "air_temperature",
#     "Operative Temperatur": "operative_temperature",
#     "Strahlungstemperatur": "radiant_temperature",
#     "Relative Luftfeuchtigkeit": "relative_humidity",
#     "Luftgeschwindigkeit": "air_speed",
#     "Außentemperatur": "outdoor_air_temperature",
#     "SET": "standard_effective_temperature",
#     "PMV": "predicted_mean_vote",
#     "PPD": "predicted_percentage_dissatisfied",
#     "Bekleidungsisolation": "clothing_ensemble_insulation",
#     "Metabolische Aktivität": "metabolic_rate",
#     "Thermisches Empfinden": "thermal_sensation",
#     "Thermischer Komfort": "thermal_comfort",
#     "Thermische Präferent": "thermal_preference",
#     "Thermische Akzeptanz": "thermal_acceptability"
# }

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("Filter & Achsenwahl")

regions = ["Alle"] + sorted(df["region"].dropna().unique())
region = st.sidebar.selectbox("Region", regions)

if region == "Alle":
    countries = ["Alle"] + sorted(df["country"].dropna().unique())
else:
    countries = ["Alle"] + sorted(df[df["region"] == region]["country"].dropna().unique())
country = st.sidebar.selectbox("Land", countries)

if country == "Alle":
    cities = ["Alle"] + sorted(df["city"].dropna().unique())
else:
    cities = ["Alle"] + sorted(
        df[(df["region"] == region) & (df["country"] == country)]["city"].dropna().unique()
    )
city = st.sidebar.selectbox("Stadt", cities)

climate = st.sidebar.selectbox("Klimatyp", ["Alle"] + sorted(df["climate"].dropna().unique()))
climate_zone = st.sidebar.selectbox("Klimazone", ["Alle"] + sorted(df["climate_zone"].dropna().unique()))
building_type = st.sidebar.selectbox("Gebäudetyp", ["Alle"] + sorted(df["building_type"].dropna().unique()))
cooling_type = st.sidebar.selectbox("Kühlungsart", ["Alle"] + sorted(df["cooling_type"].dropna().unique()))
season = st.sidebar.selectbox("Jahreszeit", ["Alle"] + sorted(df["season"].dropna().unique()))
gender = st.sidebar.selectbox("Gender", ["Alle"] + sorted(df["gender"].dropna().unique()))

clo_min = df["clothing_ensemble_insulation"].min()
clo_max = df["clothing_ensemble_insulation"].max()
clo = st.sidebar.slider("Bekleidung (clo)", float(clo_min), float(clo_max), (float(clo_min), float(clo_max)))

metabolic_min = df["metabolic_rate"].min()
metabolic_max = df["metabolic_rate"].max()
metabolic_rate = st.sidebar.slider("Metabolische Aktivität (met)", float(metabolic_min), float(metabolic_max), (float(metabolic_min), float(metabolic_max)))

# ============================================================
# APPLY FILTERS SAFELY (used by ALL tabs)
# ============================================================
df_filtered = df.copy()

if region != "Alle": df_filtered = df_filtered[df_filtered["region"] == region]
if country != "Alle": df_filtered = df_filtered[df_filtered["country"] == country]
if city != "Alle": df_filtered = df_filtered[df_filtered["city"] == city]
if climate != "Alle": df_filtered = df_filtered[df_filtered["climate"] == climate]
if building_type != "Alle": df_filtered = df_filtered[df_filtered["building_type"] == building_type]
if cooling_type != "Alle": df_filtered = df_filtered[df_filtered["cooling_type"] == cooling_type]
if season != "Alle": df_filtered = df_filtered[df_filtered["season"] == season]
if gender != "Alle": df_filtered = df_filtered[df_filtered["gender"] == gender]

if clo != (clo_min, clo_max):
    df_filtered = df_filtered[
        df_filtered["clothing_ensemble_insulation"].between(clo[0], clo[1])
    ]

if metabolic_rate != (metabolic_min, metabolic_max):
    df_filtered = df_filtered[
        df_filtered["metabolic_rate"].between(metabolic_rate[0], metabolic_rate[1])
    ]

##################################################################################################################
##################################################################################################################

total_rows = len(df_original)

total_building_types = df_original["building_type"].nunique() if "building_type" in df_original else None
total_countries = df_original["country"].nunique() if "country" in df_original else None
total_regions = df_original["region"].nunique() if "region" in df_original else None
total_cities = df_original["city"].nunique() if "city" in df_original else None

total_climates = df_original["climate"].nunique() if "climate" in df_original else None
total_climate_zones = df_original["climate_zone"].nunique() if "climate_zone" in df_original else None

if "season" in df_original.columns:
    total_seasons = df_original["season"].dropna().nunique()
else:
    total_seasons = None

missing_total = df_original.isna().sum().sum()
missing_percent = (missing_total / df_original.size) * 100

if "year" in df_original.columns:
    df_original["year"] = pd.to_numeric(df_original["year"], errors="coerce")
    min_year = int(df_original["year"].min())
    max_year = int(df_original["year"].max())
    year_range = f"{min_year} – {max_year}"
else:
    year_range = "–"

# Komfortbewertungen
comfort_vars = ["thermal_sensation", "thermal_preference", "thermal_acceptability", "thermal_comfort"]
available_comfort_vars = [c for c in comfort_vars if c in df_filtered.columns]

st.markdown("---")

# ============================================================
# KPI Cards – 5 pro Reihe
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)
# --- Spalte 1 ---
with col1:
    st.metric("📦 Einträge", f"{total_rows:,}")
    st.metric("⚠️ Fehlende Werte (%)", f"{missing_percent:.2f}%")

# --- Spalte 2 ---
with col2:
    st.metric("📍 Regionen", f"{total_regions:,}" if total_regions else "–")
    st.metric("🌡️ Klimatypen", f"{total_climates:,}" if total_climates else "–")

# --- Spalte 3 ---
with col3:
    st.metric("🌍 Länder", f"{total_countries:,}" if total_countries else "–")
    st.metric("🗺️ Klimazonen", f"{total_climate_zones:,}" if total_climate_zones else "–")

# --- Spalte 4 ---
with col4:
    st.metric("🏙️ Städte", f"{total_cities:,}" if total_cities else "–")
    st.metric("🍂 Jahreszeiten", f"{total_seasons:,}" if total_seasons else "–")

# --- Spalte 5 ---
with col5:
    st.metric("🗓️ Zeitraum", f"{year_range}" if year_range else "–")
    st.metric("🏢 Gebäude Typen", f"{total_building_types:,}" if total_building_types else "–")

st.markdown("---")


#########################################################################################################
#########################################################################################################


# ---------------------------------------------------------
# Hauptbereich: X/Y-Plot
# ---------------------------------------------------------
# --- Überschrift ---
st.subheader("📊 Verteilungen nach Kategorie")

# ---------------------------------------------------------
# Spalten
# ---------------------------------------------------------
col1, spacer, col2 = st.columns([1, 0.2, 1])
# ---------------------------------------------------------
# Kategorien definieren für Filter Variable 1
# ---------------------------------------------------------
mapping_filter1 = {
    "Region": "region",
    "Land": "country",
    "Klimazone": "climate_zone",
    "Klimatyp": "climate",
    "Gebäudetyp": "building_type",
    "Kühlungsart": "cooling_type",
    "Jahreszeit": "season",
    "Gender": "gender"
}

# ---------------------------------------------------------
# Spalte 1: Filter Variable 1
# ---------------------------------------------------------
with col1:
    # --- Variable auswählen ---
    selected_variable1 = st.selectbox(
        "Variable auswählen",
        list(mapping_filter1.keys()),
        key="verteilung_variable1"
    )

column1 = mapping_filter1[selected_variable1]

# ---------------------------------------------------------
# Kategorien definieren für Filter Variable 2
# ---------------------------------------------------------
mapping_filter2 = {
    "Thermischer Komfort": "thermal_comfort",
    "Thermisches Empfinden": "thermal_sensation",
    "Thermische Präferenz": "thermal_preference",
    "Thermische Akzeptanz": "thermal_acceptability"
}

# ---------------------------------------------------------
# Spalte 2: Filter Variable 2
# ---------------------------------------------------------
with col2:
    # --- Variable auswählen ---
    selected_variable2 = st.selectbox(
        "Thermische Variable auswählen",
        list(mapping_filter2.keys()),
        key="verteilung_variable2"
    )

column2 = mapping_filter2[selected_variable2]


#########################################################################################################
#########################################################################################################

# ---------------------------------------------------------
# Hauptbereich: Grafiken
# ---------------------------------------------------------

# ---------------------------------------------------------
# Spalten
# ---------------------------------------------------------
col_plot1, spacer, col_plot2 = st.columns([2, 0.2, 2])
col_map, col3 = st.columns([2, 0.2])

# ---------------------------------------------------------
# Spalte col_plot1 
# ---------------------------------------------------------
with col_plot1:  
    # ---------------------------------------------------------
    # 3. Berechnungen basierend auf Sidebar-gefilterten Daten
    # ---------------------------------------------------------
    counts = df_filtered[column1].value_counts()
    percent = counts / counts.sum() * 100

    selection_df = pd.DataFrame({
        selected_variable1: counts.index,
        "Anzahl": counts.values,
        "Prozent": percent.round(2).astype(str) + " %"
    })
    selection_df = selection_df.sort_values("Anzahl", ascending=True)

    
    # ---------------------------------------------------------
    # 4. Grafik mit Anzahl-Labels
    # ---------------------------------------------------------
    # --- Titel ---
    st.subheader(f"Anzahl Einträge je {selected_variable1}")

    # --- Grafik ---
    chart = (
        alt.Chart(selection_df)
        .mark_bar(color="#4C72B0")
        .encode(
            x=alt.X("Anzahl:Q", title="Anzahl Einträge"),
            y=alt.Y(
                f"{selected_variable1}:N",
                sort=alt.EncodingSortField(
                    field="Anzahl",
                    op="sum",
                    order="descending"
                ),
                title=selected_variable1
            ),
            tooltip=[selected_variable1, "Anzahl", "Prozent"]
        )
        .properties(height=450)
    )

    # --- Anzahl-Labels über den Balken ---
    labels = (
        alt.Chart(selection_df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=5, 
            color="black",
            fontSize=12
        )
        .encode(
            x="Anzahl:Q",
            y=alt.Y(
                f"{selected_variable1}:N",
                sort=alt.EncodingSortField(
                    field="Anzahl",
                    op="sum",
                    order="descending"
                )
            ),
            text="Anzahl:Q"
        )
    )

    st.altair_chart(chart + labels, width="stretch")

# ---------------------------------------------------------
# Spalte col_plot2 
# ---------------------------------------------------------
with col_plot2:
    # ---------------------------------------------------------
    # 3. Berechnungen basierend auf Sidebar-gefilterten Daten
    # ---------------------------------------------------------
    counts = df_filtered[column2].value_counts()
    percent = counts / counts.sum() * 100

    selection_df = pd.DataFrame({
        selected_variable2: counts.index,
        "Anzahl": counts.values,
        "Prozent": percent.round(2).astype(str) + " %"
    })
    selection_df = selection_df.sort_values("Anzahl", ascending=True)

    # ---------------------------------------------------------
    # 4. Grafik mit Anzahl-Labels
    # ---------------------------------------------------------
    # ---  Titel ---
    st.subheader(f"Anzahl Einträge {selected_variable2}")

    # --- Grafik ---
    chart = (
        alt.Chart(selection_df)
        .mark_bar(color="#4C72B0")
        .encode(
            x=alt.X("Anzahl:Q", title="Anzahl Einträge"),
            y=alt.Y(f"{selected_variable2}:N", sort="-x", title=selected_variable2),
            tooltip=[selected_variable2, "Anzahl", "Prozent"]
        )
        .properties(height=450)
    )

    # --- Anzahl-Labels über den Balken ---
    labels = (
        alt.Chart(selection_df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=5, 
            color="black",
            fontSize=12
        )
        .encode(
            x="Anzahl:Q",
            y=f"{selected_variable2}:N",
            text="Anzahl:Q"
        )
    )

    st.altair_chart(chart + labels, width="stretch")

# ---------------------------------------------------------
# Spalte col_map 
# ---------------------------------------------------------
with col_map:
    # --- Titel ---
    st.subheader("🗺️ Geografische Verteilung der Messdaten")

    # ---------------------------------------------------------
    # Grafik für Variablen ohne Klimazone oder Klimatyp
    # ---------------------------------------------------------
    if selected_variable1 not in ["Klimazone", "Klimatyp"]:
        view_state = pdk.ViewState(
            latitude=df_filtered["latitude"].mean() if len(df_filtered) else 0,
            longitude=df_filtered["longitude"].mean() if len(df_filtered) else 0,
            zoom=2 if len(df_filtered) > 1 else 4
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position='[longitude, latitude]',
            get_fill_color='[0, 120, 255]',
            get_radius=100000,
            pickable=True
        )

        tooltip = {
            "html": """
                Region: {region}<br/>
                Country: {country}<br/>
                City: {city}<br/>
                Climatezone: {climate_zone}<br/>
                Building Type: {building_type}<br/>
                Cooling: {cooling_type}<br/>
                Season: {season}<br/>
                Records: {records}
            """,
            "style": {"color": "white"}
        }

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style=None
            )
        )

    # ---------------------------------------------------------
    # Grafik für Variablen Klimazone und Klimatyp 
    # ---------------------------------------------------------
    elif selected_variable1 in ["Klimazone", "Klimatyp"]:

        # ---------------------------------------------------------
        # Grafik für Variablen Klimazone und Klimatyp erstellen
        # ---------------------------------------------------------
        # --- Filter Klima/Klimazone anwenden ---
        if selected_variable1 == "Klimatyp":
            selected_climate_column = "climate"
        else:
            selected_climate_column = "climate_zone"

        # --- Filter-Auswahl Klimatyp ---
        # Land-Klimatyp-Kombinationen erstellen
        country_climate = (
            df_original[["country", "latitude", "longitude", selected_climate_column]]
            .groupby("country")
            .agg({
                "latitude": "mean",
                "longitude": "mean",
                selected_climate_column: lambda x: list(x.dropna().unique())
            })
            .reset_index()
        )
        # Klimanamen bereinigen
        country_climate[selected_climate_column] = (
            country_climate[selected_climate_column]
            .apply(
                lambda climates: [
                    c.strip().replace("\xa0", " ")
                    for c in climates
                    if isinstance(c, str)
                ]
            )
        )

        # --- Filter-Auswahl Klimazone ---
        # Land-Klimazonen-Kombinationen erstellen
        country_climate_zone = (
            df_original[["country", "latitude", "longitude", selected_climate_column]]
            .groupby("country")
            .agg({
                "latitude": "mean",
                "longitude": "mean",
                selected_climate_column: lambda x: list(x.dropna().unique())
            })
            .reset_index()
        )
        # Klimanamen bereinigen
        country_climate_zone[selected_climate_column] = (
            country_climate_zone[selected_climate_column]
            .apply(
                lambda climates: [
                    c.strip().replace("\xa0", " ")
                    for c in climates
                    if isinstance(c, str)
                ]
            )
        )

        # --- Farben festlegen ---
        # Farben für Klimatypen vergeben
        if selected_climate_column == "climate":
            color_mapping = {
            # Tropical
            "wet equatorial": [220, 80, 120, 180],
            "tropical rainforest": [200, 60, 120, 180],
            "tropical monsoon": [230, 100, 140, 180],
            "tropical savanna": [240, 130, 150, 180],
            "tropical wet savanna": [230, 110, 160, 180],
            "tropical dry savanna": [210, 90, 140, 180],
            "tropical": [220, 120, 160, 180],

            # Dry
            "hot arid": [245, 210, 80, 180],
            "desert (hot arid)": [240, 190, 60, 180],
            "hot desert": [230, 170, 40, 180],
            "semi arid midlatitude": [220, 180, 70, 180],
            "semi arid high altitude": [200, 170, 90, 180],
            "hot semi-arid": [235, 200, 90, 180],
            "cold semi-arid": [190, 170, 100, 180],
            "subtropical hot and dry": [250, 180, 50, 180],

            # Mediteranian
            "mediterranean": [180, 160, 70, 180],
            "hot-summer mediterranean": [200, 150, 60, 180],
            "warm-summer mediterranean": [170, 150, 80, 180],
            "cool-summer mediterranean": [140, 160, 100, 180],

            # Temperate
            "temperate": [80, 180, 90, 180],
            "humid subtropical": [60, 170, 100, 180],
            "temperature marine": [60, 150, 120, 180],
            "temperate oceanic": [40, 140, 170, 180],
            "west coast marine": [50, 130, 190, 180],
            "subtropical highland": [100, 190, 100, 180],

            # Continental
            "humid midlatitude": [120, 100, 200, 180],
            "warm-summer humid continental": [140, 100, 210, 180],
            "monsoon-influenced humid subtropical": [160, 120, 220, 180],
            "monsoon-influenced temperate oceanic": [130, 150, 220, 180],
            "monsoon-influenced hot-summer humid continental": [150, 90, 190, 180],

            # Subarctic
            "continental subarctic": [80, 90, 150, 180]
        }
            
        # Farben für Kliamzonen erstellen
        else:
            color_mapping = {
                "Tropical": [220, 120, 120, 180],
                "Dry": [245, 210, 80, 180],
                "Temperate": [0, 180, 0, 180],
                "Continental": [150, 0, 150, 180]
            }


        # ---------------------------------------------------------
        # Grafik für Variablen Klimazone und Klimatyp ausgeben
        # ---------------------------------------------------------
        # --- Grafik für Klimatypen ---
        if selected_climate_column == "climate":
            # Mehrfarbige Kreise für einzelne Klimata
            pie_data_climate = create_pie_segments(
                country_climate,
                selected_climate_column
            )

            layer_climate = pdk.Layer(
                "PolygonLayer",
                data=pie_data_climate,
                get_polygon="polygon",
                get_fill_color="color",
                pickable=True,
                stroked=False
            )

        # --- Grafik für Klimazonen ---
        else:
            # Mehrfarbige Kreise für Klimazonen
            pie_data_climate_zone = create_pie_segments(
                country_climate_zone,
                selected_climate_column
            )

            layer_climate_zone = pdk.Layer(
                "PolygonLayer",
                data=pie_data_climate_zone,
                get_polygon="polygon",
                get_fill_color="color",
                pickable=True,
                stroked=False
            )

        # ---------------------------------------------------------
        # Karte rendern (ohne Mapbox-Key!)
        # ---------------------------------------------------------
        # --- Spalten ---
        col1, spacer, col2 = st.columns([3, 0.2, 0.5])

        # --- Spalte 1: Karte ---
        with col1:
            # --- Kartenansicht definieren ---
            # Variable Klimatyp
            if selected_variable1 == "Klimatyp":
                view_state_climate = pdk.ViewState(
                    latitude=country_climate["latitude"].mean() if len(country_climate) else 0,
                    longitude=country_climate["longitude"].mean() if len(country_climate) else 0,
                    zoom=1
                )
                # Tooltip-Design
                tooltip_climate = {
                    "html": """
                    <b>{country}</b><br/>
                    Klimatyp: {climate}
                    """,
                    "style": {
                        "color": "white"
                    }
                }
            # Variable Klimazone
            else:
                view_state_climate_zone = pdk.ViewState(
                    latitude=country_climate_zone["latitude"].mean() if len(country_climate_zone) else 0,
                    longitude=country_climate_zone["longitude"].mean() if len(country_climate_zone) else 0,
                    zoom=1
                )
                # Tooltip-Design
                tooltip_climate_zone = {
                    "html": """
                    <b>{country}</b><br/>
                    Klimazone: {climate_zone}
                    """,
                    "style": {
                        "color": "white"
                    }
                }

            # --- Ausgabe der Karte ---
            # Variable Klimatyp
            if selected_variable1 == "Klimatyp":
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer_climate],
                        initial_view_state=view_state_climate,
                        tooltip=tooltip_climate,
                        map_style=None
                    )
                )
            # Variable Klimazone
            else:
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer_climate_zone],
                        initial_view_state=view_state_climate_zone,
                        tooltip=tooltip_climate_zone,
                        map_style=None 
                    )
                )

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- Spalte 2: Legende hinzufügen ---
        with col2:
            st.markdown("""
            **Klimazonen:**

            🔴 Tropical  
            🟡 Dry  
            🟢 Temperate  
            🟣 Continental
            """)

        # ---------------------------------------------------------
        # Zuordnung Klimata zu Klimazonen
        # ---------------------------------------------------------
        # --- Expander mit Zuordnungen und Weiteren Informationen---
        with st.expander("""ℹ️**Zuordnung von Klimatypen, Regionen und Ländern zu den Hauptklimazonen**"""):
            # Zuordnungen 
            for zone in sorted(df["climate_zone"].dropna().unique()):
                # --- Continental ---
                if zone == "Continental":
                    # Expander zu Continental
                    with st.expander(f"🟣 {zone}"):
                        zone_df = (
                            df[df["climate_zone"] == zone]
                            [["climate", "region", "country"]]
                            .drop_duplicates()
                            .sort_values(
                                by=["climate", "region", "country"]
                            )
                        )
                        # Dataframe
                        st.dataframe(
                            zone_df,
                            width="stretch",
                            hide_index=True
                        )

                # --- Dry ---
                elif zone == "Dry":
                    # Expander zu Dry
                    with st.expander(f"🟡 {zone}"):
                        zone_df = (
                            df[df["climate_zone"] == zone]
                            [["climate", "region", "country"]]
                            .drop_duplicates()
                            .sort_values(
                                by=["climate", "region", "country"]
                            )
                        )
                        # Dataframe
                        st.dataframe(
                            zone_df,
                            width="stretch",
                            hide_index=True
                        )

                # --- Temperate ---
                elif zone == "Temperate":
                    # Expander zu Temperate
                    with st.expander(f"🟢 {zone}"):
                        zone_df = (
                            df[df["climate_zone"] == zone]
                            [["climate", "region", "country"]]
                            .drop_duplicates()
                            .sort_values(
                                by=["climate", "region", "country"]
                            )
                        )
                        # Dataframe
                        st.dataframe(
                            zone_df,
                            uwidth="stretch",
                            hide_index=True
                        )

                # --- Tropical ---
                else:
                    # --- Expander zu Zuordnung ---
                    with st.expander(f"🔴 {zone}"):
                        zone_df = (
                            df[df["climate_zone"] == zone]
                            [["climate", "region", "country"]]
                            .drop_duplicates()
                            .sort_values(
                                by=["climate", "region", "country"]
                            )
                        )
                        # Dataframe
                        st.dataframe(
                            zone_df,
                            width="stretch",
                            hide_index=True
                        )

            # Hinweis zu Klimazonen-Zuweisung
            with st.expander("Weitere Informationen zu Klimatypen und Klimazonen"):
                st.markdown("""  
                - Hinweise:
                    - Die **5. Hauptklimazone Polar** ist hier nicht mit aufgeführt, da es für diese Klimazone in diesem Datensatz keine Daten gibt
                    - Es wurde **keine offizielle Zuordnung der Klimatypen zu den Klimazonen** gefunden, daher kann sich die hier gewählte Zuordnung von anderen unterscheiden
                """)

                st.markdown(""" 
                - **Beschreibungen zu Klimazonen:**
                    - **Tropical**: Ganzjährig hohe Temperaturen, geringe jahreszeitliche Schwankungen 
                    - **Dry**: Geringe Niederschläge, aride und semiaride Gebiete
                    - **Temperate**: Moderate Temperaturen, ausgeprägte Jahreszeiten
                    - **Continental**: Große Temperaturunterschiede zwischen Sommer und Winter
                """)
