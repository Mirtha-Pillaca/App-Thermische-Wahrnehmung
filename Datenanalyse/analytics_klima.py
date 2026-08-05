import streamlit as st
import pandas as pd
import pydeck as pdk
from streamlit_echarts import st_echarts
import seaborn as sns
import altair as alt
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import chi2_contingency
import plotly.express as px
import math


# ---------------------------------------------------------
# Seitenkonfigurationen
# --------------------------------------------------------- 
st.set_page_config(page_title="Analyse Klima", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------
# Funktionen definieren
# ---------------------------------------------------------
# Funktion für Interpretation der Effektgröße
def interpret_effect(v):
    if v < 0.1:
        return "sehr schwach"
    elif v < 0.3:
        return "schwach"
    elif v < 0.5:
        return "mittel"
    elif v < 0.7:
        return "stark"
    else:
        return "sehr stark"
    

# Funktion für Erstellen mehrfarbiger Kreise für Klimazonen und Klimatypen
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
                "color": climate_colors.get(climate_zone)
            })

    return pd.DataFrame(segments)

# --- Funktion für Balkendiagramm ---
def create_bars(y_domain):
    plot_df = df.copy()
    
    # --- Berechnungen für Diagramm und Ergebnistabelle ---
    thermal_stats = (
        plot_df
        .groupby(selected_environment_column)[selected_thermal_column]
        .agg(
            Mittelwert="mean",
            Median="median",
            Anzahl="count"
        )
        .reset_index()
    )

    thermal_stats["Mittelwert"] = thermal_stats["Mittelwert"].round(2)
    thermal_stats["Median"] = thermal_stats["Median"].round(2)

    thermal_stats = thermal_stats.sort_values(by="Mittelwert", ascending=False)

    # --- Grafik erstellen ---
    # Balken: Mittelwert
    bars = (
        alt.Chart(thermal_stats)
        .mark_bar(color="steelblue")
        .encode(
            x=alt.X(
                f"{selected_environment_column}:N",
                sort="-y",
                title=selected_variable_environment,
                axis=alt.Axis(labelAngle=-45)
            ),
            y=alt.Y(
                "Mittelwert:Q",
                title=f"Mittelwert {selected_variable_thermal}",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(tickMinStep=1)
            ),
            tooltip=[
                alt.Tooltip(
                    f"{selected_environment_column}:N",
                    title=selected_variable_environment
                ),
                alt.Tooltip(
                    "Mittelwert:Q",
                    format=".2f"
                ),
                alt.Tooltip(
                    "Median:Q",
                    format=".0f"
                ),
                alt.Tooltip(
                    "Anzahl:Q"
                )
            ]
        )
    )

    # Punkte: Median
    median_points = (
        alt.Chart(thermal_stats)
        .mark_point(
            color="red",
            filled=True,
            size=80
        )
        .encode(
            x=alt.X(
                f"{selected_environment_column}:N",
                sort="-y"
            ),
            y=alt.Y("Median:Q")
        )
    )

    # Diagramme mit Balken und Punkten kombinieren
    chart = (
        bars + median_points
    ).properties(
        height=500
    )

    st.altair_chart(
        chart,
        use_container_width=True
    )

    return thermal_stats

# ---------------------------------------------------------
# Daten laden
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Daten/db_bereinigt_final.csv")
df = load_data()

# ---------------------------------------------------------
# Seitentitel
# ---------------------------------------------------------
st.title("🌍 Analyse klimatische/geografische Variablen und thermische Wahrnehmung")

# ---------------------------------------------------------
# tabs definieren
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Geografische Verteilung",
    "🔍 Untersuchung der Unterschiede nach klimatischen/geografischen Variablen", 
    "📊 Betrachtung der Unterschiede",
    "📘 Zusammenfassung"
    ])

#########################################################################################################
#########################################################################################################

# ---------------------------------------------------------
# Tab 1: Klimatische Verteilung
# ---------------------------------------------------------
with tab1:
    # ---------------------------------------------------------
    # Vorbereitung Dataframe und Überschrift
    # ---------------------------------------------------------
    # --- Nur Zeilen behalten, die gültige Koordinaten haben ---
    df = df.dropna(subset=["latitude", "longitude"])
    # --- Überschrift ---
    st.subheader("Geografische Verteilung")

    # --- Text ---
    st.markdown("""
    - Die Klimatypen wurden 4 Hauptklimazonen zugeordnet 
    - Die Karte zeigt die geografische Verteilung der Hauptklimazonen bzw. der Klimatypen
    """
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Filter-Widget (Kima/Klimazone)
    # ---------------------------------------------------------
    climate_filter = st.selectbox(
        "Klimatische Variable auswählen",
        ["Klimazone", "Klimatyp"],
        key="climate_variable"
    )

    # ---------------------------------------------------------
    # Filter anwenden
    # ---------------------------------------------------------
    if climate_filter == "Klimatyp":
        selected_climate_column = "climate"
    else:
        selected_climate_column = "climate_zone"

    # ---------------------------------------------------------
    # Kombinationen von Ländern und Klimatypen erstellen
    # ---------------------------------------------------------
    # --- Land-Klimatyp-Kombinationen erstellen ---
    country_climate = (
        df[["country", "latitude", "longitude", selected_climate_column]]
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

    # ---------------------------------------------------------
    # Kombinationen von Ländern und Klimazonen erstellen
    # ---------------------------------------------------------
    # Land-Klimazonen-Kombinationen erstellen
    country_climate_zone = (
        df[["country", "latitude", "longitude", selected_climate_column]]
        .groupby("country")
        .agg({
            "latitude": "mean",
            "longitude": "mean",
            selected_climate_column: lambda x: list(x.dropna().unique())
        })
        .reset_index()
    )

    # Klimazonennamen bereinigen
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

    # ---------------------------------------------------------
    # Farben für Kreise vergeben
    # ---------------------------------------------------------
    # --- Farben für Klimatypen vergeben ---
    if selected_climate_column == "climate":
        climate_colors = {
        # Tropische Klimatypen
        "wet equatorial": [220, 80, 120, 180],
        "tropical rainforest": [200, 60, 120, 180],
        "tropical monsoon": [230, 100, 140, 180],
        "tropical savanna": [240, 130, 150, 180],
        "tropical wet savanna": [230, 110, 160, 180],
        "tropical dry savanna": [210, 90, 140, 180],
        "tropical": [220, 120, 160, 180],

        # Aride / trockene Klimatypen
        "hot arid": [245, 210, 80, 180],
        "desert (hot arid)": [240, 190, 60, 180],
        "hot desert": [230, 170, 40, 180],
        "semi arid midlatitude": [220, 180, 70, 180],
        "semi arid high altitude": [200, 170, 90, 180],
        "hot semi-arid": [235, 200, 90, 180],
        "cold semi-arid": [190, 170, 100, 180],
        "subtropical hot and dry": [250, 180, 50, 180],

        # Mediterrane Klimatypen
        "mediterranean": [180, 160, 70, 180],
        "hot-summer mediterranean": [200, 150, 60, 180],
        "warm-summer mediterranean": [170, 150, 80, 180],
        "cool-summer mediterranean": [140, 160, 100, 180],

        # Gemäßigte Klimatypen
        "temperate": [80, 180, 90, 180],
        "humid subtropical": [60, 170, 100, 180],
        "temperature marine": [60, 150, 120, 180],
        "temperate oceanic": [40, 140, 170, 180],
        "west coast marine": [50, 130, 190, 180],
        "subtropical highland": [100, 190, 100, 180],

        # Kontinentale Klimatypen
        "humid midlatitude": [120, 100, 200, 180],
        "warm-summer humid continental": [140, 100, 210, 180],
        "monsoon-influenced humid subtropical": [160, 120, 220, 180],
        "monsoon-influenced temperate oceanic": [130, 150, 220, 180],
        "monsoon-influenced hot-summer humid continental": [150, 90, 190, 180],

        # Subarktisches Klima
        "continental subarctic": [80, 90, 150, 180]
    }

    # --- Farben für Klimazonen vergeben ---
    else:
        climate_colors = {
            "Tropical": [220, 120, 120, 180],
            "Dry": [245, 210, 80, 180],
            "Temperate": [0, 180, 0, 180],
            "Continental": [150, 0, 150, 180]
        }

    # ---------------------------------------------------------
    # Grafik erstellen
    # ---------------------------------------------------------
    # --- Grafik für Klimatypen ---
    if selected_climate_column == "climate":
        # Mehrfarbige Kreise für einzelne Klimatypen
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
    # Kartenansicht definieren
    # ---------------------------------------------------------
    # --- Klimatypen ---
    if climate_filter == "Klimatyp":
        view_state_climate = pdk.ViewState(
            latitude=country_climate["latitude"].mean() if len(country_climate) else 0,
            longitude=country_climate["longitude"].mean() if len(country_climate) else 0,
            zoom=1
        )

        tooltip_climate = {
            "html": """
            <b>{country}</b><br/>
            Klimatyp: {climate}
            """,
            "style": {
                "color": "white"
            }
        }

    # --- Klimazonen ---
    else:
        view_state_climate_zone = pdk.ViewState(
            latitude=country_climate_zone["latitude"].mean() if len(country_climate_zone) else 0,
            longitude=country_climate_zone["longitude"].mean() if len(country_climate_zone) else 0,
            zoom=1
        )

        tooltip_climate_zone = {
            "html": """
            <b>{country}</b><br/>
            Klimazone: {climate_zone}
            """,
            "style": {
                "color": "white"
            }
        }

    col1, spacer, col2 = st.columns([2, 0.2, 0.5])

    # ---------------------------------------------------------
    # Spalte 1: Karte rendern
    # ---------------------------------------------------------
    with col1:
        # --- Klimatypen ---
        if climate_filter == "Klimatyp":
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer_climate],
                    initial_view_state=view_state_climate,
                    tooltip=tooltip_climate,
                    map_style=None
                )
            )

        # --- Klimazonen ---
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


    # ---------------------------------------------------------
    # Spalte 2: Legende hinzufügen
    # ---------------------------------------------------------
    with col2:
        st.markdown("""
        **Klimazonen:**

        🔴 Tropical  
        🟡 Dry  
        🟢 Temperate  
        🟣 Continental
        """)

    
    # ---------------------------------------------------------
    # Zuordnung Klimatypen zu Klimazonen
    # ---------------------------------------------------------
    # --- Überschrift ---
    st.markdown("### Zuordnung von Klimatypen, Regionen und Ländern zu den Hauptklimazonen")

    # --- Expander mit den Zuordnungen
    for zone in sorted(df["climate_zone"].dropna().unique()):
        # Klimazone Continental
        if zone == "Continental":
            # Expander mit Dataframe
            with st.expander(f"🟣 {zone}"):
                zone_df = (
                    df[df["climate_zone"] == zone]
                    [["climate", "region", "country"]]
                    .drop_duplicates()
                    .sort_values(
                        by=["climate", "region", "country"]
                    )
                )

                st.dataframe(
                    zone_df,
                    use_container_width=True,
                    hide_index=True
                )
        # Klimazone Dry
        elif zone == "Dry":
            # Expander mit Dataframe
            with st.expander(f"🟡 {zone}"):
                zone_df = (
                    df[df["climate_zone"] == zone]
                    [["climate", "region", "country"]]
                    .drop_duplicates()
                    .sort_values(
                        by=["climate", "region", "country"]
                    )
                )

                st.dataframe(
                    zone_df,
                    use_container_width=True,
                    hide_index=True
                )
        # Klimazone Temperate
        elif zone == "Temperate":
            # Expander mit Dataframe
            with st.expander(f"🟢 {zone}"):
                zone_df = (
                    df[df["climate_zone"] == zone]
                    [["climate", "region", "country"]]
                    .drop_duplicates()
                    .sort_values(
                        by=["climate", "region", "country"]
                    )
                )

                st.dataframe(
                    zone_df,
                    use_container_width=True,
                    hide_index=True
                )
        # Klimazone Tropical
        else:
            # Expander mit Dataframe
            with st.expander(f"🔴 {zone}"):

                zone_df = (
                    df[df["climate_zone"] == zone]
                    [["climate", "region", "country"]]
                    .drop_duplicates()
                    .sort_values(
                        by=["climate", "region", "country"]
                    )
                )

                st.dataframe(
                    zone_df,
                    use_container_width=True,
                    hide_index=True
                )

    # --- Expander mit Hinweis zu Klimazonen-Zuweisung ---
    with st.expander("ℹ️ Weitere Informationen zu Klimatypen und Klimazonen"):
        st.markdown("""  
        - Hinweise:
            - Die **5. Hauptklimazone Polar** ist hier nicht mit aufgeführt, da es für diese Klimazone in diesem Datensatz keine Daten gibt
            - Es wurde **keine offizielle Zuordnung der Klimatypen zu den Klimazonen** gefunden, daher kann sich die hier gewählte Zuordnung von anderen unterscheiden
        """)

        st.markdown(""" 
        - **Beschreibungen zu Klimazonen:**
            - **Tropical**: Ganzjährig hohe Temperaturen, geringe jahreszeitliche Schwankungen 
            - **Dry**: Geringe Niederschläge, kann heiß (Wüste) oder kalt (Kältewüste) sein
            - **Temperate**: Mäßige Temperaturen, ausgeprägte Jahreszeiten
            - **Continental**: Große Temperaturunterschiede zwischen Sommer und Winter
        """)



#########################################################################################################
#########################################################################################################

# ---------------------------------------------------------
# Tab 2: Untersuchung der Unterschiede zwischen klimatischen/geografischen Gruppen
# ---------------------------------------------------------
with tab2:
    # ---------------------------------------------------------
    # Überschrift
    # ---------------------------------------------------------
    st.subheader("📊 Gibt es Unterschiede in der thermischen Wahrnehmung nach Klimatyp, Klimazone, Region und Land?")
    st. markdown("""
     #### ❓ Fragestellungen der Analyse:


    - Gibt es Unterschiede zwischen den Ausprägungen der Gruppen innerhalb der klimatischen/geografischen Variablen (z.B. zwischen verschiedenen Klimatypen)?
    - Wie ausgeprägt sind diese Unterschiede je nach Variable (z.B. Sind die Unterschiede nach Klimatypen größer als nach Klimazonen)?
    """
    )
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Spalten definieren
    # ---------------------------------------------------------
    col_heatmap, col_results = st.columns([2,1.7])
    col_tests, col4 = st.columns([2, 1])

    # ---------------------------------------------------------
    # Mapping-Dictionaries
    # ---------------------------------------------------------
    # Mapping-Dictionary Klima
    environment_mapping = {
        "Klimatyp": "climate",
        "Klimazone": "climate_zone",
        "Region": "region",
        "Land": "country"      
    }
      

    # Mapping-Dictionary thermische Variablen
    thermal_mapping = {
        "Thermischer Komfort": "thermal_comfort",
        "Thermisches Empfinden": "thermal_sensation",
        "Thermische Präferenz": "thermal_preference",
        "Thermische Akzeptanz": "thermal_acceptability"
    }
    
    # ---------------------------------------------------------
    # Statistischen Zusammenhang berechnen
    # --------------------------------------------------------- 
    # --- Leere Liste erstellen ---
    results = []

    # --- Schleife für alle klimatischen Variablen ---
    for environment_label, environment_column in environment_mapping.items():

        # Schleife für alle thermischen Variablen 
        for thermal_label, thermal_column in thermal_mapping.items():

            # Unknown-Einträge entfernen in thermal_preference und thermal_acceptability
            if thermal_column in ["thermal_preference", "thermal_acceptability"]:
                df_test = df[df[thermal_column] != "Unknown"]
            else:
                df_test = df

            # Kreuztabelle erstellen
            contingency_table = pd.crosstab(
                df_test[environment_column],
                df_test[thermal_column]
            )

            # Chi2-Test durchführen
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            # Berechnung der Gesamtzahl aller Beobachtungen
            n = contingency_table.sum().sum()
            # Phi2 berechnen
            phi2 = chi2 / n
            # Tabellenform bestimmen
            r, k = contingency_table.shape
            # Effektgröße Cramers V berechnen
            cramers_v = np.sqrt(phi2 / min(k-1, r-1))

            # Ergebnisse speichern 
            results.append({
                "Klimatische/geografische Variable": environment_label,
                "Thermische Wahrnehmungsvariable": thermal_label,
                "p-Wert": "p < 0.001" if p < 0.001 else f"{p:.4f}",
                "Signifikant": "✅" if p < 0.05 else "✗",
                "Effektgröße": round(cramers_v, 3),
                "Interpretation des Zusammenhangs": interpret_effect(cramers_v)                 
            })

            # Ergebnisse als Dataframe ausgeben
            chi2_results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Erstellung und Ausgabe der Heatmap
    # --------------------------------------------------------- 
    # --- Spalte Heatmap ---
    with col_heatmap:
        # Dataframe für Heatmap erzeugen
        heatmap_df = chi2_results_df.pivot(
            index="Klimatische/geografische Variable",
            columns="Thermische Wahrnehmungsvariable",
            values="Effektgröße"
        )

        # Heatmap erstellen
        fig = px.imshow(
            heatmap_df,
            text_auto=".2f",
            color_continuous_scale="Blues",
            zmin=0,
            zmax=1,
            labels={
                "color": "Cramérs V"
            }
        )

        # Hover-Informationen deaktivieren
        fig.update_traces(
            hovertemplate=None,
            hoverinfo="skip"
        )

        # Größe der Grafik festlegen
        fig.update_layout(
            height=600
        )

        # Heatmap anzeigen lassen
        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Spalte Ergebnisse ---
    with col_results:
        st.markdown("""
        #### 📌 Wichtige Ergebnisse:

        ➡️ Bei allen klimatischen/geografischen Variablen **unterscheiden sich die Gruppen statistisch signifikant** hinsichtlich der thermischen Wahrnehmung ✅:
        """
            )

        st.markdown("""


        - **Stärkste Unterschiede** bei thermischen Wahrnehmungsvariablen: bei **thermischem Komfort und thermischer Akzeptanz**          
        - Unterschiede nach klimatischen/geografischen Variablen:
            - **Stärkste Unterschiede bei Klimatyp** ➝ mittlere bis schwach ausgeprägte Unterschiede
                        
            - **Geringste Unterschiede bei Klimazone und Region** ➝ schwach bis sehr schwach ausgeprägte Unterschiede 
                    
        ➡️ Unterschiede in thermischer Wahrnehmung zeigen sich deutlicher bei **feinerer klimatischer Klassifikation** als bei übergeordneten Klimazonen oder Länder-/Regionszugehörigkeit
    """
    )
        
    # ---------------------------------------------------------
    # Ergebnis-Dataframe für statistischen Zusammenhang ausgeben
    # --------------------------------------------------------- 
    # --- Spalte mit Details zu statistischen Tests ---
    with col_tests:
        # Überschrift
        st.subheader("ℹ️ Details zu statistischen Tests")

        # Für jede klimatische Variable einen Expander mit Dataframe ausgeben
        for variable in environment_mapping.keys():
            # Expander mit Detail-Dataframe
            with st.expander(f"**📈 {variable} ↔ Thermische Wahrnehmung**"):
                st.dataframe(
                    chi2_results_df[chi2_results_df["Klimatische/geografische Variable"] == variable],
                    hide_index=True,
                    use_container_width=True
                )
        st.markdown("<br>", unsafe_allow_html=True)

        # Expander mit Informationen zum Lesen der Unterschiede
        with st.expander("ℹ️ Informationen zum Lesen der Unterschiede"):
            st.markdown("""                  
            - **Erklärung der Werte:**
                - **p-Wert**: gibt an, ob ein Zusammenhang statistisch signifikant ist 
                    
                    ➝ wenn p < 0.05 ➝ signifikant ✅
                - **Effektgröße**: gibt die Größe des Zusammenhangs an ➝ Interpretation bei Cramérs V zur Orientierung:
                     - < 0.10 ➝ sehr schwach (geringe Unterschiede zwischen den Gruppen)
                     - < 0.30 ➝ schwach (leichte Unterschiede zwischen den Gruppen)
                     - < 0.50 ➝ mittel (deutliche Unterschiede zwischen den Gruppen)
                     - &gt; 0.50 ➝ stark (stark ausgeprägte Unterschiede zwischen den Gruppen)
            
            - **Hinweise:** 
                - Für die Signifikanzprüfung wurde der Chi²-Test verwendet, für die Ermittlung der Effektstärke wurde Cramérs V berechnet
                    
                
                    ➝ Thermischer Komfort und Thermisches Empfinden sind ordinal skaliert, weshalb auch zusätzliche Rangtests verwendet werden könnten
                    ➝ Für die vergleichende Darstellung wurde jedoch eine einheitliche kategoriale Betrachtung gewählt
                - Es kann nur eine Aussage darüber gemacht werden, ob ein Zusammenhang besteht, jedoch nicht in welche Richtung dieser wirkt
            """)             
        st.markdown("<br><br>", unsafe_allow_html=True)


#########################################################################################################
#########################################################################################################

# ---------------------------------------------------------
# Tab 3: Betrachtung der Unterschiede zwischen klimatischen Gruppen
# ---------------------------------------------------------
with tab3:
    # ---------------------------------------------------------
    # Überschrift
    # ---------------------------------------------------------
    st.subheader("Wie sehen die Unterschiede in der thermischen Wahrnehmung nach klimatischen/geografischen Variablen aus?")
    

    # ---------------------------------------------------------
    # Text
    # ---------------------------------------------------------
    st.markdown("""
        ℹ️ Die statistische Untersuchung hat gezeigt, dass es Unterschiede zwischen den Gruppen der klimatischen und geografischen Variablen hinsichtlich thermischer Wahrnehmung gibt
        
        **➝ Wie sehen diese Unterschiede aus?**
    """
    )
    st.markdown("<br>", unsafe_allow_html=True)


    # ---------------------------------------------------------
    # Spalten definieren
    # ---------------------------------------------------------
    col_filter, col2, col_thermals = st.columns([1,0.08, 1])
    col_chart, spacer = st.columns([2, 0.2])
    col_results, spacer = st.columns([2, 0.2])
    col8, col9 = st.columns([10, 0.2])
         
    # ---------------------------------------------------------
    # Spalte mit Filtern
    # ---------------------------------------------------------
    with col_filter:
        # ---------------------------------------------------------
        # Filter-Widget
        # ---------------------------------------------------------
        # Filter-Widget (Klima/Klimazone)
        selected_variable_environment = st.selectbox(
            "Klimatische/geografische Variable auswählen",
            list(environment_mapping.keys()),
            key="selectbox_environment"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Filter-Widget (thermische Bewertungsvariablen)
        selected_variable_thermal = st.selectbox(
            "Thermische Wahrnehmungsvariable auswählen",
            list(thermal_mapping.keys()),
            key="selectbox_thermal"
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Spalte mit Erklärung zu ausgewählter thermischer Wahrnehmungsvariable
    # ---------------------------------------------------------
    with col_thermals:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        # Ausgabe bei Thermischem Komfort
        if selected_variable_thermal == "Thermischer Komfort":
            st.markdown("""
                ##### Thermischer Komfort 
            **Sehr unkomfortabel ◄────────────────► Sehr komfortabel**  
                    `  1             2            3           4           5            6   `
            """)
            st.markdown("<br>", unsafe_allow_html=True)

        # Ausgabe bei Thermischem Empfinden
        elif selected_variable_thermal == "Thermisches Empfinden":
            st.markdown("""
            ##### Thermisches Empfinden
            **Kalt  ◄────── Neutral ──────►  Heiß**  
            `-3    -2    -1    0    +1    +2    +3 `
            """)
            st.markdown("<br>", unsafe_allow_html=True)

        # Ausgabe bei Thermischer Präferenz
        elif selected_variable_thermal == "Thermische Präferenz":
            st.markdown("""
            ##### Thermische Präferenz 
            **Kühler ◄──────── Keine Änderung ────────► Wärmer**  
            `  -1                         0                         +1     `
            """)
            st.markdown("<br><br>", unsafe_allow_html=True)

        # Ausgabe bei Thermischer Akzeptanz
        elif selected_variable_thermal == "Thermische Akzeptanz":
            st.markdown("""
            ##### Thermische Akzeptanz
            ○ Nicht akzeptabel  
            ○ Akzeptabel  
            """)
            st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Mapping anwenden
    # ---------------------------------------------------------
    # --- Mapping für Klima anwenden ---
    selected_environment_column = environment_mapping[selected_variable_environment] 
    # --- Mapping für thermische Wahrnehmungsvariable anwenden --- 
    selected_thermal_column = thermal_mapping[selected_variable_thermal]      

    # ---------------------------------------------------------
    # Grafiken erstellen
    # ---------------------------------------------------------
    # --- Diagramm ---
    with col_chart:

        # ---------------------------------------------------------
        # Thermischer Komfort
        # ---------------------------------------------------------

        # --- Diagramm Thermischer Komfort ---
        if selected_variable_thermal == "Thermischer Komfort":
            # --- Titel für Diagramm Thermischer Komfort ---
            st.subheader(f"Thermischer Komfort und {selected_variable_environment}")
            # --- Balkendiagramm ausgeben lassen ---
            thermal_stats = create_bars(y_domain=[0, 6])

            # --- Ergebnisse ---
            with col_results:
                # Ergebnisse für Klimatypen
                if selected_variable_environment == "Klimatyp":
                    st.markdown("""  
                    - Thermischer Komfort wird **überwiegend tendenziell positiv** bewertet
                    - Bewertung des thermischen Komforts **unterscheidet sich zwischen den Klimatypen stärker als zwischen den Hauptklimazonen**        

                        - **Subtropcial highland:** bewertet Komfort tendenziell am besten mit "sehr komfortabel"(Median = 6)
                        - **Monsoon-influenced hot-summer humid continental:** bewertet Komfort tendenziell am schlechtesten mit "leicht unkomfortabel" (Median = 2)
                    """
                    )
                elif selected_variable_environment == "Klimazone":
                    st.markdown("""  
                    - Thermischer Komfort wird **überwiegend positiv** bewertet
                    - **Unterschiede:**
                                                            
                        - **Dry, Temperate und Tropical:** bewerten thermischen Komfort tendenziell positiv mit "komfortabel" (Median = 5)
                        - **Continental:** bewertet thermischen Komfort tendenziell niedriger mit "neutral" (Median = 3)
                    """
                    )
                elif selected_variable_environment == "Region":
                    st.markdown("""
                        - Thermischer Komfort wird **in allen Regionen ähnlich** bewertet mit "leicht komfortabel" bis "komfortabel" (Medianwerte 4 oder 5)
                                
                            ➝ Mittelwerte unterscheiden sich nur gering 
                        - **Unterschiede:**
                            - **Europa:** leicht niedrigere Bewertung des thermischen Komforts mit "leicht komfortabel" (Median = 4) im Vergleich zu anderen Kontinenten (Median = 5)
                    """
                    )
                else:
                    st.markdown("""
                        - Bewertungen liegen **in allen Ländern bei einem mittleren bis höheren Komfort** (Medianwerte zwischen 3 und 5)
                                
                            ➝ insgesamt positive Komfortbewertung
                        - **Unterschiede:**
                            - **Cyprus, Singapore, Denmark, China:** niedrigste Bewertung des Komforts mit "neutral" (Median = 3)
                    """
                    )

            # Ergebnistabelle und Bedeutung der Ergebnisse
            with col_results:
                # Expander mit Details
                with st.expander(
                        f"**📈 Details zu Ergebnissen Thermischer Komfort und {selected_variable_environment}**"
                        ):
                        st.dataframe(thermal_stats, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)


        # ---------------------------------------------------------
        # Thermisches Empfinden
        # ---------------------------------------------------------
        elif selected_variable_thermal == "Thermisches Empfinden":
            # --- Titel für Diagramm Thermisches Empfinden ---
            st.subheader(f"Thermisches Empfinden und {selected_variable_environment}")
            # --- Balkendiagramm ausgeben lassen ---
            thermal_stats = create_bars(y_domain=[-3, 3])
            
            # --- Ergebnisse ---
            with col_results:
                # Ergebnisse für Klimatypen
                if selected_variable_environment == "Klimatyp":
                    st.markdown("""  
                    - Thermisches Empfinden wird **tendenziell** eher als **neutral** bewertet (meiste Medianwerte bei 0)


                        ➝ mit leichter Tendenz zu wärmerer Bewertung (meiste Mittelwerte zwischen -0.2 und + 0.6) 
                    - Aber es gibt **mehr Variation** als bei den Hauptklimazonen (Medianwerte zwischen -1 und 1) 
                    """
                    )
                elif selected_variable_environment == "Klimazone":
                    st.markdown("""  
                    - In allen vier Klimazonen wird das thermische Empfinden **tendenziell** als **neutral** bewertet  (Median = 0)
                    - Mittelwerte weisen auf eine geringe Tendenz zu einer wärmeren Wahrnehmung hin (Mittelwerte zwischen 0.07 und 0.24)
                    """
                    )
                elif selected_variable_environment == "Region":
                    st.markdown("""
                        - Thermisches Empfinden wird **in allen Regionen** im Median als **neutral** bewertet (Median = 0)


                            ➝ leichte Tendenz zu wärmerer Wahrnehmung (positive Mittelwerte)
                        - **Africa**: stärkste Tendenz zu wärmerer Wahrnehmung (Mittelwert = 0.69)
                    """
                    )
                else:
                    st.markdown("""
                        - Thermisches Empfinden wird **in meisten Ländern** im Median als **neutral** bewertet
                        - Bewertungen zeigen aber **größere Variation** als bei Regionen (Medianwerte zwischen -1 und 2, Mittelwerte zwischen -1.04 und +2.14)
                        - **Unterschiede:** 
                            - **Nigeria:** stärkere Tendenz zu wärmerer Wahrnehmung (Median = 2)
                            - **Cyprus und Philippines:** kühlere Wahrnehmung (Median = -1)
                    """
                    )
                st.markdown("<br><br>", unsafe_allow_html=True)

            # Ergebnistabelle und Bedeutung der Ergebnisse
            with col_results:
                # Expander mit Details
                with st.expander(f"**📈 Ergebnisse Thermisches Empfinden und {selected_variable_environment}**"):
                    st.dataframe(thermal_stats, use_container_width=True)
                

        # ---------------------------------------------------------
        # Thermische Präferenz
        # ---------------------------------------------------------
        elif selected_variable_thermal == "Thermische Präferenz":
            
            # --- Titel für Diagramm Thermische Präferenz ---
            st.subheader(f"Thermische Präferenz und {selected_variable_environment}")
            plot_df = df.copy()

            # --- Unknown entfernen ---
            plot_df = plot_df[plot_df["thermal_preference"] != "Unknown"]

            # --- Dataframe nur mit gültigen Antworten ---
            valid_df = df[
                df["thermal_preference"].isin(
                    ["cooler", "no change", "warmer"]
                )
            ]

            # --- Berechnungen für Diagramm und Ergebnistabelle ---
            preference_pct = (
                pd.crosstab(
                    valid_df[selected_environment_column],
                    valid_df["thermal_preference"],
                    normalize="index"
                ) * 100
            ).reset_index()

            # --- Nach Anteil der no change-Werte absteigend sortieren ---
            preference_pct = (
                preference_pct
                .sort_values(
                    by="no change",
                    ascending=False
                )
            )

            order = preference_pct[selected_environment_column].tolist()

            # --- Diagramm vorbereiten ---
            preference_long = preference_pct.melt(
                id_vars=[selected_environment_column],
                var_name="Präferenz",
                value_name="Prozent"
            )

            # --- Grafik ---
            chart = (
                alt.Chart(preference_long)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{selected_environment_column}:N",
                        title=selected_variable_environment,
                        axis=alt.Axis(labelAngle=-45),
                        sort=order,
                    ),
                    y=alt.Y(
                        "Prozent:Q",
                        title="Anteil (%)",
                        scale=alt.Scale(domain=[0, 100])
                    ),
                    color=alt.Color(
                        "Präferenz:N",
                        title="Thermal Preference"
                    ),
                    tooltip=[
                        selected_environment_column,
                        "Präferenz",
                        alt.Tooltip("Prozent:Q", format=".1f")
                    ]
                )
                .properties(
                    height=500
                )
            )

            # --- Grafik anzeigen ---
            st.altair_chart(chart, use_container_width=True)

            # --- Dataframe mit Unknown erstellen ---
            unknown_pct = (
                pd.crosstab(
                    df[selected_environment_column],
                    df["thermal_preference"],
                    normalize="index"
                ) * 100
            ).reset_index()

            unknown_pct = unknown_pct[
                [
                    selected_environment_column,
                    "Unknown"
                ]
            ]

            result_df = preference_pct.merge(
                unknown_pct,
                on=selected_environment_column,
                how="left"
            )

            # --- Spaltenreihenfolge ändern (Unknown nach hinten) ---
            cols = [
                selected_environment_column,
                "cooler",
                "no change",
                "warmer",
                "Unknown"
            ]

            result_df = result_df.reindex(
                columns=cols,
                fill_value=0
            )

            # --- Ergebnisse ---
            with col_results:
                # Ergebnisse für Klimatypen
                if selected_variable_environment == "Klimatyp":
                    st.markdown("""  
                    - Bei thermischer Präferenz **überwiegt tendenziell der Wunsch nach keiner Änderung**
                    - **Unterschiede:**
                        - **Temperate**: höchster Anteil an Bewertungen mit "keine Änderung" bei gültigen Antworten (75,9%)  
                        - Anteil Bewertungen mit "wärmer"/"kühler" bei gültigen Antworten: 
                            - **Semi arid high altitude:** höchster Anteil an Bewertungen mit "wärmer" bei gültigen Antworten (45,5%)
                            - **Tropical rainforest:** höchster Anteil an Bewertungen mit "kühler" bei gültigen Antworten (55,6%)
                    - Ergebnisse sollten unter Berücksichtigung der teilweise hohen Anteile an Unknown-Antworten interpretiert werden
                    """
                    )
                elif selected_variable_environment == "Klimazone":
                    st.markdown("""  
                    - Bei thermischer Präferenz **überwiegt tendenziell der Wunsch nach keiner Änderung**
                    - Unterschiede: 
                        - **Continental**: höchster Anteil an Bewertungen mit "keine Änderung" bei gültigen Antworten (57,5%)              
                        - Anteil Bewertungen mit "wärmer"/"kühler" bei gültigen Antworten: 
                            - **Continental:** höchster Anteil an Bewertungen mit "wärmer" bei gültigen Antworten (23%)
                            - **Tropical:** höchster Anteil an Bewertungen mit "kühler" bei gültigen Antworten (41,7%)
                    - Ergebnisse sollten unter Berücksichtigung der teilweise hohen Anteile an Unknown-Antworten interpretiert werden
                    """
                    )
                elif selected_variable_environment == "Region":
                    st.markdown("""
                        - Bei thermischer Präferenz **überwiegt tendenziell der Wunsch nach keiner Änderung**
                        - **Africa:** größter Anteil an Präferenz von **kühleren Bedingungen** bei gültigen Antworten (93,9%)
                            
                            ➝ hat aber geringere Stichprobengröße
                        - Ergebnisse sollten unter Berücksichtigung der teilweise hohen Anteile an Unknown-Antworten interpretiert werden
                    """
                    )
                else:
                    st.markdown("""
                        - Bei thermischer Präferenz **überwiegt tendenziell der Wunsch nach keiner Änderung**
                        - **Unterschiede:**
                            - **Greece, Thailand, Mexiko, Denmark:** höherer Anteil an Präferenz für kühlere Bedingungen bei gültigen Antworten
                            - **Nigeria:** höchster Anteil an Präferenz für kühlere Bedingungen bei gültigen Antworten (93,9%)
                            ➝ aber teilweise geringere Stichprobengröße
                        - Ergebnisse sollten unter Berücksichtigung der teilweise hohen Anteile an Unknown-Antworten interpretiert werden
                    """
                )

            # Ergebnistabelle und Bedeutung der Ergebnisse
            with col_results:
                # Expander mit Details
                with st.expander(
                        f"**📈 Details zu Ergebnissen Thermische Präferenz und {selected_variable_environment}**"
                        ):
                        st.dataframe(result_df, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            

        # ---------------------------------------------------------
        # Thermische Akzeptanz
        # ---------------------------------------------------------
        elif selected_variable_thermal == "Thermische Akzeptanz":

            # --- Titel für Diagramm Thermische Akzeptanz ---
            st.subheader(f"Thermische Akzeptanz und {selected_variable_environment}")

            # --- Dataframe nur mit gültigen Antworten ---
            valid_df = df[
                df["thermal_acceptability"].isin(
                    ["acceptable", "unacceptable"]
                )
            ]

            # --- Berechnungen für Diagramm und Ergebnistabelle ---
            acceptability_pct = (
                pd.crosstab(
                    valid_df[selected_environment_column],
                    valid_df["thermal_acceptability"],
                    normalize="index"
                ) * 100
            ).reset_index()

            # --- Dataframe mit Unknown erstellen ---
            unknown_pct = (
                pd.crosstab(
                    df[selected_environment_column],
                    df["thermal_acceptability"],
                    normalize="index"
                ) * 100
            ).reset_index()[[selected_environment_column, "Unknown"]]

            acceptability_pct = acceptability_pct.merge(
                unknown_pct,
                on=selected_environment_column,
                how="left"
            )

            # Nach Anteil akzeptabler Werte absteigend sortieren
            acceptability_pct = (
                acceptability_pct
                .sort_values(
                    by="acceptable",
                    ascending=False
                )
            )
            order = acceptability_pct[selected_environment_column].tolist()

            # Spaltenreihenfolge ändern: Unknown nach hinten
            cols = [
                selected_environment_column,
                "acceptable",
                "unacceptable",
                "Unknown"
            ]

            acceptability_pct = acceptability_pct[cols]

            # Diagramm vorbereiten
            acceptability_long = acceptability_pct.drop(
                columns=["Unknown"]
            ).melt(
                id_vars=[selected_environment_column],
                var_name="Akzeptanz",
                value_name="Prozent"
            )

            # Grafik
            chart = (
                alt.Chart(acceptability_long)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{selected_environment_column}:N",
                        title=selected_variable_environment,
                        axis=alt.Axis(labelAngle=-45),
                        sort=order,
                    ),
                    y=alt.Y(
                        "Prozent:Q",
                        title="Anteil (%)",
                        scale=alt.Scale(domain=[0, 100])
                    ),
                    color=alt.Color(
                        "Akzeptanz:N",
                        title="Thermal Acceptability"
                    ),
                    tooltip=[
                        selected_environment_column,
                        "Akzeptanz",
                        alt.Tooltip("Prozent:Q", format=".1f")
                    ]
                )
                .properties(
                    height=500
                )
            )

            st.altair_chart(chart, use_container_width=True)


            # --- Ergebnisse ---
            with col_results:
                # Ergebnisse für Klimatypen
                if selected_variable_environment == "Klimatyp":
                    st.markdown("""  
                    - Bei thermischer Akzeptanz **überwiegt tendenziell die Bewertung mit "akzeptabel" gegenüber "unakzeptabel"** bei gültigen Antworten
                    - Unterschiede:
                        - **Monsoon-influenced temperate oceanic:** höchster Anteil an Bewertungen mit "akzeptabel" bei gültigen Antworten (92%)
                        - **Tropical savanna:** höchster Anteil an Bewertungen mit "unakzeptabel" bei gültigen Antworten (71,3%)
                    - Ergebnisse sollten unter Berücksichtigung der teilweise hohen Anteile an Unknown-Antworten interpretiert werden
                    """
                    )
                elif selected_variable_environment == "Klimazone":
                    st.markdown("""  
                    - In allen vier Klimazonen ist die thermische Akzeptanz **bei den gültigen Antworten überwiegend hoch** (Anteil "akzeptabel" > Anteil "unakzeptabel")
                    - Unterschiede:                        
                        - **Continental:** höchster Anteil an Bewertungen mit "akzeptabel" bei gültigen Antworten (82.18%)
                        - **Tropical:** höchster Anteil an Bewertungen mit "unakzeptabel" bei gültigen Antworten (38.38%)
                    - Ergebnisse sollten unter Berücksichtigung der teilweise hohen Anteile an Unknown-Antworten interpretiert werden
                    """
                    )
                elif selected_variable_environment == "Region":
                    st.markdown("""
                        - Thermische Akzeptanz **unterscheidet sich** bei den gültigen Antworten **leicht zwischen den Regionen**
                                
                            - **Americas, Asia und Europe:** Bewertungen mit "akzeptabel" überwiegen gegenüber "unakzeptabel" 
                            - **Oceania:** Anteil an Bewertungen mit "unakzeptabel" geringfügig höher als mit "akzeptabel" (51,58 % vs. 48,42 %)
                                
                        - Americas: höchster Anteil an Bewertungen mit "akzeptabel" bei gültigen Antworten (84.49%)                    
                        - Ergebnisse sollten unter Berücksichtigung der teilweise hohen Anteile an Unknown-Antworten interpretiert werden
                    """
                    )
                else:
                    st.markdown("""
                        - Thermische Akzeptanz **unterscheidet sich** bei den gültigen Antworten **zwischen den Ländern stärker** als zwischen den Regionen 
                        - Unterschiede:
                            - **Slovakia:** höchster Anteil an Bewertungen mit "akzeptabel" bei gültigen Antworten (92.73%)
                            - **South Korea:** 
                                - höchster Anteil an Bewertungen mit "unakzeptabel" bei gültigen Antworten 
                                - Anteil an Bewertungen mit "unakzeptabel" höher als mit "akzeptabel" bei gültigen Antworten (66.12 % vs. 33.87 %)
                            - Alle anderen Länder: Anteil an Bewertungen mit "akzeptabel" höher als mit "unakzeptabel" bei gültigen Antworten
                        - Ergebnisse sollten unter Berücksichtigung der teilweise hohen Anteile an Unknown-Antworten interpretiert werden
                """
                )   
            
            # --- Ergebnistabelle ---
            with col_results:
                with st.expander(
                        f"**📈 Ergebnisse Thermische Akzeptanz und {selected_variable_environment} in %**"
                        ):
                        st.dataframe(acceptability_pct, use_container_width=True)

    # ---------------------------------------------------------
    # Spalten definieren
    # ---------------------------------------------------------
    col_expander, col2 = st.columns([2, 0.2])

    # --- Spalte Expander ---
    with col_expander:
        # Expander mit Hinweisen zu den Diagrammen
        with st.expander("ℹ️ Allgemeine Hinweise zum Lesen und zur Interpretation der Diagramme"):
            st.markdown("""
            - **Hinweise zum Lesen der Diagramme:**
                - Balken: stellen Mittelwerte dar
                - Rote Punkte: stellen Mediane dar
                        
            - **Hinweise zur Interpretation der Diagramme:**
                        
    
                - Teilweise stark unterschiedliche Stichprobengrößen ➝ Ergebnisse sollten vorsichtig und überwiegend deskriptiv interpretiert werden
                - Teilweise viele fehlende Werte ➝ Vergleichbarkeit zwischen den Gruppen ist eingeschränkt (z.B. bei Thermischer Akzeptanz)
                - Thermische Akzeptanz und thermische Präferenz:
                        
                    - Unknown: Anteil der ursprünglichen Antworten ohne gültige Bewertung in %
                    - acceptable/unacceptable bzw. cooler/no change/warmer: beziehen sich ausschließlich auf die gültigen Antworten
            
            """
            )
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Zusammenfassung
    # ---------------------------------------------------------
with tab4:
    # --- Überschrift ---
    st.subheader("ℹ️ Zusammenfassung der Ergebnisse")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Text ---
    st.info("""
    - Die Ergebnisse zeigen, dass es **Unterschiede in der thermischen Wahrnehmung zwischen den Gruppen der klimatischen und geografischen Variablen** gibt ➝ erklären jedoch nur einen Teil der Variation der thermischen Wahrnehmung


    - Die Unterschiede sind **mittel bis sehr schwach** ausgeprägt


    - **Stärkste Unterschiede** zeigen sich bei den **Klimatypen**


    - **Relevanz für Ziel des Projekts:** Um ideale Bedingungen für Gebäude zu schaffen, sollten die klimatischen und geografischen Gegebenheiten berücksichtigt werden


    - **Mögliche nächste Untersuchungsfrage:** Was bewirkt die Unterschiede in der thermischen Wahrnehmung zwischen den Gruppen der klimatischen/geografischen Variablen?
    """
    )