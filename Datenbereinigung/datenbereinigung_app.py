import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt
from app_projekt import df as df_bereinigt
from app_projekt import df_komplett

# ---------------------------------------------------------
# Seitenkonfigurationen
# --------------------------------------------------------- 
st.set_page_config(page_title="Datenbereinigung - ASHRAE", layout="wide",initial_sidebar_state="expanded")

# ---------------------------------------------------------
# Seitentitel
# --------------------------------------------------------- 
st.title("🔍 Bereinigung des Datensatzes")

# # ---------------------------------------------------------
# # Datensätze laden
# # --------------------------------------------------------- 
# # Bereinigter Datensatz
# @st.cache_data
# def load_data():
#     return pd.read_csv("Daten/db_bereinigt_final.csv")
# df_bereinigt = load_data()

# # Datensatz vor Standardisierung von thermal_comfort und thermal_sensation
# @st.cache_data
# def load_data():
#     return pd.read_csv("Daten/db_datensatz_komplett.csv")
# df_komplett = load_data()

# ---------------------------------------------------------
# Tabs definieren
# --------------------------------------------------------- 
tab1, tab2, tab3 = st.tabs([
    "ℹ️ Datensatz",
    "⚠️ Prozess und Herausforderungen",
    "🧹 Bereinigter Datensatz"
])

###############################################################################################################################################
###############################################################################################################################################

# ---------------------------------------------------------
# Tab1: Datensatz
# --------------------------------------------------------- 
with tab1:   

    # ---------------------------------------------------------
    # Datensatz Aufbau
    # --------------------------------------------------------- 
    st.subheader("ℹ️ Datensatz")
    st.write("Der Datensatz ist in **zwei Haupttabellen** gegliedert: ")

    col1, spacer, col2 = st.columns([2, 0.2, 2])
    # --- Spalte 1: Metadata-Tabelle ---
    with col1:
        st.markdown("""
        **`metadata` Tabelle**

        - Enthält allgemeine **Gebäude- und Studieninformationen**
        - Bereitgestellt als Standard-CSV file
        """)
    # --- Spalte 2: Measurements-Tabelle ---
    with col2:
        st.markdown("""
        **`measurements` Tabelle**

        - Enthält die **Messdaten** (z.B.)
            - Fragebogenantworten → zentral für Untersuchung der thermischen Bewertung
            - Physikalische Messdaten

        - Bereitstellung:
            - Als komprimierte CSV-Datei (.csv.gz) in UTF-8-Kodierung
        """)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Einteilung der Variablen
    # --------------------------------------------------------- 
    st.subheader("📋 Übersicht über Variablen")

    # --- Dataframe für Übersicht der Variablengruppen ---
    data = {
        "Gruppe": [
            "🏢 Gebäude- und Studiendaten",
            "👤 Personenbezogene Variablen",
            "🌡️ Umgebungsvariablen",
            "🧍 Subjektive Komfortbewertungen",
            "📊 Komfort-Indizes"
        ],
        "Beschreibung": [
            "Informationen zum Messkontext",
            "Eigenschaften der Personen",
            "Physikalische Bedingungen",
            "Komfortangaben der Personen",
            "Berechnete thermische Kennwerte"
        ],
        "Variablen (Bsp.)": [
            "building_type, cooling_type, country, climate, season",
            "age, gender, met, clo",
            "air_temperature, humidity, air_velocity",
            "thermal_sensation, thermal_comfort, thermal_preference, thermal_acceptability",
            "PMV, PPD, SET"
        ]
    }

    df_groups = pd.DataFrame(data)

    st.dataframe(
    df_groups,
    width="stretch",
    hide_index=True
    )

    # ---------------------------------------------------------
    # Expander mit Hinweis
    # ---------------------------------------------------------     
    with st.expander("ℹ️ Hinweis zu den Variablen"):
        st.markdown("""
        Der Datensatz enthält sehr viele Parameter ➝ nicht alle wurden für die Analyse und das Machine Learning genutzt
        """
        )

###############################################################################################################################################
###############################################################################################################################################

# ---------------------------------------------------------
# Tab2: Prozess und Herausforderungen
# --------------------------------------------------------- 
with tab2:

    # ---------------------------------------------------------
    # Übersicht zu Prozess und Herausforderungen
    # --------------------------------------------------------- 
    st.subheader("🧹 Prozess und Herausforderungen der Datenbereinigung")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- 1. Zusammenführen der Datensätze ---
    st.info("""
    1. **Zusammenführen** der beiden Datensätze für Analysen in Python
    """) 

    # --- 2. Bereinigung Datentypen ---
    st.info("""
    2. **Bereinigung von Datentypen** 
    """)  

    # --- 3. Fehlende Werte ---
    st.info("""
    3. 🔍 **Untersuchung der fehlenden Werte** 
    """)  

    col4, col5, col6 = st.columns([1.5, 0.2, 2])
    # Spalte 4: Herausforderung bei fehlenden Werten
    with col4:
        st.markdown("⚠️ **Herausforderung**: ")
        st.markdown("""
        - Sehr viele **fehlende Werte**
        """
        )

        # Expander mit weiteren Informationen zu fehlenden Werten
        with st.expander("Weitere Informationen"):
            st.markdown("""
            Spalten variieren stark bezüglich Anzahl der fehlenden Werte (z.B.):
                        
            - age ➝ 55% (60039 Einträge)
            - thermal_sensation ➝ 3% (2862 Einträge)
            - thermal_comfort ➝ 65% (70998 Einträge)
        """
        )
    # Spalte 5: Pfeil             
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; margin: 0; font-size: 20px;'>➡️</h1>",
            unsafe_allow_html=True,
        )
    # Spalte 6: Umgang mit fehlenden Werten
    with col6:
        st.markdown("🛠️ **Umgang mit Herausforderung:**")
        st.markdown(
        """
        - Untersuchung der fehlenden Werte auf Muster
        - **Gemeinsame Überlegungen, welche Voraussetzungen wir benötigen** für Datenanalyse und Machine Learning
        

        - **Entscheidung:**

            - kategoriale Spalten: mit "Unknown" auffüllen
            - numerische Spalten: fehlende Werte nicht bearbeiten
            - für Machine Learning: Entfernen der Zeilen mit fehlenden Werten in relevanten Variablen
        """
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # --- 4. Bearbeitung von Spalten ---
    st.info("""
    4. Bearbeitung der Spalten: 
    - **Umbenennung von Spalten** für besseres Verständnis
    - **Entfernen** von nicht benötigten Spalten 
    - Erstellen einer neuen **Spalte mit vier Hauptklimazonen** ➝ für generelle Betrachtung bei Analyse
    """)  

    # --- 5. Standardisierung ---
    st.info("""
    5. **Standardisierung**: Runden der Werte von thermischem Komfort und thermischem Empfinden für klare Kategorien 
    """)  

    col10, col11, col12 = st.columns([1.5, 0.2, 2])
    # Spalte 10: Herausforderung bei Standardisierung
    with col10:
        st.markdown("⚠️ **Herausforderung**: ")
        st.markdown("""
        - Werte in den Spalten **thermal_comfort** und **thermal_sensation** enthalten Dezimalwerte
        """
        )

        # Expander mit weiteren Informationen zu Schwierigkeit
        with st.expander("Weitere Informationen"):
            st.markdown("""
            **Worin liegt die Schwierigkeit?**
                        
            - ASHRAE Global Thermal Comfort Database II sammelt Daten aus vielen verschiedenen Studien, Ländern, Klimazonen und Gebäudetypen
            - Folge: **unterschiedliche Werte, Skalen und Formate, teilweise auch Aggregationen** für dieselben Komfortparameter
        """)
    # Spalte 11: Pfeil          
    with col11:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; margin: 0; font-size: 20px;'>➡️</h1>",
            unsafe_allow_html=True,
        )
    # Spalte 12: Umgang mit Stanardisierung
    with col12:
        st.markdown("🛠️ **Umgang mit Herausforderung:**")
        st.markdown(
        """
        - **Standardisierung** durch Runden der Dezimalwerte ➜ Für bessere Vergleichbarkeit und Auswertung der Daten
        """)

        # Expander mit weiteren Informationen zu Standardisierung
        with st.expander("Weitere Informationen"):
            st.markdown("""
            Durch Standardisierung werden alle Werte auf die **ASHRAE‑Skala** (z.B. 1–6) abgebildet
            """
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("🔢 **Runden der thermischen Komfortparameter:**")

            # 1. Thermischer Komfort
            st.markdown(
            """
            **1. Thermischer Komfort**
            """)
            
            col1, spacer, col2 = st.columns([0.5, 0.1, 0.5])

            
            # Spalte 1: Originalwerte
            with col1:
                
                # Grafik für thermal_comfort
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(df_komplett["thermal_comfort"].dropna(), bins=20, color="#4C72B0", edgecolor="white")
                ax.set_title("Originale Thermal Comfort Werte")
                ax.set_xlabel("Wert")
                ax.set_ylabel("Häufigkeit")
                st.pyplot(fig)
                st.markdown("<br>", unsafe_allow_html=True)
            # Spalte 2: Standardisierte Werte
            with col2:

                # Grafik für thermal_comfort
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(df_bereinigt["thermal_comfort"].dropna(), bins=20, color="#4C72B0", edgecolor="white")
                ax.set_title("Standardisierte Thermal Comfort Werte")
                ax.set_xlabel("Wert")
                ax.set_ylabel("Häufigkeit")
                st.pyplot(fig)
                st.markdown("<br>", unsafe_allow_html=True)

            # 2. Thermisches Empfinden
            st.markdown(
            """
            **2. Thermisches Empfinden**
            """)

            col3, spacer, col4 = st.columns([0.5, 0.1, 0.5])
            # Spalte 3: Originalwerte
            with col3:
                # Grafik für thermal_sensation
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(df_komplett["thermal_sensation"].dropna(), bins=20, color="#4C72B0", edgecolor="white")
                ax.set_title("Originale Thermal Sensation Werte")
                ax.set_xlabel("Wert")
                ax.set_ylabel("Häufigkeit")
                st.pyplot(fig)
            # Spalte 4: Standardisierte Werte
            with col4:
                # Grafik für thermal_sensation
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(df_bereinigt["thermal_sensation"].dropna(), bins=20, color="#4C72B0", edgecolor="white")
                ax.set_title("Standardisierte Thermal Sensation Werte")
                ax.set_xlabel("Wert")
                ax.set_ylabel("Häufigkeit")
                st.pyplot(fig)

        st.markdown("<br><br><br>", unsafe_allow_html=True)

###############################################################################################################################################
###############################################################################################################################################

# ---------------------------------------------------------
# Tab3: Bereinigter Datensatz
# ---------------------------------------------------------  
with tab3:
    col1, col2, col3, spacer = st.columns([1,0.2, 1, 0.3])
    # ---------------------------------------------------------  
    # Übersicht Dimensionen vor und nach Bereinigung
    # --------------------------------------------------------- 
    #  Dimensionen vor der Bereinigung 
    with col1:
        st.write("### 📏 Dimensionen vor Bereinigung")
        st.write(f"**Zeilen:** {df_komplett.shape[0]}")
        st.write(f"**Spalten:** {df_komplett.shape[1]}")
    # Pfeil
    with col2:
        st.markdown(
            "<h1 style='text-align: center; margin: 0; font-size: 20px;'>➡️</h1>",
            unsafe_allow_html=True,
        )
    # Dimensionen nach der Bereinigung
    with col3:   
        # Dimensionen nach der Bereinigung 
        st.write("### 📏 Dimensionen nach Bereinigung")
        st.write(f"**Zeilen:** {df_bereinigt.shape[0]}")
        st.write(f"**Spalten:** {df_bereinigt.shape[1]}")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------------------------------------------  
    # Tabelle Datensatz nach Bereinigung
    # --------------------------------------------------------- 
    # Überschrift
    st.subheader("🧾 Datensatz nach der Bereinigung")

    # Dataframe ausgeben lassen
    st.dataframe(df_bereinigt)


