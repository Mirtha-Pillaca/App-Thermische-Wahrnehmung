import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt
import pydeck as pdk    
import numpy as np

import json

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(page_title="Database", layout="wide", initial_sidebar_state="expanded")

st.title("Datenbank")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

# ---------------------------------------------------------
# TABS (ONLY ON MAIN PAGE)
# ---------------------------------------------------------
tab_1, tab_2, tab_3, tab_4 = st.tabs(["📘 Neon PostgreSQL – Datenbank", "🧱 Datenbank-Architektur & Optimierung", "📊 Power BI-Integration", "📋 Verfahren"])

# ---------------------------------------------------------
# SOURCE TAB
# ---------------------------------------------------------
with tab_1:
    # Title
    st.markdown("## **Cloudnative Infrastruktur & Datenmodellierung**")
   
    col_center, col_right = st.columns([5, 3])
    
    with col_center:
        # Einleitende Infobox zur Neon-Plattform
        st.info(
            "**Neon** ist eine cloudnative, serverlose PostgreSQL-Datenbank, "
            "die speziell für moderne Entwickler konzipiert wurde.\n\n"
            "Sie bietet sofortiges Branching (Datenbank-Klonen) sowie eine vollständig automatische "
            "Skalierung der Rechenleistung."
        )
    
    st.markdown("---")
    
    # Symmetrisches Zwei-Spalten-Layout für die technische Dokumentation
    col_doc1, col_doc2 = st.columns(2)
    
    with col_doc1:
        st.markdown("### 📊 **Ausgangssituation & Herausforderung**")
        st.markdown(
            "**Datensatz:** ASHRAE v2.1-Datenbank mit über **109.033 Messungen** und **60 structured Spalten**.\n\n"
            "**Zielstellung:** Aufbau einer hochverfügbaren, performanten Cloud-Infrastruktur, um Echtzeit-Analysen "
            "für das gesamte Projektteam plattformunabhängig bereitzustellen."
        )
        
with tab_2:
    st.markdown("---")
    
    col_doc1, col_doc2 = st.columns(2)
    
    with col_doc1:
        st.markdown("### 🧱 **Datenbank-Architektur & Optimierung**")
        st.markdown(
            "**Modellierung:** Erfolgreiche Überführung einer Tabelle in eine **optimierte, relationale Datenbankstruktur**.\n\n"
            "Durch diese gezielte Normalisierung wurde die Performance der Abfragen in Python signifikant optimiert."
        )
        
        st.info("**dim_buildings:** Stammdaten-Katalog für die Gebäudestrukturen (9 Spalten).")
        st.info("**fact_thermal_records:** Zentrale Faktentabelle mit 50 Metrik- und Sensor-Spalten (Sensation, Comfort, Preference, Acceptability).")
        
with tab_3:
    col_doc1, col_doc2 = st.columns(2)
    
    with col_doc1:
        # 2. Saubere Power BI Erklärung
        st.markdown("### 📊 **Power BI-Integration**")
        st.markdown(
            "**Daten-Schnittstelle:** Die normalisierte relationale Struktur ermöglicht eine direkte, "
            "native Anbindung an **Microsoft Power BI** über standardisierte PostgreSQL-Connectors."
        )

# ==============================================================================
# 🏗️ PROJEKT-ARCHITEKTUR: DATENBANK-PIPELINE & NEON CLOUD INFRASTRUKTUR
# ==============================================================================
import os 
with tab_4:
    st.header("🏗️ Datenbasis & Cloud-Infrastruktur-Prozess")
    st.caption(
        "Der folgende Abschnitt dokumentiert das vollständige Backend-Engineering des Projekts. "
        "Die Pipeline erstreckt sich von der Bereitstellung der Cloud-Datenbank über das relationale "
        "Mapping bis hin zur finalen Integration in das Business-Intelligence-Infrastruktur-System."
    )

    st.markdown("---")

    # Helper-Funktion, um Abstürze bei fehlenden oder falsch geschriebenen Bildern komplett zu verhindern
    def safe_st_image(file_path, caption_text):
        if os.path.exists(file_path):
            st.image(file_path, caption=caption_text, use_container_width=True)
        else:
            # Versucht automatisch eine korrigierte Großbuchstaben-Erweiterung zu finden (.PNG)
            alt_path = file_path.replace(".png", ".PNG")
            if os.path.exists(alt_path):
                st.image(alt_path, caption=caption_text, use_container_width=True)
            else:
                st.warning(f"⚠️ Bild nicht gefunden: '{file_path}'. Bitte überprüfen Sie den Ordner-Pfad.")

    # ------------------------------------------------------------------------------
    # ABSCHNITT 1: NEON CLOUD INITIALISIERUNG
    # ------------------------------------------------------------------------------
    st.subheader("1. Bereitstellung der Serverlosen PostgreSQL-Datenbank in Neon")
    c1, c2 = st.columns(2)

    with c1:
        safe_st_image("Datenbank/images/neon_01.jpg", "Abbildung 1: Initialisierung des Cloud-Projekts auf AWS Frankfurt.")
        st.markdown(
            "**Cloud-Provisionierung:** Einrichtung des serverlosen PostgreSQL-Clusters in der Region "
            "AWS Europe Central 1 (Frankfurt) zur Gewährleistung minimaler Latenzzeiten bei Abfragen."
        )

    with c2:
        safe_st_image("Datenbank/images/neon_02.jpg", "Abbildung 2: Architektur-Übersicht der Compute-Ressourcen im Neon-Dashboard.")
        st.markdown(
            "**Infrastruktur-Monitoring:** Überwachung von Speicher (Storage) und Rechenleistung (Compute-Units) "
            "in Echtzeit. Generierung des sicheren SSL-Verbindungsstrings für die Backend-Kopplung."
        )

    st.markdown("---")

    # ------------------------------------------------------------------------------
    # ABSCHNITT 2: DATABASE OVERVIEW & SQL OPERATIONS
    # ------------------------------------------------------------------------------
    st.subheader("2. Datenbank-Projekt-Struktur & SQL DDL-Spezifikation")
    c3, c4 = st.columns(2)

    with c3:
        safe_st_image("Datenbank/images/neon_03.jpg", "Abbildung 3: Detaillierte Projekt-Übersicht und Verbindungsparameter.")
        st.markdown(
            "**Datenbank-Konfiguration:** Verwaltung der Datenbank-Instanzen und Endpunkte. Sichere Bereitstellung "
            "der Zugriffsrechte für den DB-Owner zur Datenmanipulation."
        )

    with c4:
        safe_st_image("Datenbank/images/neon_04.jpg", "Abbildung 4: SQL DDL-Skript im integrierten Neon SQL Editor.")
        st.markdown(
            "**Tabellen-Strukturierung (DDL):** Generierung des relationalen Datenbankschemas. Erstellung der "
            "Dimensionstabelle (`dim_buildings`) und der zentralen Faktentabelle (`fact_thermal_records`) "
            "mit strikten Primärschlüssel-Restriktionen."
        )

    st.markdown("---")

    # ------------------------------------------------------------------------------
    # ABSCHNITT 3: PYTHON PIPELINE & VERIFIKATION
    # ------------------------------------------------------------------------------
    st.subheader("3. Automatisiertes Einfügen der Daten & Tabellen-Verifikation")
    c5, _ = st.columns([1,1])

    with c5:
        safe_st_image("Datenbank/images/neon_05.jpg", "Abbildung 5: Robustes Python-Skript.")
        st.markdown(
            "**ETL-Pipeline:** Automatisierter Datentransfer via Python. Das Skript bereinigt"
            "Metadaten und lädt die **109.033 Zeilen** mithilfe optimierter Blockgrößen "
            "(Chunksize = 10.000) effizient in die Cloud hoch."
        )

    st.markdown("---")
 
    # ----------------------------------------------------------------------------
    # ABSCHNITT 4: POWER BI INTEGRATION
    # ----------------------------------------------------------------------------
    st.subheader("4. Business-Intelligence-Anbindung (Power BI)")
    c6, c7 = st.columns(2)

    with c6:
        safe_st_image("Datenbank/images/neon_06.jpg", "Abbildung 6: Strukturierte Tabellen-Ansicht innerhalb der Neon-Datenbank-Konsole.")
        st.markdown(
            "**Daten-Validierung:** Direktprüfung der hochgeladenen Datensätze in der Cloud zur Gewährleistung "
            "von Datenkonsistenz und fehlerfreien Datentypen vor der BI-Verknüpfung."
        )

    with c7:
        safe_st_image("Datenbank/images/neon_07.jpg", "Abbildung 7: Konfiguration der nativen PostgreSQL-Schnittstelle in Power BI.")
        st.markdown(
            "**DirectQuery / Import-Modus:** Anbindung der Cloud-Datenbank an das analytische Frontend. "
            "Einfügen des verschlüsselten AWS-Endpunkts und Authentifizierung des DB-Owners."
        )
    
    c8, _ = st.columns([1,1])
    with c8:
        safe_st_image("Datenbank/images/neon_08.jpg", "Abbildung 8: Ausführung und Laden der Daten in die BI-Umgebung.")
        st.markdown(
            "**Verbindungs-Aufbau:** Datenübertragung aus Neon in das relationale Modell. Die Tabellen "
            "werden ohne Informationsverlust in den Hauptspeicher der BI-Anwendung geladen."
        )

    st.markdown("---")

    # ------------------------------------------------------------------------------
    # ABSCHNITT 5: RELATIONAL MODEL & COMPONENT SCHEME
    # ------------------------------------------------================--------------
    st.subheader("5. Relationales Schema & Komponenten-Integrität")
    c9, c10 = st.columns(2)

    with c9:
        safe_st_image("Datenbank/images/neon_09.jpg", "Abbildung 9: Komponenten-Übersicht des Datenmodells.")
        st.markdown(
            "**System-Architektur:** Validierung der einzelnen Datenkomponenten. Überprüfung der Datenintegrität "
            "und Vorbereitung der Kennzahlen-Berechnungen (Measures)."
        )

    with c10:
        safe_st_image("Datenbank/images/neon_10.jpg", "Abbildung 10: Finales Schema.")
        st.markdown(
            "**Datenmodellierung (Star Schema):** Verknüpfung der Faktentabelle mit der Dimensionstabelle über "
            "das gemeinsame Feld `building_id`. Dieses relationale Design sichert die referenzielle Integrität "
            "und bildet das mathematische Fundament für die Berechnungen unseres Dashboards."
        )
 
    c11, _ = st.columns([1,1])
    with c11:
        safe_st_image("Datenbank/images/neon_11.jpg", "Abbildung 11: Power Bi Relationales Datenmodell.")
       