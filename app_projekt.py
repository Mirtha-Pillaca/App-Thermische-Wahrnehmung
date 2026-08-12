import warnings
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Daten laden
# ---------------------------------------------------------
@st.cache_data
def lade_csv(pfad):
    return pd.read_csv(pfad)

df = lade_csv("Daten/db_bereinigt_final.csv")
df_komplett = lade_csv("Daten/db_datensatz_komplett.csv")


# ---------------------------------------------------------
# Titel der App
# --------------------------------------------------------- 
apptitle = 'ASHRAE Data Analysis'

# ---------------------------------------------------------
# Seitenkonfigurationen
# --------------------------------------------------------- 
st.set_page_config(page_title="ASHRAE", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------
# Seitennavigation
# --------------------------------------------------------- 
pg = st.navigation(
    [
        st.Page("Startseite/Startseite.py", title="Startseite", icon=":material/home:"),
        st.Page("Einfuehrung/introduction.py", title="Einführung", icon=":material/info:"),
        st.Page("Datenbereinigung/datenbereinigung_app.py", title="Datenbereinigung", icon=":material/menu_book:"),
        st.Page("Datenbank/database.py", title="Datenbank", icon=":material/dashboard:"),
        st.Page("Datenanalyse/analytics_klima.py", title="Klima-Analyse", icon=":material/analytics:"),
        st.Page("Datenanalyse/analytics_belueftung.py", title="Belüftungsart-Analyse", icon=":material/analytics:"),
        st.Page("Datenanalyse/thermal_comfort_analysis.py", title="Physikalische Parameter-Analyse", icon=":material/analytics:"),
        st.Page("Machine_Learning/ml_ashrae.py", title="Machine Learning Modelling", icon=":material/smart_toy:"),
        st.Page("Dashboard/dashboard.py", title="Dashboard", icon=":material/analytics:"),
        st.Page("Zusammenfassung_Fazit/zusammenfassung.py", title="Zusammenfassung", icon=":material/menu_book:")
    ]
)

pg.run()