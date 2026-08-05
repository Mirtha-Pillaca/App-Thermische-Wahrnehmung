import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 🌟 Importiert für die perfekte Skalierung der Legenden-Quadrate
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Belüftungsart", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Belüftungsart-Analyse")

# ============================================================================== 
# 🧠 1. INITIALISIERUNG DES SESSION STATES (GANZ OBEN IM SCRIPT PLATZIEREN) 
# ============================================================================== 
if "reset_trigger_t2" not in st.session_state: 
    st.session_state.reset_trigger_t2 = False 

if "reset_trigger_t3" not in st.session_state: 
    st.session_state.reset_trigger_t3 = False 

# Standardwerte für Reiter 2 (Belüftungsart)
DEFAULT_VALUES_T2 = { 
    "geo_opt_t2": "Region",       
    "geo_cho_t2": "Americas",     
    "bld_t2": "classroom",           
    "cool_t2": "air conditioned"
} 

# Standardwerte für Reiter 3 (Alter und Geschlecht)
DEFAULT_VALUES_T3 = { 
    "geo_opt_t3": "Region",       
    "geo_cho_t3": "Americas",     
    "bld_t3": "classroom",           
    "gen_t3": "female",           
    "cool_t3": "air conditioned",  
    "sld_t3": (06.0, 95.0)        
} 

# Wenn der Button im Tab 2 gedrückt wird, setzen wir NUR Tab 2 zurück
if st.session_state.reset_trigger_t2: 
    for key, value in DEFAULT_VALUES_T2.items(): 
        st.session_state[key] = value 
    st.session_state.reset_trigger_t2 = False 

# Wenn der Button im Tab 3 gedrückt wird, setzen wir NUR Tab 3 zurück
if st.session_state.reset_trigger_t3: 
    for key, value in DEFAULT_VALUES_T3.items(): 
        st.session_state[key] = value 
    st.session_state.reset_trigger_t3 = False 

# ==============================================================================
# 🛠️ 1. GLOBALE FUNKTIONEN (MÜSSEN AN ERSTER STELLE STEHEN)
# ==============================================================================
def map_tsv(v): 
    if pd.isna(v): return None 
    if v <= -2.5: return -3 
    elif v <= -1.5: return -2 
    elif v <= -0.5: return -1 
    elif v < 0.5: return 0 
    elif v < 1.5: return 1 
    elif v < 2.5: return 2 
    else: return 3 

def map_tc(v): 
    if pd.isna(v): return None 
    if v < 1.5: return 1 
    elif v < 2.5: return 2 
    elif v < 3.5: return 3 
    elif v < 4.5: return 4 
    elif v < 5.5: return 5 
    else: return 6 

def custom_geo_sort(values_list):
    def sort_logic(x):
        item_str = str(x).lower().strip()
        if item_str == "americas":
            return (0, item_str)
        elif "unknown" in item_str or "unbekannt" in item_str or "nan" in item_str:
            return (2, item_str)
        else:
            return (1, item_str)
    return sorted(values_list, key=sort_logic)

# 📊 PLOT-FUNKTION: Jetzt mit perfekt formatierten Legenden ohne Überlappung
def plot_comfort_variable(series, labels, colors, title): 
    series = pd.to_numeric(series, errors="coerce").dropna() 
    counts = series.value_counts().sort_index() 
    total = counts.sum() 
    if total == 0:
        return False
        
    fig, ax = plt.subplots(figsize=(6, 4.2)) 
    x_positions = [str(level) for level in counts.index]
    y_values = counts.values
    bar_colors = [colors[level] for level in counts.index]
    
    bars = ax.bar(x_positions, y_values, color=bar_colors)
    
    # Platzierung der Zahlen INNERHALB der Balken (va="top")
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height - (height * 0.05) - 0.2, 
                f"{int(height)}", 
                ha="center", va="top", fontsize=9, fontweight='bold', color="black"
            ) 
    
    # 🌟 KORREKTUR: Verwendung von mpatches.Patch anstelle von dicken Linien,
    # um unschöne Überlappungen in kleinen Diagrammen komplett zu eliminieren.
    legend_patches = []
    for level in sorted(counts.index):
        patch = mpatches.Patch(color=colors[level], label=labels[level])
        legend_patches.append(patch)
    
    ax.set_title(title, fontsize=11, pad=10) 
    ax.set_xlabel("Level Verteilung", fontsize=9) 
    ax.set_ylabel("Anzahl", fontsize=9) 
    
    plt.xticks(rotation=0, ha="center")
    
    # 🌟 KORREKTUR: handletextpad sorgt für ausreichend Platz zwischen Quadrat und Text
    ax.legend(
        handles=legend_patches, 
        title="Legende", 
        loc="best", 
        framealpha=0.9, 
        fontsize=8,
        handletextpad=0.8,
        handlelength=1.2
    )
    
    plt.tight_layout()
    st.pyplot(fig, width=550)
    return True

# ==============================================================================
# 🎨 2. DICTIONARIES UND CONFIGURATION
# ==============================================================================
tsv_labels = { -3: "–3 Sehr kalt", -2: "–2 Kalt", -1: "–1 Kühl", 0: "0 Neutral", 1: "+1 Warm", 2: "+2 Heiß", 3: "+3 Sehr heiß" } 
tsv_colors = { -3: "#4575b4", -2: "#74add1", -1: "#abd9e9", 0: "#d9d9d9", 1: "#fdae61", 2: "#f46d43", 3: "#d73027" } 
tp_labels = { -1: "–1 Kühler bevorzugt", 0: "0 Keine Präferenz", 1: "+1 Wärmer bevorzugt" } 
tp_colors = { -1: "#74add1", 0: "#d9d9d9", 1: "#f46d43" } 
tc_labels = { 1: "1 Sehr unkomfortabel", 2: "2 Leicht unkomfortabel", 3: "3 Akzeptabel / Neutral", 4: "4 Leicht komfortabel", 5: "5 komfortabel", 6: "6 Sehr komfortabel" } 
tc_colors = { 1: "#fc8d59", 2: "#fee08b", 3: "#d9d9d9", 4: "#a6d96a", 5: "#1a9850", 6: "#006837" } 
ta_labels = { 0: "0 Unakzeptabel", 1: "1 Akzeptabel" }
ta_colors = { 0: "#d73027", 1: "#1a9850" }

tp_map = {"cooler": -1, "no change": 0, "warmer": 1, "unknown": np.nan} 
ta_map = {"acceptable": 1, "unacceptable": 0, "unknown": np.nan}

# REITER INITIALISIERUNG
tab1, tab2, tab3 = st.tabs(["🌍 Globale Übersicht", "⚡Belüftungsart", "👥 Alter und Gender"])

# ==============================================================================
# 💾 3. DATENLADUNG & VARIABLEN-MAPPING
# ==============================================================================
@st.cache_data
def load_data():
    return pd.read_csv("Daten/db_bereinigt_final.csv")
df = load_data()

df["thermal_sensation_cat"] = df["thermal_sensation"].apply(map_tsv) 
df["thermal_preference_cat"] = df["thermal_preference"].map(tp_map) 
df["thermal_comfort_cat"] = df["thermal_comfort"].apply(map_tc) 
df["thermal_acceptability_cat"] = df["thermal_acceptability"].map(ta_map)

# ==============================================================================
# 🌐 TAB 1: GLOBALE ÜBERSICHT (ZUSAMMENGEFASSTE BIG-DATA ANALYSE IN 2x2 GRID)
# ==============================================================================

with tab1:
    # Absolute kategoriale Zuweisung der Komfortmetriken
    df["thermal_sensation_cat"] = df["thermal_sensation"].apply(map_tsv) 
    df["thermal_preference_cat"] = df["thermal_preference"].map(tp_map) 
    df["thermal_comfort_cat"] = df["thermal_comfort"].apply(map_tc) 
    df["thermal_acceptability_cat"] = df["thermal_acceptability"].map(ta_map)
    
    # Normalisierung der Spaltenwerte zur Vermeidung von Formatierungsfehlern
    df['cooling_type'] = df['cooling_type'].fillna('unknown').astype(str).str.lower().str.strip()
    df['building_type'] = df['building_type'].fillna('unknown').astype(str).str.strip()
    df['gender'] = df['gender'].fillna('unknown').astype(str).str.lower().str.strip()
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # 🌟 MAẞNAHME: Ausschluss von 'unknown'-Werten auf globaler Ebene für Tab 1
    df_global = df[
        (df['cooling_type'] != 'unknown') & 
        (df['building_type'].str.lower() != 'unknown')
    ]
    
    st.subheader("Wissenschaftlicher Leitfaden: Globale makroskopische Analyse nach Gebäude- und Belüftungsstruktur")
    st.caption(
        "Diese Übersicht segmentiert das weltweite Datenvolumen der ASHRAE v2.1-Datenbank automatisch. "
        "Die 4 Hauptkomfortparameter werden für jede Kombination aus Gebäudetyp und Belüftungsart separat dargestellt "
        "und direkt mit dem demografischen Profil (Alter und Gender) der jeweiligen Nutzergruppe verknüpft."
    )
    
    # Schleife über alle bereinigten Gebäudetypen im Datensatz
    gebaeudetypen = sorted(df_global['building_type'].unique().tolist())
    
    # ---------------------------------------------------------
    # 📊 Zusammenfassung 
    # --------------------------------------------------------- 
  
    with st.expander(
        #st.subheader("ℹ️ Zusammenfassung")
        f"**ℹ️ Zusammenfassung**"
    ):
        st.info(
            """
            **Ergebnisse:** \n
            - Thermal Comfort: Zeigt die Verteilung des thermischen Komfortindex spezifisch für die Raumkonfiguration. \n
            - Thermal Sensation: Dokumentiert die sensorische Wahrnehmung der operativen Temperatur im Raum. \n
            - Thermal Preference: Spiegelt den direkten Wunsch der Nutzer nach Temperaturänderungen (kühler/wärmer) wieder. \n
            - Thermal Acceptability: Kennzeichnet den prozentualen Anteil der Stimmen, die das Raumklima als akzeptabel bewerten. \n
            - Die Globale Übersicht liefert uns universelle Benchmarks für strategische Richtlinien.\n
            - Die demografische Analyse ermöglicht uns schließlich ein präzises Nutzer-Targeting, um den Energieverbrauch exakt an das biologische Profil der  Menschen anzupassen.

            """
        )

    for gebaeude in gebaeudetypen:
        df_bldg = df_global[df_global['building_type'] == gebaeude]
        
        if df_bldg.empty:
            continue
            
        st.markdown(f"## 🏢 Gebäudetyp: {gebaeude}")
        
        # Sortierung der verbleibenden validen Belüftungsarten
        belueftungsarten = sorted(df_bldg['cooling_type'].unique().tolist())
        
        for belueftung in belueftungsarten:
            df_final_t1 = df_bldg[df_bldg['cooling_type'] == belueftung]
            
            total_voten = len(df_final_t1)
            if total_voten == 0:
                continue
                
            avg_age_seg = df_final_t1['age'].mean() if not df_final_t1['age'].dropna().empty else 0.0
            
            # Verteilung der Geschlechter ermitteln
            gender_counts = df_final_t1['gender'].value_counts()
            female_votes = gender_counts.get('female', 0)
            male_votes = gender_counts.get('male', 0)
            undefined_votes = gender_counts.get('undefined', 0)
            unknown_votes = gender_counts.get('unknown', 0)  

            total_str = f"{total_voten:,}".replace(",", ".")
            female_str = f"{female_votes:,}".replace(",", ".")
            male_str = f"{male_votes:,}".replace(",", ".")
            undefined_str = f"{undefined_votes:,}".replace(",", ".")
            unknown_str = f"{unknown_votes:,}".replace(",", ".")
            
            # Formatierung des Durchschnittsalters mit Komma für die deutsche Dezimalstelle (z.B. 21,2)
            avg_age_str = f"{avg_age_seg:.1f}".replace(".", ",")
            
            # Formatierung für die Anzeige der Belüftungsart im Titel
            belueftung_title = belueftung.replace('-', ' ').title()
            st.markdown(f"### ⚡ Belüftungsart: {belueftung_title}")
            st.markdown(
                # f"**Statistik:** Gesamtstimmen: `{total_voten:,}` | ø-Alter: `{avg_age_seg:.1f} Jahre` | "
                # f"Demografie: 👩 Frauen: `{female_votes:,}` | 👨 Männer: `{male_votes:,}` | 👤 Undefined: `{undefined_votes:,}` | 👥 Unbekannt: `{unknown_votes:,}`"
                f"**Statistik:** Gesamtstimmen: `{total_str}` | ø-Alter: `{avg_age_str} Jahre` | "
                f"Demografie: 👩 Frauen: `{female_str}` | 👨 Männer: `{male_str}` | 👤 Undefined: `{undefined_str}` | 👥 Unbekannt: `{unknown_str}`"                
            )
            
            # Evaluierung der Zeilenmenge vor dem Zeichnen der Matrix
            has_c1 = not pd.to_numeric(df_final_t1["thermal_comfort_cat"], errors="coerce").dropna().empty
            has_c2 = not pd.to_numeric(df_final_t1["thermal_sensation_cat"], errors="coerce").dropna().empty
            
            # 📊 GRUPPE 1: Comfort und Sensation nebeneinander mit deiner Breiten-Fixierung (width=550)
            col_r1_1, col_r1_2 = st.columns(2)
            with col_r1_1: ####
                if has_c1:
                    plot_comfort_variable(df_final_t1["thermal_comfort_cat"], tc_labels, tc_colors, f"Thermal Comfort ({gebaeude} - {belueftung_title})")
                else:
                    st.container()
            with col_r1_2:
                if has_c2:####
                    plot_comfort_variable(df_final_t1["thermal_sensation_cat"], tsv_labels, tsv_colors, f"Thermal Sensation ({gebaeude} - {belueftung_title})")
            
            # Evaluierung der Zeilenmenge für die zweite Reihe
            has_c3 = not pd.to_numeric(df_final_t1["thermal_preference_cat"], errors="coerce").dropna().empty
            has_c4 = not pd.to_numeric(df_final_t1["thermal_acceptability_cat"], errors="coerce").dropna().empty
            
            # 📊 GRUPPE 2: Preference und Acceptability nebeneinander mit deiner Breiten-Fixierung (width=550)
            col_r2_1, col_r2_2 = st.columns(2)
            with col_r2_1:
                if has_c3:####
                    plot_comfort_variable(df_final_t1["thermal_preference_cat"], tp_labels, tp_colors, f"Thermal Preference ({gebaeude} - {belueftung_title})")
                else:
                    st.container()
            with col_r2_2:
                if has_c4:####
                    plot_comfort_variable(df_final_t1["thermal_acceptability_cat"], ta_labels, ta_colors, f"Thermal Acceptability ({gebaeude} - {belueftung_title})")
                else:
                    st.container()
            
            st.markdown("---") # Trennlinie zwischen den verschiedenen Belüftungsarten
        st.markdown("<br><br>", unsafe_allow_html=True) # Deutlicher Abstand zum nächsten Gebäudety
     
# ==============================================================================
# 📊 TAB 2: GEOGRAFISCHE KOMFORTANALYSE (DINAMISCHES 2x2 NEBENEINANDER LAYOUT)
# ==============================================================================
with tab2:
    st.subheader("Analyse-Leitfaden: Beeinflusst die Belüftungsart den aktuellen Parameter?")
    
    geo_map = { "Region": "region", "Land": "country", "Stadt": "city" } 
    col_t2_f1, col_t2_f2, col_t2_f3, col_t2_f4 = st.columns(4)
    #col_f4,_ = st.columns([1,1])

    with col_t2_f1:
        geo_option = st.selectbox("Geografische Verteilung anzeigen nach:", list(geo_map.keys()), key="geo_opt_t1") 
        geo_colname = geo_map[geo_option] 

     ### Button ####
        # ============================================================================== 
        # 🎛️ INJECTION TAB 2: RESET-BUTTON DIREKT UNTER COOLING TYPE
        # ============================================================================== 
        def trigger_system_reset_t2(): 
            st.session_state.reset_trigger_t2 = True 
                
        st.button(
                "🔄 Filter zurücksetzen", 
                on_click=trigger_system_reset_t2, 
                key="btn_reset_tab2",
                help="Setzt alle Filter aller Reiter sofort auf die Standardeinstellungen zurück.", 
        )
        ######       
    with col_t2_f2:
        raw_geo_values_t1 = df[geo_colname].dropna().unique().tolist()
        sorted_geo_values_t1 = custom_geo_sort(raw_geo_values_t1)
        geo_choice = st.selectbox(f"{geo_option} auswählen:", sorted_geo_values_t1, key="geo_cho_t1") 
    with col_t2_f3:
        df['building_type'] = df['building_type'].fillna('Unknown')
        lista_building_types = sorted(df['building_type'].unique().tolist())
        try:
            default_bldg_index = [str(x).lower().strip() for x in lista_building_types].index('office')
        except ValueError:
            default_bldg_index = 0
            
        building_choice = st.selectbox(
            "Building Type auswählen:", 
            lista_building_types, 
            index=default_bldg_index, 
            key="bld_t2"
        )
    with col_t2_f4:
    # 🌟 NEUER FILTER FÜR COOLING TYPE (TAB 1)
        df['cooling_type'] = df['cooling_type'].fillna('unknown')
        raw_cooling_t1 = df['cooling_type'].unique().tolist()
            
        def cooling_sort_logic(x):
                c_str = str(x).lower().strip()
                if "air conditioned" in c_str or "air conditioned" in c_str or "klimatisiert" in c_str:
                    return (0, c_str) # Zeigt Air Conditioned als Standard an erster Stelle
                elif "unknown" in c_str or "unbekannt" in c_str:
                    return (2, c_str) # Schiebt unknown ans absolute Ende
                else:
                    return (1, c_str) # Alle anderen (z.B. natural ventilation) in die Mitte
                    
        sorted_cooling_t1 = sorted(raw_cooling_t1, key=cooling_sort_logic)
        cooling_choice = st.selectbox("Cooling Type auswählen:", sorted_cooling_t1, key="cool_t1")
 
    df_t1_filtered = df[
        (df[geo_colname] == geo_choice) & 
        (df['building_type'] == building_choice)
    ]
    
    st.markdown("---")
    st.markdown(f"### 📈 Thermische Verteilung in {geo_choice} ({building_choice})")
    
    row1_has_data1 = not pd.to_numeric(df_t1_filtered["thermal_comfort_cat"], errors="coerce").dropna().empty
    row1_has_data2 = not pd.to_numeric(df_t1_filtered["thermal_sensation_cat"], errors="coerce").dropna().empty
    
    if row1_has_data1 and row1_has_data2:
        r1_col1, r1_col2 = st.columns(2)
        with r1_col1:
            plot_comfort_variable(df_t1_filtered["thermal_comfort_cat"], tc_labels, tc_colors, "Thermal Comfort Verteilung")
            st.info(
                "**Analyse-Leitfaden**: Natürlich belüftete Gebäude weisen oft breitere Verteilungen auf, da Nutzer adaptive Anpassungsmechanismen wie das Öffnen von Fenstern nutzen.")
        with r1_col2:
            plot_comfort_variable(df_t1_filtered["thermal_sensation_cat"], tsv_labels, tsv_colors, "Thermal Sensation Verteilung")
            st.info("**Analyse-Leitfaden**: Die Belüftungsart steuer direkt die Luftgeschwindigkeit und die operative Temperatur, was die physikalischen Parameter massiv verschiebt.")
    elif row1_has_data1:
        _, center_col, _ = st.columns([0.25, 1.5, 0.25])
        with center_col:
            plot_comfort_variable(df_t1_filtered["thermal_comfort_cat"], tc_labels, tc_colors, "Thermal Comfort Verteilung")
            st.info("**Analyse-Leitfaden**: Natürlich belüftete Gebäude weisen oft breitere Verteilungen auf, da Nutzer adaptive Anpassungsmechanismen wie das Öffnen von Fenstern nutzen.")
    elif row1_has_data2:
        _, center_col, _ = st.columns([0.25, 1.5, 0.25])
        with center_col:
            plot_comfort_variable(df_t1_filtered["thermal_sensation_cat"], tsv_labels, tsv_colors, "Thermal Sensation Verteilung")
            st.info("**Analyse-Leitfaden**: Die Belüftungsart steuer direkt die Luftgeschwindigkeit und die operative Temperatur, was die physikalischen Parameter massiv verschiebt.")

    st.markdown("---")

    row2_has_data3 = not pd.to_numeric(df_t1_filtered["thermal_preference_cat"], errors="coerce").dropna().empty
    row2_has_data4 = not pd.to_numeric(df_t1_filtered["thermal_acceptability_cat"], errors="coerce").dropna().empty
    
    if row2_has_data3 and row2_has_data4:
        r2_col1, r2_col2 = st.columns(2)
        with r2_col1:
            plot_comfort_variable(df_t1_filtered["thermal_preference_cat"], tp_labels, tp_colors, "Thermal Preference Verteilung")
            st.info("**Analyse-Leitfaden**: In natürlich belüfteten Gebäuden tolerieren die Befragten höhere Innentemperaturen. Nutzer in mechanisch gekühlten Räumen fordern permanent kühlere Zustände.")
        with r2_col2:
            plot_comfort_variable(df_t1_filtered["thermal_acceptability_cat"], ta_labels, ta_colors, "Thermal Acceptability Verteilung")
            st.info("**Analyse-Leitfaden**: Die thermische Akzeptanz sinkt in klimatisierten Räumen drastisch, wenn die relative Luftfeuchtigkeit außerhalb des optimalen Bereichs liegt.")
    elif row2_has_data3:
        _, center_col, _ = st.columns([0.25, 1.5, 0.25])
        with center_col:
            plot_comfort_variable(df_t1_filtered["thermal_preference_cat"], tp_labels, tp_colors, "Thermal Preference Verteilung")
            st.info("**Analyse-Leitfaden**: In natürlich belüfteten Gebäuden tolerieren die Befragten höhere Innentemperaturen. Nutzer in mechanisch gekühlten Räumen fordern permanent kühlere Zustände.")
    elif row2_has_data4:
        _, center_col, _ = st.columns([0.25, 1.5, 0.25])
        with center_col:
            plot_comfort_variable(df_t1_filtered["thermal_acceptability_cat"], ta_labels, ta_colors, "Thermal Acceptability Verteilung")
                
# ==============================================================================
# 👥 TAB 3: DEMOGRAFISCHE KOMFORTANALYSE (STRIKT GENDER & ALTER IN 2x2 GRID)
# ==============================================================================
with tab3:
    
    st.subheader("Analyse-Leitfaden: Beeinflusst die Belüftungsart den aktuellen Parameter?")
    
    # Aufteilung der Benutzeroberfläche in Filter und interaktiven Grafikbereich
    col_t3_f1, col_t3_f2, col_t3_f3, col_t3_f5 = st.columns(4)
    col_t3_f4, col_t3_f6 = st.columns(2)
    
    with col_t3_f1:
        geo_option_t2 = st.selectbox("Geografische Verteilung anzeigen nach:", list(geo_map.keys()), key="geo_opt_t2") 
        geo_colname_t2 = geo_map[geo_option_t2] 
    with col_t3_f2:
        # 🌟 Sortierung: Americas immer zuerst, Unknown am Ende (Tab 2)
        raw_geo_values_t2 = df[geo_colname_t2].dropna().unique().tolist()
        sorted_geo_values_t2 = custom_geo_sort(raw_geo_values_t2)
        geo_choice_t2 = st.selectbox(f"{geo_option_t2} auswählen:", sorted_geo_values_t2, key="geo_cho_t2") 

    ### Button ###
    # ============================================================================== 
    # 🎛️ INJECTION TAB 3: RESET-BUTTON DIREKT UNTER REGION
    # ============================================================================== 
    def trigger_system_reset_t3(): 
        st.session_state.reset_trigger_t3 = True 

    col_btn_t2_1, col_btn_t2_center, col_btn_t2_2 = st.columns([1, 0.4, 1])
    st.button(
        "🔄 Filter zurücksetzen", 
        on_click=trigger_system_reset_t3, 
        key="btn_reset_tab3", # Key única para el Tab 3
        help="Setzt alle Filter aller Reiter sofort auf die Standardeinstellungen zurück.", 
    )
    ######
    with col_t3_f3:
        df['building_type'] = df['building_type'].fillna('Unknown')
        lista_building_types = sorted(df['building_type'].unique().tolist())
        
        try:
            default_bldg_index = [str(x).lower().strip() for x in lista_building_types].index('office')
        except ValueError:
            default_bldg_index = 0
            
        building_choice_t3 = st.selectbox(
            "Building Type auswählen:", 
            lista_building_types, 
            index=default_bldg_index, 
            key="bld_t3"
        )    
    with col_t3_f4:
        df['gender'] = df['gender'].fillna('unknown')
        
        # Erstellt die exakte Wunsch-Reihenfolge basierend auf den vorhandenen Werten
        raw_genders = df['gender'].unique().tolist()
        
        def gender_sort_logic(x):
            g_str = str(x).lower().strip()
            if "female" in g_str:
                return (0, g_str)
            elif "male" in g_str:
                return (1, g_str)
            elif "undefined" in g_str or "indefizierte" in g_str:
                return (2, g_str)
            else:
                return (3, g_str) # unknown und andere nulos landen ganz unten
                
        lista_genders = sorted(raw_genders, key=gender_sort_logic)
        gender_choice = st.selectbox("Gender auswählen:", lista_genders, key="gen_t2")    
    
    with col_t3_f5:
        # 🌟 NEUER FILTER FÜR COOLING TYPE (TAB 2)
        df['cooling_type'] = df['cooling_type'].fillna('unknown')
        raw_cooling_t2 = df['cooling_type'].unique().tolist()
        
        def cooling_sort_logic_t2(x):
            c_str = str(x).lower().strip()
            if "air condition" in c_str or "air-condition" in c_str or "klimatisiert" in c_str:
                return (0, c_str)
            elif "unknown" in c_str or "unbekannt" in c_str:
                return (2, c_str)
            else:
                return (1, c_str)
                
        sorted_cooling_t2 = sorted(raw_cooling_t2, key=cooling_sort_logic_t2)
        cooling_choice_t2 = st.selectbox("Cooling Type auswählen:", sorted_cooling_t2, key="cool_t2")

    with col_t3_f6:    
    # Kontinuierlicher numerischer Altersschieberegler für präzise Kohorten-Analysen
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        edad_min = float(df['age'].min()) if not pd.isna(df['age'].min()) else 0.0
        edad_max = float(df['age'].max()) if not pd.isna(df['age'].max()) else 100.0
         
        rango_edad = st.slider(
        "Alter (Age) Bereich:", 
        min_value=edad_min, 
        max_value=edad_max, 
        value=(48.0, 66.0), 
        step=1.0, 
        key="sld_t3"
        )

        # 🌟 AKTUALISIERTE FILTERPIPELINE FÜR TAB 2 (Inklusive Cooling Type)
        edad_min_sel, edad_max_sel = rango_edad
        df_t2_filtered = df[
            (df[geo_colname_t2] == geo_choice_t2) & 
            (df['building_type'] == building_choice_t3) &
            (df['gender'] == gender_choice) &
            (df['cooling_type'] == cooling_choice_t2) &
            (df['age'] >= edad_min_sel) &
            (df['age'] <= edad_max_sel)
        ]
           
    # Ausführung der kaskadierenden Datenfilterung im Hintergrund
    edad_min_sel, edad_max_sel = rango_edad
    df_t2_filtered = df[
        (df[geo_colname_t2] == geo_choice_t2) & 
        (df['building_type'] == building_choice_t3) &
        (df['gender'] == gender_choice) &
        (df['age'] >= edad_min_sel) &
        (df['age'] <= edad_max_sel)
    ]
    
    st.markdown("---")
    st.header(f"👥 Demografisches Profil ({gender_choice}, Age: {int(edad_min_sel)}-{int(edad_max_sel)})")
    
    # 🌟 FILA 1 DEMOGRAFISCHE: Steuerung und automatische Zentrierung bei fehlenden Daten
    t2_has_data1 = not pd.to_numeric(df_t2_filtered["thermal_comfort_cat"], errors="coerce").dropna().empty
    t2_has_data2 = not pd.to_numeric(df_t2_filtered["thermal_sensation_cat"], errors="coerce").dropna().empty
    
    if t2_has_data1 and t2_has_data2:
        t2_r1c1, t2_r1c2 = st.columns(2)
        with t2_r1c1:
            plot_comfort_variable(df_t2_filtered["thermal_comfort_cat"], tc_labels, tc_colors, "Thermal Comfort (Demografie)")
            st.info("**Analyse-Leitfaden:** *Die demografische Verteilung zeigt, wie das Zusammenspiel aus Lüftungssystem und Genderspezifikation das Wohlbefinden prägt. Bestimmte Gruppen zeigen eine höhere Toleranz in adaptiven Umgebungen.*")
        with t2_r1c2:
            plot_comfort_variable(df_t2_filtered["thermal_sensation_cat"], tsv_labels, tsv_colors, "Thermal Sensation (Demografie)")
            st.info("**Analyse-Leitfaden:** *Altersabhängige Stoffwechselraten verändern die sensorische Wahrnehmung der operativen Raumtemperatur drastisch. Ältere Kohorten reagieren in frei belüfteten Räumen empfindlicher auf Luftbewegungen.*")
    elif t2_has_data1:
        _, center_col, _ = st.columns([0.25, 1.5, 0.25])
        with center_col:
            plot_comfort_variable(df_t2_filtered["thermal_comfort_cat"], tc_labels, tc_colors, "Thermal Comfort (Demografie)")
            st.info("**Analyse-Leitfaden:** *Die demografische Verteilung zeigt, wie das Zusammenspiel aus lüftungssystem und Genderspezifikation das Wohlbefinden prägt. Bestimmte Gruppen zeigen eine höhere Toleranz in adaptiven Umgebungen.*")
    elif t2_has_data2:
        _, center_col, _ = st.columns([0.25, 1.5, 0.25])
        with center_col:
            plot_comfort_variable(df_t2_filtered["thermal_sensation_cat"], tsv_labels, tsv_colors, "Thermal Sensation (Demografie)")
            st.info("**Analyse-Leitfaden:** *Altersabhängige Stoffwechselraten verändern die sensorische Wahrnehmung der operativen Raumtemperatur drastisch. Ältere Kohorten reagieren in frei belüfteten Räumen empfindlicher auf Luftbewegungen.*")

    st.markdown("---")

    # 🌟 FILA 2 DEMOGRAFISCHE: Steuerung und automatische Zentrierung bei fehlenden Daten
    t2_has_data3 = not pd.to_numeric(df_t2_filtered["thermal_preference_cat"], errors="coerce").dropna().empty
    t2_has_data4 = not pd.to_numeric(df_t2_filtered["thermal_acceptability_cat"], errors="coerce").dropna().empty
    
    if t2_has_data3 and t2_has_data4:
        t2_r2c1, t2_r2c2 = st.columns(2)
        with t2_r2c1:
            plot_comfort_variable(df_t2_filtered["thermal_preference_cat"], tp_labels, tp_colors, "Thermal Preference (Demografie)")
            st.info("**Analyse-Leitfaden:** *Die Neigung zur Anforderung kühlerer Luftströme variiert signifikant je nach Gender, wobei mechanische Kühlsysteme oft zu ungleichmäßigen Präferenz-Clustern führen.*")
        with t2_r2c2:
            plot_comfort_variable(df_t2_filtered["thermal_acceptability_cat"], ta_labels, ta_colors, "Thermal Acceptability (Demografie)")
            st.info("**Analyse-Leitfaden:** *Die Raumklima-Akzeptanz stabilisiert sich bei Gruppen, denen das Gebäude eine hohe adaptive Freiheit gewährt. Unbekannte demografische Variablen korrelieren oft mit standardisierter mechanischer Belüftung.*")
    elif t2_has_data3:
        _, center_col, _ = st.columns([0.25, 1.5, 0.25])
        with center_col:
            plot_comfort_variable(df_t2_filtered["thermal_preference_cat"], tp_labels, tp_colors, "Thermal Preference (Demografie)")
            st.info("**Analyse-Leitfaden:** *Die Neigung zur Anforderung kühlerer Luftströme variiert signifikant zwischen je nach Gender, wobei mechanische Kühlsysteme oft zu ungleichmäßigen Präferenz-Clustern führen.*")
    elif t2_has_data4:
        _, center_col, _ = st.columns([0.25, 1.5, 0.25])
        with center_col:
            plot_comfort_variable(df_t2_filtered["thermal_acceptability_cat"], ta_labels, ta_colors, "Thermal Acceptability (Demografie)")
            st.info("**Analyse-Leitfaden:** *Die Raumklima-Akzeptanz stabilisiert sich bei Gruppen, denen das Gebäude eine hohe adaptive Freiheit gewährt. Unbekannte demografische Variablen korrelieren oft mit standardisierter mechanischer Belüftung.*")


