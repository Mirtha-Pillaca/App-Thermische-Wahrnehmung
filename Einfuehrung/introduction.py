import streamlit as st

# ---------------------------------------------------------
# Seitenkonfigurationen
# --------------------------------------------------------- 
st.set_page_config(
    page_title="Introduction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Seitentitel
# ---------------------------------------------------------
st.title("🏢 Thermische Wahrnehmung in Innenräumen")

# -------------------------
# Tabs definieren
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Wahl des Themas",
    "2️⃣ Datenquelle",
    "3️⃣ Team und Fragestellungen",
    "4️⃣ Tools"
])

#########################################################################################################
#########################################################################################################

# ---------------------------------------------------------
# TAB 1 – Wahl des Themas
# ---------------------------------------------------------
with tab1:
    # --- Titel ---
    st.subheader("Ein datengetriebener Ansatz für Komfort & Energieeffizienz")

    # --- Bild zu Auftrag, Ansatz und Ziel ---
    st.image("Einfuehrung/images/bild_intro.png", width=1300)
    st.write("Bild: KI-generiert")
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- Text zu Themenwahl ---
    st.markdown(
    """
    #### ℹ️ **Thermische Wahrnehmung in Innenräumen ist ein wichtiges Thema** ➝ Es verbindet:
    """
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""      
                    
        🧍 Menschen (Produktivität, Gesundheit)
                

        🌍 Klima
                
        
        💻 Technologie
                

        ⚡ Energie (Effizienz, Nachhaltigkeit)
        """
        )

#########################################################################################################
#########################################################################################################

# ---------------------------------------------------------
# TAB 2 – Datenquelle und wichtige Variablen
# ---------------------------------------------------------
with tab2:
    # ---------------------------------------------------------
    # Datenquelle 
    # ---------------------------------------------------------
    # --- Titel ---
    st.subheader("🧾 Datenquelle: ASHRAE Global Thermal Comfort Database v2.1")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Text mit Informationen zu Datenquelle ---
    st.write("Erstellt von der *American Society of Heating, Refrigerating and Air‑Conditioning Engineers*")
    st.markdown(
    """
    #### **🔗 Dataset Source:**

    [ASHRAE Global Thermal Comfort Database II](https://datadryad.org/dataset/doi:10.6078/D1F671) 
    """
    )

    # --- Expander mit weiteren Informationen zur Datenquelle ---
    with st.expander("Weitere Informationen zur Datenquelle"):
        st.markdown(
            """
            - Umfassende Datenbank zur **Untersuchung des thermischen Komforts in Gebäuden weltweit**
            - Zusammenstellung von Feldstudien aus dem Zeitraum **1995–2016**
            - Datensatz findet sich in verschiedenen Versionen z.B. bei kaggle 
            - Für dieses Projekt wurde der **originale Datensatz von ASHRAE** genutzt
        """
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Wichtige Variablen
    # ---------------------------------------------------------
    # --- Titel ---
    st.markdown("""
        #### 🎯 Wichtige Variablen für Untersuchung der thermischen Wahrnehmung
    """
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Spalten
    # ---------------------------------------------------------
    col1, spacer, col2 = st.columns([2, 0.5, 2])
    col3, spacer, col4 = st.columns([2, 0.5, 2])

    # ---------------------------------------------------------
    # Spalte 1: Thermischer Komfort
    # ---------------------------------------------------------
    with col1:
        st.markdown("""
            ##### 1. Thermischer Komfort 
        **Sehr unkomfortabel ◄────────────────► Sehr komfortabel**  
                `  1             2            3           4           5            6   `
        """)
        st.markdown("<br>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # Spalte 2: Thermisches Empfinden
    # ---------------------------------------------------------
    with col2:
        st.markdown("""
        ##### 2. Thermisches Empfinden
        **Kalt  ◄────── Neutral ──────►  Heiß**  
        `-3    -2    -1    0    +1    +2    +3 `
        """)
        st.markdown("<br>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # Spalte 3: Thermische Akzeptanz
    # ---------------------------------------------------------
    with col3:
        st.markdown("""
        ##### 3. Thermische Akzeptanz
        ○ Nicht akzeptabel  
        ○ Akzeptabel  
        """)
        st.markdown("<br>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # Spalte 4: Thermische Präferenz
    # ---------------------------------------------------------
    with col4:
        st.markdown("""
        ##### 4. Thermische Präferenz 
        **Kühler ◄──────── Keine Änderung ────────► Wärmer**  
        `  -1                         0                         +1     `
        """)
        st.markdown("<br><br>", unsafe_allow_html=True)

#########################################################################################################
#########################################################################################################

# ---------------------------------------------------------
# TAB 3 – Team und Analysethemen
# ---------------------------------------------------------
with tab3:
    # --- Titel ---
    st.markdown("""
        #### 👥 Team und Fragestellungen von Datenanalyse und Machine Learning
    """
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Text zu Team ---
    st.markdown("""
        - Wir sind vier Personen mit **unterschiedlichen beruflichen Hintergründen**  
        - Unsere Gruppe besteht aus **zwei Data Analysts und zwei Data Scientists**
    """)
    
    # ---- CARD STYLE ----
    card_style = """
        <style>
        .team-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            transition: 0.3s;
            font-size: 1.7em;
            font-weight: 700;
        }
        .team-card:hover {
            transform: translateY(-5px);
            box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
        }
        .team-role {
            font-size: 1.5em;
            font-weight: 600;
            text-align: center;
        }
        .team-role2 {
            font-size: 1.1em;
            font-weight: 600;
            color: #1E88E5;
            text-align: center;
        }
            .team-role3 {
            font-size: 1.1em;
            font-weight: 600;
            color: #ff7f0e;
            text-align: center;
        }
        .team-task {
            font-size: 0.95em;
            opacity: 0.85;
        }
        </style>
        """
    
    st.markdown(card_style, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Spalten
    # ---------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    # ---------------------------------------------------------
    # Spalte 1: Sabrina
    # ---------------------------------------------------------
    with col1:
        st.markdown('<div class="team-card">Sabrina</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("Einfuehrung/images/psychology.png", width=180)
        st.markdown('<p style="font-size: 12px;">Bild: KI-generiert</p>', unsafe_allow_html=True)
        st.markdown('<p class="team-role2">Data Analyst</p>', unsafe_allow_html=True)
        st.markdown('<p class="team-task">Aufgaben: <br> Datenbereinigung <br> Datenanalyse</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; margin: 0; font-size: 30px;'>⬇️</h1>",
            unsafe_allow_html=True,
        )
    # ---------------------------------------------------------
    # Spalte 2: Dianela
    # ---------------------------------------------------------
    with col2:
        st.markdown('<div class="team-card">Dianela</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("Einfuehrung/images/informatic_engineering.png", width=180)
        st.markdown('<p style="font-size: 12px;">Bild: KI-generiert</p>', unsafe_allow_html=True)
        st.markdown('<p class="team-role2">Data Analyst</p>', unsafe_allow_html=True)
        st.markdown('<p class="team-task">Aufgaben: <br> Datenbank <br> Datenanalyse</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; margin: 0; font-size: 30px;'>⬇️</h1>",
            unsafe_allow_html=True,
        )
    # ---------------------------------------------------------
    # Spalte 3: Mirtha
    # ---------------------------------------------------------
    with col3:
        st.markdown('<div class="team-card">Mirtha</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("Einfuehrung/images/physicist.png", width=180)
        st.markdown('<p style="font-size: 12px;">Bild: KI-generiert</p>', unsafe_allow_html=True)
        st.markdown('<p class="team-role3">Data Scientist</p>', unsafe_allow_html=True)
        st.markdown('<p class="team-task">Aufgaben: <br> Entwicklung der GitHub- und Streamlit-Architektur <br> Datenanalyse</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; margin: 0; font-size: 30px;'>⬇️</h1>",
            unsafe_allow_html=True,
        )
    # ---------------------------------------------------------
    # Spalte 4: Daniel
    # ---------------------------------------------------------
    with col4:
        st.markdown('<div class="team-card">Daniel</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("Einfuehrung/images/civil_engineering.png", width=180)
        st.markdown('<p style="font-size: 12px;">Bild: KI-generiert</p>', unsafe_allow_html=True)
        st.markdown('<p class="team-role3">Data Scientist</p>', unsafe_allow_html=True)
        st.markdown('<p class="team-task">Aufgaben:  <br> Machine Learning  <br>  Entwicklung prädiktiver Modelle </p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; margin: 0; font-size: 30px;'>⬇️</h1>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Spalten
    # ---------------------------------------------------------
    col1, spacer, col2, spacer, col3, spacer, col4 = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1])
    col5, spacer, col6, spacer, col7, spacer, col8 = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1])
    
    # ---------------------------------------------------------
    # Spalte 1: Klima-Analyse
    # ---------------------------------------------------------
    # --- Überschrift ---
    with col1:
        st.markdown(
            """
            <h5 style="text-align: center;">
               🌍 Klima-Analyse
            </h5>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.html("<div style='height: 10px;'></div>")   
    # --- Fragestellung ---
    with col5:
        st.info("- Gibt es Unterschiede in der thermischen Wahrnehmung nach Klimatyp, Klimazone, Region und Land?")
        st.info("- Wie ausgeprägt sind diese Unterschiede je nach klimatischer/geografischer Variable?")
    # ---------------------------------------------------------
    # Spalte 2: Belüftungsart-Analyse
    # ---------------------------------------------------------
    # --- Überschrift ---
    with col2:
        st.markdown(
            """
            <h5 style="text-align: center;">
                🌀 Belüftungsart-Analyse
            </h5>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    # --- Fragestellung ---
    with col6:
        st.info("- Inwiefern beeinflusst die gewählte Belüftungsart die themische Wahrnehmung?")
        st.info("- Welche Unterschiede gibt es bei Gender und Alter innerhalb der verschiedenen Kühlungssysteme?")
    # ---------------------------------------------------------
    # Spalte 3: Physikalische Parameter-Analyse
    # ---------------------------------------------------------
    # --- Überschrift ---
    with col3:
        st.markdown(
            """
            <h5 style="text-align: center;">
                🌡️ Physikalische Parameter-Analyse
            </h5>
            """,
            unsafe_allow_html=True
        )
        st.html("<div style='height: 10px;'></div>")   
    # --- Fragestellung ---
    with col7:
        st.info("- Wie hängen die subjektiven und physikalischen thermischen Komfortvariablen miteinander zusammen?")
        st.markdown("<br>", unsafe_allow_html=True)
    # ---------------------------------------------------------
    # Spalte 4: Machine Learning
    # ---------------------------------------------------------
    # --- Überschrift ---
    with col4: 
        st.markdown(
            """
            <h5 style="text-align: center;">
            🎯Machine Learning
            </h5>
            """,
            unsafe_allow_html=True
        )
        st.html("<div style='height: 33px;'></div>")    
    # --- Fragestellung ---
    with col8:
        st.info("- Lässt sich thermisches Empfinden mit Hilfe von Machine Learning bestimmen?")
        st.info("- Gibt es andere Kenngrößen die das Wohlbefinden beeinflussen die sich vorhersagen lassen?")

#########################################################################################################
#########################################################################################################

# ---------------------------------------------------------
# TAB 4 – Tools
# ---------------------------------------------------------
with tab4:
    # ---------------------------------------------------------
    # Titel und Text
    # ---------------------------------------------------------
    st.markdown("""
        #### 🛠️ Data Science & Analytics Tools
    """
    )
    st.write("Moderne Tools für Datenanalyse, Machine Learning und interaktive Dashboards")
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Style für Felder
    # ---------------------------------------------------------
    style = """
    <style>
    .tech-card {
        background-color: #f7f7f7;
        padding: 10px;
        font-size: 1.5em;
        font-weight: 800;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.12);
        transition: 0.3s ease;
        color: #0A2540;
        border: 1px solid #e6e6e6;
        margin-bottom: 20px;
    }
    .tech-icon {
        width: 55px;
        height: 55px;
        object-fit: contain;
        margin-bottom: 10px;
    }
    .tech-title {
        font-size: 1.2em;
        font-weight: 700;
        color: #0A2540;
        margin-bottom: 6px;
    }
    .tech-desc {
        font-size: 0.95em;
        color: #333333;
        opacity: 0.85;
    }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Grid - 3 Spalten
    # ---------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    # ---------------------------------------------------------
    # COLUMN 1
    # ---------------------------------------------------------
    with col1:
        st.markdown('<div  class="tech-card">Python</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="tech-card">Explorative Datenanalyse (EDA)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="tech-card">PostgreSQL Neon</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # COLUMN 2
    # ---------------------------------------------------------
    with col2:
        st.markdown('<div class="tech-card">Power BI</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="tech-card">Jupyter Notebook</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="tech-card">NumPy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div  class="tech-card">GitHub</div>', unsafe_allow_html=True)
    # ---------------------------------------------------------
    # COLUMN 3
    # ---------------------------------------------------------
    with col3:
        st.markdown('<div class="tech-card">Pandas</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="tech-card">scikit-learn (Machine Learning)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="tech-card">Streamlit</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
