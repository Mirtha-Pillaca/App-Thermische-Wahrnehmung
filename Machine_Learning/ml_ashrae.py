import streamlit as st

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

import base64
from pathlib import Path

import joblib
import shap

import altair as alt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

import io
import threading
import time

# Seiteneinstellungen konfigurieren
st.set_page_config(
    page_title="ASHRAE Cooling Type Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Daten und Modell laden
@st.cache_resource
def load_resourcesTP():
    data_dict = joblib.load('Machine_Learning/joblib_modelle/finales_modell_thermal_preference.joblib')
    #data_dict = joblib.load('joblib_modelle/finales_modell_thermal_preference.joblib')
    return data_dict

# Daten und Modell laden
@st.cache_resource
def load_resources():
    data_dict = joblib.load('Machine_Learning/joblib_modelle/finales_klassifikations_modell_RandomForest.joblib')
    return data_dict

try:
    resources = load_resources()
    model = resources["model"]
    metrics = resources["metrics"]
    explainer = resources["explainer"]
    shap_values = resources["shap_values"]
    
    # Fehlerresistente Prüfung für den LabelEncoder
    if "LabelEncoder" in resources:
        le = resources["LabelEncoder"]
        target_names = list(le.classes_)
        has_le = True
    else:
        st.warning("⚠️ 'LabelEncoder' nicht in der Modelldatei gefunden. Nutze Standard-Klassennamen. Bitte führen Sie Ihr Trainingsskript neu aus.")
        if hasattr(model, "classes_"):
            target_names = [f"Klasse {c}" for c in model.classes_]
        else:
            target_names = ["Klasse 0", "Klasse 1", "Klasse 2"]
        has_le = False
    
    # Feature-Namen bestimmen
    if hasattr(shap_values, "feature_names") and shap_values.feature_names is not None:
        feature_names = shap_values.feature_names
    else:
        feature_names = [
            'air_temperature', 'outdoor_air_temperature', 'relative_humidity', 
            'air_speed', 'clothing_ensemble_insulation', 'metabolic_rate'
        ]
except Exception as e:
    st.error(f"Kritischer Fehler beim Laden der Modelldatei: {e}")
    st.stop()

def get_base64_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


st.title(":material/smart_toy: Machine Learning")

#tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📋Übersicht", "🧘Thermische Wahrnehmung", "🧘 Modelle thermische Wahrnehmung", "🌡️Klassifikation - Kühlungsstrategie", "👕Regression Kleidungsisolation", "🚨Anomaliebetrachtungen", "🏁Fazit"])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📋Übersicht", "🧘Thermische Wahrnehmung", "🧘 Modelle thermische Wahrnehmung", "🌡️Klassifikation - Thermische Wahrnehmung", "🌡️Klassifikation - Kühlungsstrategie", "👕Regression Kleidungsisolation", "🚨Anomaliebetrachtungen", "🏁Fazit"])

FONT_SIZE_TEXT = "24px"
FONT_SIZE_BULLET = "34px"
BULLET_COLOR = "#ff4b4b"
TEXT_COLOR = "#31333F"

SMALL_FONT_SIZE_TEXT = "18px"
SMALL_FONT_SIZE_BULLET = "24px"
SMALL_BULLET_COLOR = "#7f8c8d"
INDENT_WIDTH = "40px"

def slide_point(text):
    # var(--text-color) passt sich automatisch an Hell/Dunkel an
    return f"""
    <div style="font-size: {FONT_SIZE_TEXT}; color: var(--text-color); line-height: 1.6; margin-bottom: 12px; display: flex; align-items: center;">
        <span style="font-size: {FONT_SIZE_BULLET}; color: {BULLET_COLOR}; margin-right: 15px;">•</span>
        <div>{text}</div>
    </div>
    """

def slide_smallpoint(text):
    return f"""
    <div style="font-size: {SMALL_FONT_SIZE_TEXT}; color: var(--text-color); line-height: 1.6; margin-bottom: 8px; margin-left: {INDENT_WIDTH}; display: flex; align-items: center;">
        <span style="font-size: {SMALL_FONT_SIZE_BULLET}; color: {SMALL_BULLET_COLOR}; margin-right: 12px;">–</span>
        <div>{text}</div>
    </div>
    """

with tab1: # Übersicht

    st.subheader('📚 Bearbeitete Themengebiete')

    FONT_SIZE_TEXT = "24px"
    FONT_SIZE_BULLET = "34px"
    BULLET_COLOR = "#ff4b4b"  # Streamlit-Rot
    TEXT_COLOR = "#31333F"  # Schönes Dunkelgrau

    #st.write("\n")

    col1, col2 = st.columns([3,1], vertical_alignment="center")
    with col1:
        st.html(slide_point("Untersuchungen zur Vorhersage der thermischen Wahrnehmung (Klassifizierung)"))
        st.html(slide_smallpoint("<b>Dashboard zur Erstellung von Modellen zur Bestimmung der möglichen Genauigkeit</b>"))
        st.html(slide_smallpoint("<span style='color: #FF0000; font-weight: bold;'>Kein Dashboard zur Vorhersage von Klassen (unzureichende Genauigkeit der Modell)!</span>"))
    with col2:
        links, mitte, rechts = st.columns([1,2,1])
        with mitte:
            st.image("Machine_Learning/images/Problem_label_noise_thermal_comfort.png")

    col1, col2 = st.columns([3,1], vertical_alignment="center")
    with col1:
        st.html(slide_point("Untersuchungen zur Vorhersage der Kühlungsstrategie (Cooling Type) (Klassifizierung)"))
        st.html(slide_smallpoint("<b>Dashboard zur Vorhersage der Kühlungsstrategie</b>"))
    with col2:
        links, mitte, rechts = st.columns([1,2,1])
        with mitte:
            st.image("Machine_Learning/images/random_forest_sketch.jpg")

    col1, col2 = st.columns([3,1], vertical_alignment="center")
    with col1:
        #st.write("\n")
        st.html(slide_point("Untersuchungen zur Vorhersage des Bekleidungsisolationswertes (clothing_ensemble_insulation) (Regression)"))
        st.html(slide_smallpoint("<b>Dashboard zur Vorhersage des Bekleidungsisolationswertes</b>"))
    with col2:
        links, mitte, rechts = st.columns([1,2,1])
        with mitte:
            st.image("Machine_Learning/images/random_forest_regressor.png")

    col1, col2 = st.columns([3,1], vertical_alignment="center")
    with col1:
        #st.write("\n")
        st.html(slide_point("Anomaliebetrachtungen"))
        st.html(slide_smallpoint("Erste Untersuchungen"))
    with col2:
        links, mitte, rechts = st.columns([1,2,1])
        with mitte:
            st.image("Machine_Learning/images/DBSCAN.png")


with tab2: # subjektive Komfortbewertungen

    st.subheader("🧘 Vorhersage thermische Wahrnehmung")

    st.html(slide_point("Idee: Vorhersage der subjektiven thermischen Wahrnehmung<br> (<b>Thermischer Komfort, Thermisches Empfinden, Thermische Präferenz</b>) mit Hilfe von Featuren wie,"))
    links, mitte, rechts = st.columns(3)
    with links:
        st.html(slide_smallpoint("Innentemperatur"))
        st.html(slide_smallpoint("Außentemperatur"))
        st.html(slide_smallpoint("relative Luftfeuchtigkeit"))
    with mitte:
        st.html(slide_smallpoint("Luftgeschwindigkeit"))
        st.html(slide_smallpoint("Wärmedurchgangswiderstand der Bekleidung"))
    with rechts:
        st.html(slide_smallpoint("Mittlere Strahlungstemperatur"))
        st.html(slide_smallpoint("Aktivitätsgrad"))

    st.html("<div style='margin-bottom: 50px;'></div>")

    st.subheader("Aufgetretende Probleme")

    st.html(slide_point("Label Noise"))
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Die gleichen Bedingungen führen zu subjektiv unterschiedlichen Ergebnissen.")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Es bilden sich keine klassischen Cluster oder Möglichkeiten für einen Classifier für Unterscheidungen.")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Klassifizierungen für diese Zielgrößen erzielen eine Genauigkeit die kaum Mehrwert bietet.")

    links, rechts = st.columns([1,1])

    with links:
        st.image("Machine_Learning/images/Problem_label_noise_thermal_comfort.png", caption="Beispiel Label Noise thermal comfort mit 6 Klassen")

    with rechts:
        st.image("Machine_Learning/images/Problem_label_noise_thermal_preference.png", caption="Beispiel Label Noise thermal preference mit 3 Klassen")

    links, links_links, p1, mitte_links, p2, mitte, p3, mitte_rechts, rechts = st.columns([0.5,2,0.1,2,0.1,2,0.1,2,0.5], vertical_alignment='center')
    with links_links:
        st.image("Machine_Learning/images/verteilung_target_thermal_comfort.png", caption="Beispielverteilung - Thermal Comfort Basismodell - Klassenimbalance")
    with mitte_links:
        st.image("Machine_Learning/images/classification_report_random_forest_thermal_comfort_basis_.png", caption="Beispiel Classifcation Report - Thermal Comfort Basismodell")
    with mitte:
        st.image("Machine_Learning/images/confusion_matrix_thermal_comfort.png", caption="Beispiel Confusion matrix - Thermal Comfort")
    with mitte_rechts:
        st.html(slide_point("<center>Label Noise und Klassenimbalance machen es dem Random Forest schwer.</center>"))

    st.html("<div style='margin-bottom: 50px;'></div>")

    st.subheader("Versuchte Maßnahmen")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Kleinere Datensätze mit Filterung für homogenere Daten")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Reduzierung der Klassen des Targets")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Einsatz von TomekLink Filtern und SMOTE (\"künstliches\" Schärfen der Klassengrenzen und Data Augmentation)")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - **Aufbau eines Dashboards zum schnellen Untersuchen möglicher Modelle durch Filterung und Featureauswahl**")

    st.html(slide_point("Auf Grund der Daten lassen sich keine verlässlichen Vorhersagen bezüglich der subjektiven Komfortbewertungen treffen."))

    # Tabelle der Modelle für die thermische Wahrnehmung
    svg_path = "Machine_Learning/images/tabelle_thermal.svg"

    # Datei als Text/String einlesen
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_code = f.read()

    # Nativ in Streamlit anzeigen (wichtig: width="stretch" für die volle Breite)
    st.image(svg_code, width="stretch")

    with st.expander("🔍 Detaillierte Tabellenanalyse und Erklärungen"):
        st.markdown("- Es liegt anscheinend ein **nichtlineares Problem** vor (z.B. U-förmiger Einfluss der Temperatur auf das Befinden), dass sich mit den linearen Algorithmen nicht gut abbilden lässt.")
        st.markdown("- Auch die **Klassenungleichheit kann durch lineare Modelle schlechter berücksichtigt werden**. Sie neigen zur Dominanz der Mehrheitsklasse. Entscheidungsbäume haben es damit leichter.")
        st.markdown("- Entscheidungsbäume und kNN können die nichtlineare Situation besser abbilden, zeigen aber infolge des Label Noise keine guten Ergebnisse.")
        st.markdown("- Gerade für kNN ist Label Noise problematisch. Entscheidungsbäume neigen hingegen zu Overfitting und einem sprunghaften Wechsel zwischen den Klassen (Auswendiglernen des Trainingssets), ohne hohe Genauigkeiten im Testset zu liefern.")
        st.markdown("- Ensemblemethoden, wie RandomForest und HistGradientBoosting können das Label Noise etwas glätten und sind immun gegen Multikolinearität.")
        st.markdown("- **Die besten**, wenn auch recht schwachen, **Ergebnisse zeigen Random Forest und HistGradientBoosting**. Diese sind zudem weniger anfällig bzgl. Korrelationen und benötigen keine Skalierung der Features.")
        st.markdown("- Eine Reduzierung der Klassen verbessert die Werte der Metriken primär scheinbar, da dadurch auch die Wahrscheinlichkeiten für jede Klasse steigen. Das Problem des Label Noise bleibt.")
        st.markdown("- An angesrebter Zielwert des **macro F1-Score von ungefähr 0.7 wird nicht erreicht**. Das Modell ist nur bedingt praxistauglich.")
        st.markdown("- Für bessere Ergebnisse wären wahrscheinlich weitere subjektive Informationen der befragten Personen wichtig.")
        st.markdown("- In der ASHRAE-Norm heißt es, dass eine Klimatisierung dann als erfolgreich und normgerecht gilt, wenn mindestens 80% der Raumnutzer zufrieden sind.")


    st.html("<div style='height: 140px;'></div>")    

    links, rechts = st.columns([1,1])
    with links:
        st.image("Machine_Learning/images/VergleichModelle_thermal_comfort_F1.png", caption="Beispielvergleich der Ergebnisse unterschiedlicher Algorithmen- Thermal comfort")    

    with rechts:
        st.image("Machine_Learning/images/VergleichModelle_thermal_preference_F1.png", caption="Beispielvergleich der Ergebnisse unterschiedlicher Algorithmen- Thermal Preference")    


with tab3: # modelle Komfort

    st.subheader("🧘 Vorhersage thermische Wahrnehmung (Modellbildung)- Random Forest Classifier")

    # cachen der csv-Daten
    @st.cache_data
    def load_data():

        df_loaded = pd.read_csv("Daten/db_bereinigt_final.csv")
            
        return df_loaded

    raw_df = load_data()

    # === BEREINIGUNG DER ZIELVARIABLE (TARGET) ===
    raw_df['thermal_preference'] = raw_df['thermal_preference'].astype(str).str.strip()
    raw_df['thermal_preference'] = raw_df['thermal_preference'].replace(['unknown', 'Unknown', 'UNKNOWN', 'nan', 'None'], np.nan)
    raw_df['cooling_type'] = raw_df['cooling_type'].replace(['unknown', 'Unknown', 'UNKNOWN', 'nan', 'None'], np.nan)
    df = raw_df.dropna(subset=['thermal_preference']).copy()

    # df sind Anzhal Datensätze mit thermal_preference, im Basismodell 69825 + 15063 = 84888

    # ausklappbarer Bereich im Hauptfenster, standardmäßig offen
    with st.expander("⚙️ Modellkonfiguration & Hyperparameter", expanded=True):

        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.subheader("📋 Features")
            possible_features = ['air_temperature', 'relative_humidity', 'air_speed', 'metabolic_rate', 'clothing_ensemble_insulation', 'radiant_temperature']
            
            default_features = ['air_temperature', 'relative_humidity', 'air_speed', 'metabolic_rate', 'clothing_ensemble_insulation']
            
            selected_features = []
            for feature in possible_features:
                is_checked = feature in default_features
                
                if st.checkbox(feature, value=is_checked, key=f"feat_{feature}"):
                    selected_features.append(feature)

        # with row1_col2:
        #     st.subheader("🧽 Daten-Handling")
        #     impute_strategy = st.radio(
        #         "Strategie für Fehlwerte:",
        #         options=["Zeilen mit Fehlwerten löschen (dropna)", "Mit Median auffüllen (Imputer)"]
        #     )

        impute_strategy = "Zeilen mit Fehlwerten löschen (dropna)"

        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        
        with row2_col1:
            # NEU: Filter für Cooling Type über Checkboxen
            st.subheader("❄️ Kühlungstypfilter\n\n (Cooling Type)")
            available_coolings = sorted(list(df['cooling_type'].dropna().unique()))
            selected_coolings = []
            for cooling in available_coolings:
                # Standardmäßig alle Kühlungstypen auswählen
                if st.checkbox(f"{cooling}", value=True, key=f"cool_{cooling}"):
                    selected_coolings.append(cooling)
                    
        with row2_col2:
            st.subheader("📅 Jahreszeitfilter\n\n (Seasons)")
            available_seasons = sorted(list(df['season'].dropna().unique()))
            selected_seasons = []
            for season in available_seasons:
                if st.checkbox(f"{season}", value=True, key=f"seas_{season}"):
                    selected_seasons.append(season)

        with row2_col3:
            st.subheader("🌍 Klimafilter\n\n (Climate)")
            available_climates = sorted(list(df['climate_zone'].dropna().unique()))
            selected_climates = []
            for climate in available_climates:
                if st.checkbox(f"{climate}", value=True, key=f"clim_{climate}"):
                    selected_climates.append(climate)

        with row2_col4:
            st.subheader("🏢 Gebäudetypfilter\n\n (Building Type)")
            available_buildings = sorted(list(df['building_type'].dropna().unique()))
            selected_buildings = []
            for building in available_buildings:
                if st.checkbox(f"{building}", value=True, key=f"build_{building}"):
                    selected_buildings.append(building)

        # Trennlinie für die zweite Reihe im Expander
        st.write("---")
        
        # Eine Trennlinie innerhalb des Expanders für die Hyperparameter-Auswahl
    #    st.write("---")
        st.subheader("🌲 Random Forest Hyperparameter (Grid)")
    #    param_col1, param_col2 = st.columns(2)

        such_modus = st.selectbox(
            "GridSearch / RandomSearch-Intensität (online-Version beeinhaltet nur schnelle Suche!):",
            #["Schnelle Suche (Live-Demo)", "Normale Suche (mit RandomSearch)", "Intensive Suche (mit GridSearch)"],
            ["Schnelle Suche (Live-Demo)"],
        )    

        ######################################################
        ############ GridSearchParameter #####################
        ######################################################

        if such_modus == "Schnelle Suche (Live-Demo)":
            # Extrem schlank: Fokus auf die 3 wichtigsten Hebel, minimale Listen
            param = {
            "classifier__max_depth": [6],
            "classifier__n_estimators": [50],
            "classifier__min_samples_leaf": [2],
            #"classifier__min_samples_split": [70],
            "classifier__max_features": ["sqrt"],
            #"classifier__class_weight": ["balanced"],  # bereits im Estimator definert
                }

            cv_folds = 3  # Weniger Folds sparen massiv Zeit bei großen Datensätzen
            max_kombinationen = 2
            use_random_search = True  # Bei so wenigen Kombinationen ist GridSearchCV schneller

        elif such_modus == "Normale Suche (mit RandomSearch)":  # kann auch mal 20 min dauern
            param = {
            "classifier__max_depth": [6, 9, 12],
            "classifier__n_estimators": [100],
            "classifier__min_samples_leaf": [20, 50],
            "classifier__min_samples_split": [50],
            "classifier__max_features": ["sqrt"],
            #"classifier__class_weight": ["balanced"],
                }

            cv_folds = 5
            max_kombinationen = 6
            use_random_search = True

        else:  # Intensive Suche (Nicht für Live-Vortrag geeignet)
            param = {
            'classifier__n_estimators': [100, 200],
            # Maximale Baumtiefe (Der gesuchte Sweet-Spot zwischen 3 und 16)
            'classifier__max_depth': [6, 9, 12],
            # Mindestanzahl an Datenpunkten pro Endblatt (Schutz vor Rauschen/Overfitting)
            'classifier__min_samples_leaf': [10, 30, 60],
            # Mindestanzahl an Datenpunkten, um einen Knoten überhaupt zu teilen
            'classifier__min_samples_split': [25, 70],
            # Feature-Begrenzung pro Split ('sqrt' nutzt ca. 2 von 5 Features, None nutzt alle 5)
            'classifier__max_features': ['sqrt', None]
                }

            cv_folds = 5
            max_kombinationen = 0
            use_random_search = False


    # Fehlermeldungen, Abfangen ungültige Auswahlen
    if not selected_seasons:
        st.error("Bitte wähle mindestens eine Jahreszeit im Zeitfilter aus.")
        st.stop()

    if not selected_climates:
        st.error("Bitte wähle mindestens eine Klimazone im Klimafilter aus.")
        st.stop()

    if not selected_coolings:
        st.error("Bitte wähle mindestens einen Kühlungstyp im Kühlungsfilter aus.")
        st.stop()

    if not selected_buildings:
        st.error("Bitte wähle mindestens einen Gebäudetyp im Gebäudefilter aus.")
        st.stop()

    if not selected_features:
        st.error("Bitte wähle mindestens ein Feature über die Checkboxen aus.")
        st.stop()

    # Filter für Zeilen (Saison UND Klima UND Kühlungstyp)
    df_filtered_rows = df[
        (df['season'].isin(selected_seasons)) & 
        (df['climate_zone'].isin(selected_climates)) &
        (df['cooling_type'].isin(selected_coolings)) &
        (df['building_type'].isin(selected_buildings))
    ].copy()

    # Nur noch die benötigten Feature-Spalten + Target behalten
    df_filtered = df_filtered_rows[selected_features + ['thermal_preference']].copy()

    # STATISTIKEN BERECHNEN
    total_missing_cells = df_filtered[selected_features].isna().sum().sum()
    rows_with_missing_values = df_filtered[selected_features].isna().any(axis=1).sum()

    # Strategie für NaNs anwenden - alt
    if impute_strategy == "Zeilen mit Fehlwerten löschen (dropna)":
        df_filtered = df_filtered.dropna()

    X = df_filtered[selected_features]
    y = df_filtered['thermal_preference']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Dynamische Datensatzstatistiken
    st.subheader("📋 Datensatz-Statistiken")
    stat_col1, stat_col2, stat_col3 = st.columns([1, 1, 2])

    with stat_col1:
        st.metric(label="Verfügbare Zeilen (nach Filterung)", value=f"{len(df_filtered):,}")
    with stat_col2:
        st.metric(label="Größe Trainingsset (80%)", value=f"{len(X_train):,}")
        st.metric(label="Größe Testset (20%)", value=f"{len(X_test):,}")
    with stat_col3:
        st.metric(
            label="Fehlende Daten im gefilterten Set", 
            value=f"unvollst. Zeilen: {rows_with_missing_values:,} \n\nfehlende Werte: {total_missing_cells:,}"
        )
        if impute_strategy == "Zeilen mit Fehlwerten löschen (dropna)":
            st.caption("🔴 Die betroffenen Zeilen wurden aus dem Modell entfernt.")
        else:
            st.caption("🟢 Lücken werden im Trainingsverlauf durch den Median ersetzt.")

    st.write("---")

    #########################################################################################################
    # Training & GridSearch 
    #########################################################################################################
    @st.cache_resource
    #def train_model(X_tr, y_tr, features, estimators, depths, strategy, seasons, climates, coolings, max_kombinationen, cv_folds):
    def train_model(X_tr, y_tr, features, param, strategy, seasons, climates, coolings, buildings, max_kombinationen, cv_folds):
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # in der Regel ohnehin fehlende Werte entfernt
            ('scaler', PowerTransformer()),   # RandomForest, also nicht von Bedeutung
            ('classifier', RandomForestClassifier(class_weight="balanced", random_state=42))    # fixe Parameter setzen!!!
        ])
        

        #########################################################
        ############## Herzstück von GridSearch #################
        #########################################################
        
        # Cross-Validation mit 3 Folds (Durchgängen), Bewertungskriterium f1_macro, Overfitting wird nicht berücksichtigt / ermittelt
        # Durch Kreuzvalidierung wird die Gefahr auf Overfitting jedoch stark reduziert!!!
        
        if (use_random_search):
            param_distributions = param
            grid_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, cv=cv_folds, n_iter=max_kombinationen, scoring='f1_macro', n_jobs=-1)
            st.write(f"RandomSearch durchgeführt!")
        else:
            #param_grid = param
            grid_search = GridSearchCV(pipeline, param_grid=param, cv=cv_folds, scoring='f1_macro', n_jobs=-1)
            st.write(f"GridSearch durchgeführt!")
        
        
        
        grid_search.fit(X_tr, y_tr)
        
        return grid_search.best_estimator_, grid_search.best_params_
    # ENDE train_model

    # Button zum Starten des Trainings
    if st.button("🚀 Modell trainieren & validieren"):

        # Variablen für  initialisieren Zeitmessung
        start_time = time.time()

        if (use_random_search):
            searchtext = 'RandomSearch läuft... Bitte warten...'
        else:
            searchtext = 'GridSearch läuft... Bitte warten...'
        
        with st.status(searchtext, expanded=True) as status_box:

            best_pipeline, best_params = train_model(   ######################################################### Aufruf RANDOM FOREST FUNCTION ##################
                X_train, y_train, tuple(selected_features), param, #tuple(n_estimators_options), tuple(max_depth_options), 
                impute_strategy, tuple(selected_seasons), tuple(selected_climates), tuple(selected_coolings), tuple(selected_buildings), max_kombinationen, cv_folds
            )
        

            gesamtdauer = time.time() - start_time

        st.session_state['model_thermal'] = best_pipeline
        st.session_state['best_params'] = best_params
        st.success(f"Training erfolgreich abgeschlossen! Gesamte Rechenzeit: **{gesamtdauer:.2f} Sekunden**")

    # 2. Vorhersagen für die Metriken generieren
        y_pred = best_pipeline.predict(X_test)
        unique_labels = sorted(list(y_test.unique()))
        
        # Textbasierten Classification Report als String generieren
        class_report_str = classification_report(y_test, y_pred, labels=unique_labels)
        
        # SHAP-Werte exakt hier EINMAL berechnen
        transformed_X_test = best_pipeline.named_steps['scaler'].transform(
            best_pipeline.named_steps['imputer'].transform(X_test)
        )
        transformed_df = pd.DataFrame(transformed_X_test, columns=selected_features)
        rf_model = best_pipeline.named_steps['classifier']
        
        shap_sample = transformed_df.sample(min(100, len(transformed_df)), random_state=42)
        explainerTH = shap.TreeExplainer(rf_model)
        shap_valuesTH = explainerTH.shap_values(shap_sample)
        actual_class_names = list(rf_model.classes_)
        
        # Das KOMPLETTE EXPERIMENT-PAKET als Dictionary
        experiment_data = {
            'pipeline': best_pipeline,
            'best_params': best_params,
            'y_test': y_test,
            'y_pred': y_pred,
            'unique_labels': unique_labels,
            'classification_report': class_report_str, # Der gespeicherte Report
            'shap_values': shap_valuesTH,
            'shap_sample': shap_sample,
            'shap_class_names': actual_class_names
        }
        
        # Session-State für die UI-Anzeige sichern
        st.session_state['experiment'] = experiment_data
        
        # MODELL + ERGEBNISSE LOKAL IM ORDNER SPEICHERN ===
        local_filename = "Machine_Learning/joblib_modelle/ashrae_thermal_classification_model.joblib"
        joblib.dump(experiment_data, local_filename)
        
        st.info(f"💾 Das Gesamtpaket wurde als **'{local_filename}'** im Projektordner gespeichert.")

    if st.button("🧹 Cache leeren"):
    
        st.cache_data.clear()

        st.cache_resource.clear()

    # Ergebnisse anzeigen
    if 'model_thermal' in st.session_state:
        model_thermal = st.session_state['model_thermal']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Modell-Metriken")
            st.write(f"**Beste Parameter:** {st.session_state['best_params']}")
            
            y_pred = model_thermal.predict(X_test)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            st.metric(label="F1-Score (Macro)", value=f"{f1_macro:.2f}")
            
            unique_labels = sorted(list(y_test.unique()))
            st.write("**Classification Report:**")
            st.code(classification_report(y_test, y_pred, labels=unique_labels))
            
            st.write("---") # Kleine Trennlinie für die Optik

            if 'experiment' in st.session_state:
                exp = st.session_state['experiment']
                # Modell speichern
                # Erstellt ein virtuelles Dateiobjekt im Arbeitsspeicher
                st.write("**Komplettes Experiment-Paket exportieren:**")
                buffer = io.BytesIO()
                joblib.dump(exp, buffer)
                joblib_bytes = buffer.getvalue()
                
                st.download_button(
                    label="📥 Gesamtpaket (.joblib) für Browser herunterladen",
                    data=joblib_bytes,
                    file_name="Machine_Learning/joblib_modelle/ashrae_thermal_classification_bundle.joblib",
                    mime="application/octet-stream",
                    help="Herunterladen von Pipeline, Parametern, Testdaten, Classification Report und SHAP-Werten."
                )

            with st.spinner("Generiere Pairplot... Bitte warten..."):
                fig_pp = plt.figure(figsize=(12, 10))
                plot_df = df_filtered.dropna()

                if len(plot_df) > 5000:
                    plot_df = plot_df.sample(5000, random_state=42)
                    st.caption("ℹ️ Hinweis: Der Datensatz ist sehr groß. Es wird eine repräsentative Stichprobe von 5.000 Zeilen visualisiert.")

                sns.pairplot(
                plot_df, 
                    hue='thermal_preference', 
                    palette='coolwarm', # Farbpalette für thermischen Komfort
                    corner=True # Verhindert doppelte Plots in der oberen Hälfte für bessere Übersicht
                )   

                st.pyplot(plt.gcf()) # plt.gcf() holt sich die aktuelle Seaborn-Grafik
                plt.close('all')

        with col2:
            st.subheader("🧬 SHAP Analyse")
            
            # Daten transformieren, damit SHAP mit skalierten Daten arbeitet
            transformed_X_test = model_thermal.named_steps['scaler'].transform(
                model_thermal.named_steps['imputer'].transform(X_test)
            )
            transformed_df = pd.DataFrame(transformed_X_test, columns=selected_features)
            
            # Den Classifier aus der Pipeline extrahieren
            rf_model = model_thermal.named_steps['classifier']
            
            # Die exakten Textklassennamen direkt aus dem Modell auslesen
            actual_class_names = list(rf_model.classes_)
            
            with st.spinner("Generiere SHAP-Analyse... Bitte warten..."):
                # Stichprobe ziehen, um Abstürze bei großen Testsets zu verhindern
                shap_sample = transformed_df.sample(min(400, len(transformed_df)), random_state=42) # nur 400 samples!
                explainerTH = shap.TreeExplainer(rf_model)
                shap_valuesTH = explainerTH.shap_values(shap_sample)
                
                # SHAP Summary Plot erzeugen
                fig, ax = plt.subplots()
                
                # Übergabe von class_names sorgt für die korrekten Namen in der Legende!
                shap.summary_plot(
                    shap_valuesTH, 
                    shap_sample, 
                    plot_type="bar", 
                    class_names=actual_class_names, # Namen werden gemappt!
                    show=False
                )

                plt.xlabel("Einfluss auf die Modellvorhersage (Durchschnitt)")

                st.pyplot(fig)

            st.write("\n\n")

            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(6, 5)) # Größe optional anpassbar

            ConfusionMatrixDisplay.from_predictions(
                y_test, 
                y_pred, 
                labels=unique_labels, # garantiert gleiche Reihenfolge wie im Report
                cmap="Blues", 
                ax=ax
            )

            plt.title("Thermal Preference")
            plt.tight_layout() # Verhindert abgeschnittene Labels am Rand
            st.pyplot(fig)


with tab4: # classification thermal preference

    try:
        resources = load_resourcesTP()
        
        # 1. Pipeline / Modell laden
        modelTP = resources["pipeline"]
        
        # 2. Metriken-Brücke: Altes Skript sucht 'metrics', neues liefert 'classification_report'
        if "metrics" in resources:
            metricsTP = resources["metrics"]
        elif "classification_report" in resources:
            metricsTP = resources["classification_report"]
        else:
            metricsTP = "Keine Metriken verfügbar."

        
        # Klassennamenbrücke

        if "shap_class_names" in resources:
            target_namesTP = resources["shap_class_names"]
            has_le = True  # Trickst das alte Skript aus, da Namen direkt da sind
        elif "unique_labels" in resources:
            target_namesTP = [f"Klasse {c}" for c in resources["unique_labels"]]
            has_le = False
        elif hasattr(modelTP, "classes_"):
            target_namesTP = [f"Klasse {c}" for c in modelTP.classes_]
            has_le = False
        else:
            target_namesTP = ["Klasse 0", "Klasse 1", "Klasse 2"]
            has_le = False
        

    except Exception as e:
        st.error(f"Kritischer Fehler beim Laden der Modelldatei: {e}")
        st.stop()


    # Dashboard Titel
    st.title("🌡️ Vorhersage Thermal Preference")
    st.write("(keine hohe Genauigkeit!!!)")

    #tab1, tab2, tab3 = st.tabs(["🔮 Livevorhersage & SHAP", "📈 Modellperformance", "⚙️ Modellaufbau"])
    tab1, tab2 = st.tabs(["🔮 Livevorhersage & SHAP", "📈 Modellperformance"])

    # --- TAB 1: ECHTZEIT VORHERSAGE ---
    with tab1:

        # Layout in zwei Hauptbereiche unterteilen
        col_sidebar, col_main = st.columns([1,3])

        # 2. Sidebar für Benutzereingaben (Schieberegler)
        with col_sidebar:
            st.header("🎛️ Featureeingabe")
            
            air_tempTP = st.slider("Innentemperatur (air_temperature) [°C]", 10.0, 40.0, 23.0, step=0.1, format="%0.1f", key="air_temp_TP")
            out_tempTP = st.slider("Außentemperatur (outdoor_air_temperature) [°C]", -30.0, 45.0, 20.0, step=0.1, format="%0.1f", key="out_temp_TP")
            rel_humTP = st.slider("Relative Luftfeuchtigkeit (relative_humidity) [%]", 0.0, 100.0, 50.0, step=0.1, format="%0.1f", key="rel_humTP_TP")
            air_speedTP = st.slider("Luftgeschwindigkeit (air_speed) [m/s]", 0.0, 4.0, 0.1, step=0.01, format="%0.2f", key="air_speed_TP")
            cloTP = st.slider("Bekleidungsisolierung (clothing_ensemble_insulation) [clo]", 0.0, 3.0, 0.6, step=0.01, format="%0.2f", key="clo_TP")
            metTP = st.slider("Metabolische Rate (metabolic_rate) [met]", 0.5, 4.0, 1.2, step=0.1, format="%0.1f", key="met_TP")
            radiant_tempTP = st.slider("Strahlungstemperatur (radient temperature) [tr]", 10.0, 40.0, 23.0, step=0.1, format="%0.1f", key="radient_temp_TP")

            input_dataTP = pd.DataFrame([{
                'air_temperature': air_tempTP,
                'outdoor_air_temperature': out_tempTP,
                'relative_humidity': rel_humTP,
                'air_speed': air_speedTP,
                'clothing_ensemble_insulation': cloTP,
                'metabolic_rate': metTP,
                'radiant_temperature': radiant_tempTP
            }])

            input_dataTP = input_dataTP[modelTP.feature_names_in_]

        # 3. Hauptbereich für Vorhersagen und Analysen
        with col_main:
            #tab1, tab2, tab3 = st.tabs(["🔮 Vorhersage", "📊 SHAP Analyse", "📈 Modellperformance"])

                # Vorhersagen berechnen
                num_predictionTP = modelTP.predict(input_dataTP)
                pred_probaTP = modelTP.predict_proba(input_dataTP)
                
                # WICHTIG: Wahrscheinlichkeiten absolut flach klopfen (1D-Array erwingen)
                # Das verhindert den "All arrays must be of the same length" Fehler komplett.
                probabilitiesTP = np.array(pred_probaTP).flatten()
                
                text_predictionTP = str(num_predictionTP)
                
                # Metrik anzeigen
                st.metric(label="🎯 Vorhergesagte thermische Präferenz", value=str(text_predictionTP[2:-2]))

                # DataFrame absolut sicher aufbauen
                st.markdown("**Klassenwahrscheinlichkeiten:** (ohne Verwendung von CalibratedClassifierCV)")
                
                # Falls die Längen im Extremfall immer noch nicht passen, passen wir die target_names dynamisch an
                display_labelsTP = [str(c) for c in target_namesTP][:len(probabilitiesTP)]
                
                proba_df_TP = pd.DataFrame({
                    "Klasse": display_labelsTP,
                    "Wahrscheinlichkeit": probabilitiesTP
                })
                
                proba_df_TP["Wahrscheinlichkeit"] = proba_df_TP["Wahrscheinlichkeit"].round(2)

                # 1. Basis-Chart definieren
                base = alt.Chart(proba_df_TP).encode(
                    x=alt.X("Klasse:N", sort=None).axis(
                        #labelFontSize=14,     # Schriftgröße der X-Achsen-Werte
                        labelFontWeight="bold", # Fett gedruckt
                        #titleFontSize=16,     # Schriftgröße des X-Achsen-Titels
                        titleFontWeight="bold"
                    ), 
                    y=alt.Y("Wahrscheinlichkeit:Q", scale=alt.Scale(domain=[0, 1.1])).axis(
                        #labelFontSize=14,     # Schriftgröße der Y-Achsen-Werte
                        labelFontWeight="bold",
                        #titleFontSize=16,     # Schriftgröße des Y-Achsen-Titels
                        titleFontWeight="bold"
                    )
                )

                # 2. Die Balken erstellen
                bars = base.mark_bar()

                text_balken = base.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-5,
                    #fontSize=14,
                    fontWeight="bold"
                ).encode(
                    text=alt.Text("Wahrscheinlichkeit:Q", format=".1%")
                )


                chart = alt.layer(bars, text_balken).properties(width="container")
                st.altair_chart(chart, use_container_width=True)




    # --- TAB 2: MODELLPERFORMANCE ---
    with tab2:
        st.subheader("📈 Modellperformance & -metriken")
        
        y_testTP = resources["y_test"]
        y_predTP = resources["y_pred"]    
        f1_macro_testTP = f1_score(y_testTP, y_predTP, average='macro')

        y_trainTP = resources["y_train"]
        y_pred_trainTP = resources["y_pred_train"]    
        f1_macro_trainTP = f1_score(y_trainTP, y_pred_trainTP, average='macro')

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(label="F1-Score (Train)", value=f"{f1_macro_trainTP:.2f}")
        with col_metric2:
            st.metric(label="F1-Score (Test)", value=f"{f1_macro_testTP:.2f}")

                
        st.markdown("---")

        performance_left, performance_right = st.columns(2, vertical_alignment="center")

        with performance_left:

            st.markdown("**Classification Report:**")

            st.code(classification_report(y_testTP, y_predTP, target_names=target_namesTP))



        with performance_right:

            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(6, 5))

            # Ermittle die tatsächlichen numerischen Klassen, die in den Daten stecken (z.B.)
            import numpy as np
            unique_numeric_labels = sorted(list(set(y_testTP)))
            
            # Hole die exakt passenden Textbeschriftungen aus deiner target_names Liste
            #display_labels_filtered = [target_names[i] for i in unique_numeric_labels]

            from sklearn.metrics import ConfusionMatrixDisplay
            ConfusionMatrixDisplay.from_predictions(
                y_testTP, 
                y_predTP, 
                labels=unique_numeric_labels,       # IDs für die mathematische Zuordnung (0, 1, 2)
                #display_labels=display_labels_filtered, # Textnamen für die visuelle Achsenbeschriftung!
                cmap="Blues", 
                ax=ax
            )

            plt.title("Thermal preference")
            plt.tight_layout()
            ax.set_xticklabels(
                ax.get_xticklabels(), 
                rotation=90, 
            #    ha="right"
            )
            st.pyplot(fig)


with tab5: # classification cooling type

    # Dashboard Titel
    st.subheader("🌡️ Vorhersage Kühlungsstrategie (Cooling Type)")

    # Hauptreiter
    tab1, tab2, tab3 = st.tabs(["🔮 Livevorhersage & SHAP", "📈 Modellperformance", "⚙️ Modellaufbau"])

    # --- TAB 1: ECHTZEIT VORHERSAGE ---
    with tab1:

        # Layout in zwei Hauptbereiche unterteilen
        col_sidebar, col_main = st.columns([1,3])

        # Benutzereingaben (Schieberegler)
        with col_sidebar:
            st.header("🎛️ Featureeingabe")
            
            air_temp = st.slider("Innentemperatur (air_temperature) [°C]", 10.0, 40.0, 22.0, step=0.1, format="%0.1f")
            out_temp = st.slider("Außentemperatur (outdoor_air_temperature) [°C]", -30.0, 45.0, 10.0, step=0.1, format="%0.1f")
            rel_hum = st.slider("Relative Luftfeuchtigkeit (relative_humidity) [%]", 0.0, 100.0, 30.0, step=0.1, format="%0.1f")
            air_speed = st.slider("Luftgeschwindigkeit (air_speed) [m/s]", 0.0, 4.0, 0.1, step=0.01, format="%0.2f")
            clo = st.slider("Bekleidungsisolierung (clothing_ensemble_insulation) [clo]", 0.0, 3.0, 1.0, step=0.01, format="%0.2f")
            met = st.slider("Metabolische Rate (metabolic_rate) [met]", 0.5, 4.0, 1.0, step=0.1, format="%0.1f")

            input_data = pd.DataFrame([{
                'air_temperature': air_temp,
                'outdoor_air_temperature': out_temp,
                'relative_humidity': rel_hum,
                'air_speed': air_speed,
                'clothing_ensemble_insulation': clo,
                'metabolic_rate': met
            }])
            input_data = input_data[feature_names]

            st.markdown("""<small><strong>CLO-Werte</strong>:
                <ul>
                <li><strong>0.1:</strong> Kurze Hose, T-Shirt / ärmelloses Oberteil (Sehr leichte Sommerkleidung)"</li>
                <li><strong>0.3:</strong> Leichte Shorts und kurzärmeliges Hemd"</li>
                <li><strong>0.5:</strong> Kniegellange Schürze/Rock, leichtes Hemd (Typische leichte Sommerbekleidung)"</li>
                <li><strong>0.7:</strong> Leichte Hose, langärmeliges Hemd"</li>
                <li><strong>1.0:</strong> Anzug (Hose, Hemd, Sakko) oder Pullover mit Hose (Typische Bürokleidung im Winter)"</li>
                <li><strong>1.2:</strong> Anzug mit Weste / zusätzlichem leichten Unterhemd"</li>
                <li><strong>1.5:</strong> Schwere Winterkleidung (Hose, Hemd, dicker Pullover, schwere Jacke/Mantel)"</li>
                <li><strong>2.0:</strong> Arktische Spezialkleidung / Extremkleidung"</li>
                </ul>
                </small>
                 """, unsafe_allow_html=True)

            st.markdown("""<small><strong>met-Werte</strong>:
                <ul>
                  <li><strong>0.7 met:</strong> Schlafen / Liegen</li>
                  <li><strong>1.0 met:</strong> Sitzende Tätigkeit (z. B. Büroarbeit)</li>
                  <li><strong>1.2 met:</strong> Stehende, leichte Aktivität (z. B. im Labor oder Verkauf)</li>
                  <li><strong>2.0 met:</strong> Gehen (ca. 3 km/h)</li>
                  <li><strong>3.0 met bis 4.0 met:</strong> Schwere körperliche Arbeit oder Sport</li>
                </ul>
                </small>
                 """, unsafe_allow_html=True)

        # Hauptbereich für Vorhersagen und Analysen
        with col_main:
                
                # Vorhersagen berechnen
                num_prediction = model.predict(input_data)
                pred_proba = model.predict_proba(input_data)
                
                # WICHTIG: Wahrscheinlichkeiten absolut flach klopfen (1D-Array erwingen)
                # Das verhindert den "All arrays must be of the same length" Fehler komplett.
                probabilities = np.array(pred_proba).flatten()
                
                # Rücktransformation des vorhergesagten Wertes
                if has_le:
                    # Falls prediction ein Array ist, den ersten Wert nehmen
                    val_to_pred = num_prediction[0] if hasattr(num_prediction, "__len__") else num_prediction
                    text_prediction = le.inverse_transform([val_to_pred])[0]
                else:
                    text_prediction = str(num_prediction)
                
                # Metrik anzeigen
                st.metric(label="🎯 Vorhergesagte Kühlungsstrategie (Cooling Type)", value=str(text_prediction))

                # DataFrame sicher aufbauen
                st.markdown("**Klassenwahrscheinlichkeiten:** (ohne Verwendung von CalibratedClassifierCV)")
                
                display_labels = [str(c) for c in target_names][:len(probabilities)]
                
                proba_df = pd.DataFrame({
                    "Klasse": display_labels,
                    "Wahrscheinlichkeit": probabilities
                })
                
                proba_df["Wahrscheinlichkeit"] = proba_df["Wahrscheinlichkeit"].round(2)

                # Basis-Chart definieren
                base = alt.Chart(proba_df).encode(
                    x=alt.X("Klasse:N", sort=None).axis(
                        labelFontWeight="bold", # Fett gedruckt
                        titleFontWeight="bold"
                    ), 
                    y=alt.Y("Wahrscheinlichkeit:Q", scale=alt.Scale(domain=[0, 1.1])).axis(
                        labelFontWeight="bold",
                        titleFontWeight="bold"
                    )
                )

                # Die Balken erstellen
                bars = base.mark_bar()

                text_balken = base.mark_text(
                    align="center",
                    baseline="bottom",
                    dy=-5,
                    fontWeight="bold"
                ).encode(
                    text=alt.Text("Wahrscheinlichkeit:Q", format=".1%")
                )

                chart = alt.layer(bars, text_balken).properties(width="container")
                st.altair_chart(chart, use_container_width=True)

            # --- SHAP ANALYSE ---
                try:
                    # ==========================================
                    # DIAGRAMM 2: LOKALER WATERFALL-PLOT (Fehlerfreie Prozent-Kalibrierung)
                    # ==========================================
                    st.markdown("### 🔍 Live-SHAP-Wasserfalldiagramm")
                    
                    # Numerischen Index der echten Live-Vorhersage (0, 1 oder 2)
                    if hasattr(num_prediction, "item"):
                        predicted_class_idx = int(num_prediction.item())
                    elif hasattr(num_prediction, "__len__") and len(num_prediction) > 0:
                        # Fallback für Listen oder mehrdimensionale Arrays (nimmt das erste Element)
                        val = num_prediction[0]
                        predicted_class_idx = int(val.item()) if hasattr(val, "item") else int(val)
                    else:
                        predicted_class_idx = int(num_prediction)
                    
                    predicted_class_name = target_names[predicted_class_idx]
                    
                    # ZWINGENDE SYNCHRONISATION: Wenn sich die Vorhersage geändert hat, 
                    # überschreiben wir den Session State der Selectbox manuell.
                    if "last_predicted_class" not in st.session_state:
                        st.session_state["last_predicted_class"] = predicted_class_name
                        st.session_state["sb_local_shap"] = predicted_class_name
                        
                    if st.session_state["last_predicted_class"] != predicted_class_name:
                        st.session_state["last_predicted_class"] = predicted_class_name
                        st.session_state["sb_local_shap"] = predicted_class_name  # Setzt die Box hart zurück

                    # Selectbox greift nun stabil auf den manipulierten Session State zu
                    selected_class_local = st.selectbox(
                        "Kühlungsstrategie für die Erklärung der Einzelvorhersage (Waterfall):",
                        options=target_names,
                        key="sb_local_shap"  # Über diesen Key mit State verknüpft
                    )
                    
                    class_idx_local = target_names.index(selected_class_local)

                    # Rohe SHAP-Werte für das aktuelle Slider-Beispiel berechnen
                    single_shap = explainer(input_data)

                    fig_local, ax_local = plt.subplots(figsize=(8, 4))

                    if len(single_shap.shape) == 3: 
                        # =====================================================================
                        # MULTI-KLASSEN-FALL (Ihre Kalibrierung)
                        # =====================================================================
                        true_end_prob = probabilities[class_idx_local]
                        
                        if hasattr(single_shap.base_values, "ndim") and single_shap.base_values.ndim > 1:
                            raw_base_value = single_shap.base_values[0, class_idx_local]
                        elif isinstance(single_shap.base_values, (list, np.ndarray)) and len(single_shap.base_values) > class_idx_local:
                            raw_base_value = single_shap.base_values[class_idx_local]
                        else:
                            raw_base_value = single_shap.base_values
                        
                        raw_values = single_shap.values[0, :, class_idx_local]
                        raw_sum = raw_values.sum()
                        
                        if abs(raw_sum) > 1e-5:
                            scaling_factor = (true_end_prob - raw_base_value) / raw_sum
                            calibrated_values = raw_values * scaling_factor
                        else:
                            calibrated_values = raw_values
                            
                        plot_values = calibrated_values
                        base_val_for_plot = raw_base_value
                        prob_for_plot = true_end_prob

                    else:
                        # =====================================================================
                        # BINÄRER FALLBACK (Sicher gegen den "0-dimensional array"-Fehler)
                        # =====================================================================
                        # Werte für die erste Instanz extrahieren
                        if len(single_shap.values.shape) > 1:
                            plot_values = single_shap.values[0]
                        else:
                            plot_values = single_shap.values
                            
                        # Basiswert auslesen und zwingend in Python-Skalar konvertieren
                        if hasattr(single_shap.base_values, "item"):
                            base_val_for_plot = single_shap.base_values.item()
                        elif isinstance(single_shap.base_values, (list, np.ndarray)) and len(single_shap.base_values) > 0:
                            base_val_for_plot = single_shap.base_values[0]
                        else:
                            base_val_for_plot = single_shap.base_values
                            
                        # Bei binären Modellen nutzen wir oft direkt die Wahrscheinlichkeit der positiven Klasse
                        prob_for_plot = probabilities[1] if hasattr(probabilities, "__len__") and len(probabilities) > 1 else probabilities

                    # Daten sortieren (Wichtigste Merkmale nach oben)
                    indices = np.argsort(np.abs(plot_values))[::-1]
                    
                    # Begrenzung auf Top 10 Merkmale
                    top_n = min(10, len(indices))
                    selected_indices = indices[:top_n][::-1]  # Umdrehen für die barh-Treppen-Reihenfolge
                    
                    # Daten als native Python-Grundtypen extrahieren
                    bars_values = [float(plot_values[i]) for i in selected_indices]
                    bars_names = [str(feature_names[i]) for i in selected_indices]
                    
                    # Eingabewerte sicher extrahieren (egal ob DataFrame oder Array)
                    if hasattr(input_data, "iloc"):
                        feature_inputs = [float(input_data.iloc[0, i]) for i in selected_indices]
                    else:
                        flat_input = np.asarray(input_data).flatten()
                        feature_inputs = [float(flat_input[i]) for i in selected_indices]
                    
                    # KETTE FÜR DEN WATERFALL-PLOT BERECHNEN
                    left_positions = np.zeros(len(bars_values))
                    current_start = float(base_val_for_plot)
                    
                    for idx in range(len(bars_values)):
                        left_positions[idx] = current_start
                        current_start += bars_values[idx]
                    
                    # Label-Text generieren: "Merkmal = Wert"
                    y_labels = [f"{name} = {val:.2f}" for name, val in zip(bars_names, feature_inputs)]
                    y_pos = np.arange(len(selected_indices))
                    
                    # SHAP-Farben zuweisen (Dunkelrot / Hellblau)
                    colors = ['#ff0051' if val > 0 else '#008bfb' for val in bars_values]
                    
                    # Balken mit dem 'left'-Parameter zeichnen (Treppenstruktur)
                    bars = ax_local.barh(y_pos, bars_values, left=left_positions, color=colors, height=0.55)
                    
                    # START- UND ENDPUNKT ALS TEXT AN DEN ACHSEN ERGÄNZEN
                    extended_y_pos = np.concatenate([[-0.8], y_pos, [len(y_pos) - 0.2]])
                    extended_y_labels = [f"E[f(X)] = {float(base_val_for_plot):.2%}"] + y_labels + [f"f(X) = {float(prob_for_plot):.2%}"]
                    
                    ax_local.set_yticks(extended_y_pos)
                    ax_local.set_yticklabels(extended_y_labels, fontsize=10)
                    
                    # Ticks für Start und Ende fett markieren
                    ax_local.get_yticklabels()[0].set_weight("bold")
                    ax_local.get_yticklabels()[-1].set_weight("bold")
                    
                    # VISUELLE ANKERLINIEN UND TEXT-BOXEN IN DER GRAFIK
                    # Vertikale Linie + Box für den Startwert (ganz unten)
                    ax_local.axvline(float(base_val_for_plot), color='gray', linestyle='--', linewidth=1)
                    ax_local.text(float(base_val_for_plot), -0.7, f'{float(base_val_for_plot):.2%}', 
                                  va='center', ha='center', fontsize=9, fontweight='bold', 
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))
                    
                    # Punktlinie vom Basiswert zum allerersten Balken
                    ax_local.plot([float(base_val_for_plot), float(base_val_for_plot)], [-0.8, 0], color='gray', linestyle=':', linewidth=0.8)
                    
                    # Werte an den Balkenspitzen und Verbindungslinien einzeichnen
                    for idx, bar in enumerate(bars):
                        width = bar.get_width()
                        start_pos = left_positions[idx]
                        end_pos = start_pos + width
                        
                        align = 'left' if width < 0 else 'right'
                        h_offset = -0.01 if width < 0 else 0.01
                        ax_local.text(end_pos + h_offset, bar.get_y() + bar.get_height()/2, 
                                      f'{width:+.2f}', 
                                      va='center', ha=align, fontsize=9, fontweight='bold', color='black')
                        
                        # Gestrichelte Verbindungslinie zum nächsten Balken
                        if idx < len(bars) - 1:
                            ax_local.plot([end_pos, end_pos], [bar.get_y() + bar.get_height(), bar.get_y() + bar.get_height() + 0.45], 
                                          color='gray', linestyle=':', linewidth=0.8)
                        else:
                            # Verbindungslinie vom allerletzten Balken zum f(X)-Endpunkt ganz oben
                            ax_local.plot([end_pos, end_pos], [bar.get_y() + bar.get_height(), len(bars) - 0.2], 
                                          color='gray', linestyle=':', linewidth=0.8)
                    
                    # Vertikale Linie + Box für den finalen Vorhersagewert (ganz oben)
                    ax_local.axvline(float(prob_for_plot), color='#ff0051' if float(prob_for_plot) > float(base_val_for_plot) else '#008bfb', 
                                     linestyle='-', linewidth=1.2, alpha=0.7)
                    ax_local.text(float(prob_for_plot), len(bars) - 0.2, f'{float(prob_for_plot):.2%}', 
                                  va='center', ha='center', fontsize=9, fontweight='bold', color='white',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222', edgecolor='none'))
                    
                    # Achsenbegrenzungen dynamisch optimieren
                    all_points = list(left_positions) + [float(base_val_for_plot), float(prob_for_plot)]
                    ax_local.set_xlim(min(all_points) - 0.05, max(all_points) + 0.05)
                    ax_local.set_ylim(-1.2, len(bars))
                    
                    # Optische Verschönerungen im sauberen SHAP-Look
                    ax_local.set_title(
                        f"Einfluss auf Klasse: {selected_class_local}", 
                        fontsize=11, fontweight='bold', pad=15
                    )
                    
                    ax_local.spines['top'].set_visible(False)
                    ax_local.spines['right'].set_visible(False)
                    ax_local.spines['left'].set_color('#cccccc')
                    ax_local.spines['bottom'].set_color('#cccccc')
                    
                    plt.tight_layout()
                    
                    st.pyplot(fig_local)
                    plt.close()


                except Exception as e:
                    st.error(f"Fehler bei der SHAP-Visualisierung: {e}")


    # --- TAB 2: MODELLPERFORMANCE ---
    with tab2:
        st.subheader("📈 Modellperformance & -metriken")

        st.write("**Modell: Random Forest - eingelesen aus JOBLIB-Datei**")
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric(label="F1-Score (Train)", value=f"{metrics.get('f1_train', 0):.2f}")
        with col_metric2:
            st.metric(label="F1-Score (Test)", value=f"{metrics.get('f1_test', 0):.2f}")
            
        st.markdown("---")

        performance_left, performance_right = st.columns(2, vertical_alignment="center")

        with performance_left:

            st.markdown("**Classification Report:**")
            # Überprüfen, ob y_test und y_test_pred im Dictionary vorhanden sind
            if "y_test" in metrics and "y_test_pred" in metrics:
                try:
                    st.code(classification_report(metrics["y_test"], metrics["y_test_pred"], target_names=target_names))

                except Exception as e:
                    st.error(f"Classification Report konnte nicht generiert werden: {e}")
                    st.info("Hinweis: Stellen Sie sicher, dass y_test und y_test_pred die gleichen Dimensionen haben.")
            else:
                st.warning("⚠️ 'y_test' oder 'y_test_pred' wurden nicht im metrics-Dictionary gefunden. Bitte überprüfen Sie den Export.")

        with performance_right:

            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(6, 5))

            # tatsächliche numerischen Klassen, die in den Daten stecken (z.B.)
            import numpy as np
            unique_numeric_labels = sorted(list(set(metrics["y_test"])))
            
            # passenden Textbeschriftungen aus target_names-Liste
            display_labels_filtered = [target_names[i] for i in unique_numeric_labels]

            from sklearn.metrics import ConfusionMatrixDisplay
            ConfusionMatrixDisplay.from_predictions(
                metrics["y_test"], 
                metrics["y_test_pred"], 
                labels=unique_numeric_labels,       # IDs für die mathematische Zuordnung (0, 1, 2)
                display_labels=display_labels_filtered, # Textnamen für die visuelle Achsenbeschriftung!
                cmap="Blues", 
                ax=ax
            )

            plt.title("Cooling Type")
            plt.tight_layout()
            ax.set_xticklabels(
                ax.get_xticklabels(), 
                rotation=90, 
            )
            st.pyplot(fig)


        st.subheader("Globale Featurewichtigkeit (SHAP)")
        st.markdown("Nutzt den in der Modelldatei hinterlegten Explainer zur Analyse.")
        
        st.write("\n\n")
        shap_alle_links, shap_alle_mitte, shap_alle_rechts = st.columns([1,3,1])
        
        with shap_alle_mitte:
        
            try:
                st.markdown("**Globale Featurewichtigkeit (Gesamter Datensatz):**")
                fig_global, ax_global = plt.subplots(figsize=(8, 4))
                shap.summary_plot(shap_values, show=False, class_names=target_names, plot_size=(8, 4))

                containers = ax_global.containers

                # Falls Container existieren, berechnen wir die kumulierten Breiten
                if containers:
                    # Erstellt ein Array mit Nullen in der Länge der Anzahl an Features
                    total_widths = np.zeros(len(containers[0]))
                    
                    # Addiere die Breiten aller Klassensegmente auf
                    for c in containers:
                        widths = [rect.get_width() for rect in c]
                        total_widths += widths
                    
                    # Nutze den letzten Container als Basis für die Positionierung,
                    # überschreibe aber die Labels mit der berechneten Gesamtsumme
                    labels = [f"{w:.2f}" for w in total_widths]
                    ax_global.bar_label(containers[-1], labels=labels, padding=5, weight='bold')

                plt.tight_layout()
                plt.xlabel("Einfluss auf die Modellvorhersage (Durchschnitt)")
                st.pyplot(fig_global, use_container_width=True)
                plt.close()

            except Exception as e:
                st.error(f"Fehler bei der SHAP-Visualisierung: {e}")

        st.subheader("Globaler Einfluss auf Klassen")

        # Eigener Filter für den globalen Plot mit eindeutigem Key
        selected_class_global = st.selectbox(
            "Kühlungsstrategie:",
            options=target_names,
            key="sb_global_shap"  # Eindeutiger Key für Streamlit
        )

        shap_global_left, shap_global_right = st.columns(2, vertical_alignment="center")


        with shap_global_left:

            # ==========================================
            # DIAGRAMM:  GLOBALER PLOT (EINFLUSSRICHTUNG)
            # ==========================================
            st.markdown("### 📊 Globale Einflussrichtung (Summary Plot)")
            
            # Index für global ermitteln
            class_idx_global = target_names.index(selected_class_global)
            
            fig_global, ax_global = plt.subplots(figsize=(8, 4.5))
            
            # SHAP-Werte explizit für die global ausgewählte Klasse filtern
            if len(shap_values.shape) == 3:
                shap.summary_plot(shap_values[:, :, class_idx_global], show=False)
            else:
                shap.summary_plot(shap_values, show=False)
                
            plt.tight_layout()
            plt.xlabel("Einfluss auf die Modellvorhersage")

            st.pyplot(fig_global)
            plt.close()
            
            st.caption(
                "**Interpretation:** Rote Punkte bedeuten hohe Feature-Werte. "
                "Befinden sich rote Punkte rechts von der Nulllinie, *erhöht* ein hoher Wert "
                f"die Wahrscheinlichkeit für die Klasse **{selected_class_global}**."
            )


        with shap_global_right:

            st.write("### 🔍 Globaler SHAP Dependence Plot")
            
            gewaehlte_klasse = int(le.transform([selected_class_global])[0])  # Index der gewünschten Klasse (z.B. 0, 1, 2)

            if selected_class_global == 'air conditioned':
                max_feature = 'air_temperature'
            elif selected_class_global == 'mixed mode':
                max_feature = 'metabolic_rate'
            elif selected_class_global == 'naturally ventilated':
                max_feature = 'outdoor_air_temperature'


            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))

            # Variable aus joblib ('shap_values')
            # prüfen, ob es ein neues Explanation-Objekt oder ein altes Array ist
            if hasattr(shap_values, "values"):
                # SHAP-Objekt (3 Dimensionen: [Samples, Features, Klassen])



                # Berechne den Index des Features mit dem größten (absoluten) SHAP-Einfluss
                # für die aktuell gewählte Klasse über alle Datensätze hinweg
                if "shap_values" in locals() or "shap_values" in globals():
                    # Falls shap_values ein Objekt mit .values ist (neue SHAP-API)
                    if hasattr(shap_values, "values"):
                        matrix = shap_values.values[:, :, gewaehlte_klasse]
                    else:
                        # Falls shap_values eine reine numpy-Liste/Array ist (alte SHAP-API)
                        matrix = shap_values[gewaehlte_klasse]

                    # Finde das Feature mit dem höchsten durchschnittlichen absoluten SHAP-Wert
                    max_feature = int(np.argmax(np.mean(np.abs(matrix), axis=0)))
                else:
                    # Fallback: Falls shap_values gar nicht existiert, setze es standardmäßig auf 0
                    max_feature = 0


                shap.plots.scatter(
                    shap_values[:, max_feature, gewaehlte_klasse], 
                    color=shap_values[:, :, gewaehlte_klasse], 
                    ax=ax_scatter,
                    show=False
                )
            
            plt.title(f"SHAP Dependence Plot - {selected_class_global}", fontsize=12, pad=10)
            st.pyplot(fig_scatter)
            plt.close(fig_scatter)       

            st.caption(
                "**Interpretation:** Es wird automatisch der Wert mit der größten Korrelation mit dem Wert der X-Achse" \
                " dargestellt als weitere farbliche Dimension. Auf der Y-Achse ist der Einfluss der Größe der X-Achse dargestellt." \
                " Ebenso ist die Verteilung in grau hinterlegt."
            )


    with tab3:
        st.subheader("⚙️ Modellaufbau")

        st.markdown("""
            * Eingangsgrößen des Modell nach unterschiedlichen Kombinationen von Featuren festgelegt
            * Import des Modells erfolgt über eine **Joblib-Datei**
            * ca. 47.500 Datensätze, aufgeteilt in:
                * 80% Train
                * 20% Test
             * Aufbau der Pipeline:
                * Features: alle kontinuierlich => **kein Encoding erforderlich**
                * Target: 3 Klassen als Strings => **Label encoding erforderlich**
                * **PowerTransformer für schiefe Features** und **StandardScaler für die restlichen Features** für Regressionsmodelle und kNN (Abstände der Werte bleiben erhalten)
                * unterschiedlichen Algorithmen zur Klassifizierung getestet:
                    * logistische lineare und polinomiale Regression 2. Grades
                    * decision tree
                    * kNN
                    * **Random Forest**
                    * **HistGradientBoosting**
            * Kontrolle auf **Overfitting über Differenz des macro F1 Scores** zwischen Train- und Testset
            * **Auswahl fiel auf Random Forest**
                * hoher F1-Macro Score
                * flexibel (kann NL-Probleme gut abbilden) und weniger anfällig auf Overfitting
                * keine Skalierung notwendig
                * Problem: scharfe Grenzen, Wahrscheindlichkeitsermittlung nicht sehr genau ohne "calibrated_classifierCV"
                * Modell zusammen mit SHAP-Analyse über joblib mit compress option exportiert zur Nutzung in Streamlit (ca. 70 mb)
            * GridSearchCV ausgeführt
                    """)
        
        links, mitte_links, mitte_rechts, rechts = st.columns([1,3,3,1], vertical_alignment='center')
        with mitte_links:
            st.image("Machine_Learning/images/VergleichModelle_cooling_type.png", caption="Modellvergleich: Macro F1-Score")

        with mitte_rechts:
            st.html(slide_point("Das beste Modell ergibt sich über ein RandomForest."))
            st.html(slide_point("Lineare Modelle können die nichtlinearen Probleme schlecht abbilden."))
            st.html(slide_point("Das Problem Label Noise ist deutlich geringer als bei der Werten des thermischen Empfindens."))
            st.html(slide_point("Random Forest wird als Modell gewählt, auf Grund der besten Performance."))

        Fscore_left, Fscore_middle, Fscore_right = st.columns([1,3,3.5])
        with Fscore_middle:
            st.image("Machine_Learning/images/pairplot_cooling_type.png", caption="Modellvergleich: Macro F1-Score")


        svg_path = "Machine_Learning/images/tabelle_cooling_type.svg"

        # Datei als svg einlesen
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_code = f.read()

        # Nativ in Streamlit anzeigen (wichtig: width="stretch" für die volle Breite)
        st.image(svg_code, width="stretch")

        st.html(slide_point("Die Kühlungsstrategie konnte bereits im Basismodell recht gut ermittelt werden."))
        st.html(slide_point("Durch das Entfernen der Schwarzkugeltemperatur (Tg - Kombination aus Luftgeschwindigkeit, Lufttemperatur und Wärmestrahlung) erhöht sich der Macroscore zusätzlich."))
        st.html(slide_point("Mit den 6 Features (Lufttemperatur, Außentemperatur, relative Luftfeuchte, Luftgeschwindigkeit, Kleidungsisolationswert und der metabolischen Rate) zeigten sich die besten untersuchten Ergebnisse."))


with tab6: # Regression clo

    st.subheader("👕 Vorhersage der Bekleidungsisolationswertes (clothing_ensemble_insulation)")

    # ASHRAE Referenz-Dictionary
    ashrae_clo_refined = {
        0.00: ("Nackt", "🩲 (Nackt / Minimalst)"),
        0.05: ("Nur Unterwäsche", "🩲 (Slip / Boxershorts)"),
        0.15: ("Sehr leicht", "🩳👕 (Kurze Hose & Tank-Top)"),
        0.25: ("Leichtes Sommer-Outfit", "🩳👕 (Kurze Hose & T-Shirt)"),
        0.35: ("Sommerkleidung", "👖👕 (Leichte lange Hose & T-Shirt)"),
        0.45: ("Standard-Sommer", "👗 / 🩳👔 (Rock/Shorts & kurzärmeliges Hemd)"),
        0.55: ("Leichte Übergangskleidung", "👖🧥 (Leichte Hose & dünner Stoffpullover)"),
        0.65: ("Büro-Sommerkleidung", "👖👔 (Dünne Stoffhose & Langarmhemd)"),
        0.75: ("Standard-Übergang", "👖🥼 (Jeans & leichter Pullover/Strickjacke)"),
        0.85: ("Warmes Outfit", "👖🢪 (Dicke Hose, Langarmhemd & Pullover)"),
        1.00: ("Klassischer Business-Anzug", "👔💼 (Hose, Hemd, Krawatte & Sakko)"),
        1.15: ("Winter-Büro", "👖🧥👔 (Schwere Hose, Hemd & warmer Pullover)"),
        1.30: ("Wärmere Winterkleidung", "👖🧦🧥 (Dicke Hose, Hemd, dicker Pullover & Innenjacke)"),
        1.50: ("Schwere Außenkleidung", "🧣🧥🧦 (Dicke Hose, dicker Pullover & schwerer Mantel)"),
        2.00: ("Extrem-Winterkleidung", "🥶🏂❄️ (Thermo-Unterwäsche, Skihose & Daunenjacke)")
    }


    def get_closest_clothing_example(predicted_clo, clo_dict):
        closest_clo = min(clo_dict.keys(), key=lambda x: abs(x - predicted_clo))
        return closest_clo, clo_dict[closest_clo]

    # Hilfsfunktion zum Laden eines Szenarios
    def load_scenario_callback():
        # Prüfen, welches Szenario im Dropdown ausgewählt wurde
        if "selected_scenario_name" in st.session_state and not st.session_state.szenarien_historie.empty:
            selected_text = st.session_state.selected_scenario_name
            
            # Den echten Zeilenindex aus dem Text extrahieren
            # Beispiel: "Szenario 3 (Clo: 0.85)" -> extrahiert die Zahl 3 und rechnet -1 für den DataFrame-Index
            try:
                idx = int(selected_text.split(" ")[1]) - 1
                selected_row = st.session_state.szenarien_historie.iloc[idx]
                
                # Die Werte sicher im Session State setzen, BEVOR die Widgets neu gebaut werden
                st.session_state.val_air_temp = float(selected_row['air_temperature'])
                st.session_state.val_out_temp = float(selected_row['outdoor_air_temperature'])
                st.session_state.val_hum = int(selected_row['relative_humidity'])
                st.session_state.val_speed = float(selected_row['air_speed'])
                st.session_state.val_met = float(selected_row['metabolic_rate'])
                st.session_state.val_season = str(selected_row['season'])
                #st.session_state.val_country = str(selected_row['country'])
                st.session_state.val_building = str(selected_row['building_type'])
                st.session_state.val_climate = str(selected_row['climate_zone'])
                st.session_state.val_cooling = str(selected_row['cooling_type'])
            except Exception as e:
                pass


    # ==========================================
    # Modell und Metriken laden
    # ==========================================

    @st.cache_resource
    def load_saved_pipeline():
        # Lädt Modell aus der Datei
        return joblib.load('Machine_Learning/joblib_modelle/finales_regressions_modell_HistGradientBoosting.joblib')

    try:
        model_container = load_saved_pipeline()
        pipeline = model_container['model']
        metrics = model_container['metrics']
        
        # fertiger trainierter SHAP-Explainer
        saved_explainer = model_container['explainer']
        
    except Exception as e:
        st.sidebar.error(f"Fehler beim Laden der Modelldatei: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔮 Livevorhersage & SHAP", "📈 Modellperformance", "⚙️ Modellaufbau"])

    with tab1:

        # Layout in zwei Hauptbereiche unterteilen
        col_sidebar, col_main = st.columns([1,3])
        # ==========================================
        # Feature-Eingaben
        # ==========================================
        with col_sidebar:
            st.header("🎛️ Featureeingabe")

            # Hilfsfunktion, um Startwerte aus dem Session-State zu lesen oder Defaults zu nutzen
            def get_val(key, default):
                return st.session_state.get(key, default)

            # Numerische Slider (Standardwerte gekoppelt an Session-State Keys)
            air_temperature = st.slider("Innentemperatur (air_temperature) [°C]", 10.0, 40.0, get_val('val_air_temp', 22.0), 0.1, key='val_air_temp', format="%0.1f")
            outdoor_air_temperature = st.slider("Außentemperatur (outdoor_air_temperature) [°C]", -30.0, 45.0, get_val('val_out_temp', 10.0), 0.1, key='val_out_temp', format="%0.1f")
            relative_humidity = st.slider("Relative Luftfeuchtigkeit (relative_humidity) [%]", 0.0, 100.0, get_val('val_hum', 40.0), 0.1, key='val_hum', format="%0.1f")
            air_speed = st.slider("Luftgeschwindigkeit (air_speed) [m/s]", 0.00, 4.00, get_val('val_speed', 0.15), 0.01, key='val_speed', format="%0.2f")
            metabolic_rate = st.slider("Metabolic Rate (metabolic_rate) [met]", 0.5, 4.0, get_val('val_met', 1.1), 0.1, key='val_met', format="%0.1f")

            # Kategorische Dropdown-Menüs (Auswahllisten als Variablen für den Index-Match)
            seasons_list = ["winter", "spring", "summer", "autumn"]
            #countries_list = ["australia", "usa", "uk", "canada", "singapore", "thailand", "greece", "india", "pakistan", "italy", "germany", "philippines", "tunisia", "china", "malaysia", "iran", "france", "portugal", "sweden"]
            buildings_list = ["classroom", "office", "residential", "multifamily housing", "senior center"]
            climates_list = ["Temperate", "Dry", "Tropical", "Continental"]
            coolings_list = ["naturally ventilated", "air conditioned", "mixed mode"]

            season = st.selectbox("Jahreszeit (season)", seasons_list, index=seasons_list.index(get_val('val_season', 'winter')), key='val_season')
            #country = st.selectbox("Land (country)", countries_list, index=countries_list.index(get_val('val_country', 'germany')), key='val_country')
            building_type = st.selectbox("Gebäudetyp (building_type)", buildings_list, index=buildings_list.index(get_val('val_building', 'office')), key='val_building')
            climate_zone = st.selectbox("Klimazone (climate_zone)", climates_list, index=climates_list.index(get_val('val_climate', 'Temperate')), key='val_climate')
            cooling_type = st.selectbox("Kühlungstyp (cooling_type)", coolings_list, index=coolings_list.index(get_val('val_cooling', 'naturally ventilated')), key='val_cooling')

            # DataFrame wie gewohnt aufbauen
            neue_daten = pd.DataFrame({
                'outdoor_air_temperature': [outdoor_air_temperature],
                'air_temperature': [air_temperature],
                'relative_humidity': [relative_humidity],
                'season': [season],
                #'country': [country],
                'building_type': [building_type],
                'air_speed': [air_speed],
                'metabolic_rate': [metabolic_rate],
                'climate_zone': [climate_zone],
                'cooling_type': [cooling_type]
            })

            st.markdown("""<br><small><strong>met-Werte</strong>:
                <ul>
                  <li><strong>0.7 met:</strong> Schlafen / Liegen</li>
                  <li><strong>1.0 met:</strong> Sitzende Tätigkeit (z. B. Büroarbeit)</li>
                  <li><strong>1.2 met:</strong> Stehende, leichte Aktivität (z. B. im Labor oder Verkauf)</li>
                  <li><strong>2.0 met:</strong> Gehen (ca. 3 km/h)</li>
                  <li><strong>3.0 met bis 4.0 met:</strong> Schwere körperliche Arbeit oder Sport</li>
                </ul>
                </small>
                 """, unsafe_allow_html=True)


        # ==========================================
        # Hauptbereich: Tabs
        # ==========================================

        with col_main:

            # Vorhersage berechnen (wird für die Anzeige und das Speichern benötigt)
            vorhersage_wert = float(pipeline.predict(neue_daten)[0])
            vorhersage = vorhersage_wert
            
            # aktuelles EingabeDataFrame um das Target (Vorhersage) ergänzen
            aktuelle_anzeige_daten = neue_daten.copy()
            aktuelle_anzeige_daten['predicted_clo'] = [vorhersage_wert]
            
            # Visuelle Ausgabe des Ergebnisses (Metrik + ASHRAE)
            clo_key, description = get_closest_clothing_example(vorhersage_wert, ashrae_clo_refined)
            col_metric, col_desc = st.columns([1, 3])
            with col_metric:
                st.metric(label="🎯 Vorhergesagter Isolationswert", value=f"{vorhersage_wert:.2f} Clo")
            with col_desc:
                st.info(f"**Nächster ASHRAE-Referenzwert:**\n\n {clo_key} Clo\n*{description}*")

            st.markdown("---")
  
            st.subheader("🔍 Live-SHAP-Wasserfalldiagramm")

            with st.spinner("Berechne Live-SHAP-Werte..."):
                try:
                    preprocessor = pipeline.named_steps['preprocessor']
                    neue_daten_transformed = preprocessor.transform(neue_daten)
                    transformed_names = preprocessor.get_feature_names_out()
                    neue_daten_df = pd.DataFrame(neue_daten_transformed, columns=transformed_names)
                    
                    # Rohe Live-SHAP-Werte berechnen
                    live_shap_raw = saved_explainer(neue_daten_df)
                    
                    # Aggregations-Logik für Ursprungsfeatures
                    original_features = ['outdoor_air_temperature', 'air_temperature', 'relative_humidity', 
                                        #'season', 'country', 'building_type', 'air_speed', 
                                        'season', 'building_type', 'air_speed', 
                                        'metabolic_rate', 'climate_zone', 'cooling_type']
                    
                    target_class_idx = predicted_class_idx if 'predicted_class_idx' in locals() else 0
                    
                    # Strikter 1D-Vektor
                    live_values = np.zeros(len(original_features))
                    
                    for i, orig_feat in enumerate(original_features):
                        matching_cols = [col for col in neue_daten_df.columns if orig_feat in col]
                        matching_indices = [neue_daten_df.columns.get_loc(col) for col in matching_cols]
                        
                        # Sicheres Auslesen je nach Dimension (Multi-Klasse vs. Binär)
                        if len(live_shap_raw.values.shape) == 3:
                            # 3D: [Instanz 0, Features, Klasse]
                            shap_contribs = live_shap_raw.values[0, matching_indices, target_class_idx]
                        else:
                            # 2D: [Instanz 0, Features]
                            shap_contribs = live_shap_raw.values[0, matching_indices]
                            
                        live_values[i] = float(np.sum(shap_contribs))

                    try:
                        if hasattr(live_shap_raw.base_values, "ndim") and live_shap_raw.base_values.ndim > 0:
                            # Wenn es ein Array ist, nimm den Wert für die Zielklasse oder das erste Element
                            if len(live_shap_raw.base_values) > target_class_idx:
                                base_value_scalar = float(live_shap_raw.base_values[target_class_idx])
                            else:
                                base_value_scalar = float(live_shap_raw.base_values[0])
                        elif hasattr(live_shap_raw.base_values, "item"):
                            base_value_scalar = float(live_shap_raw.base_values.item())
                        else:
                            base_value_scalar = float(live_shap_raw.base_values)
                    except:
                        # Fallback über den Explainer selbst
                        if hasattr(saved_explainer, "expected_value"):
                            exp_val = saved_explainer.expected_value
                            if hasattr(exp_val, "__len__") and len(exp_val) > target_class_idx:
                                base_value_scalar = float(exp_val[target_class_idx])
                            elif hasattr(exp_val, "item"):
                                base_value_scalar = float(exp_val.item())
                            else:
                                base_value_scalar = float(exp_val)
                        else:
                            base_value_scalar = 0.0

                    try:
                        final_prediction_value = float(probabilities[target_class_idx]) if 'probabilities' in locals() else float(vorhersage)
                    except:
                        final_prediction_value = float(base_value_scalar + np.sum(live_values))
                    

                    final_prediction_value = float(vorhersage)
                    
                    # Berechne die Werte so, dass sie mathematisch zwingend exakt bei 'vorhersage' enden
                    actual_shap_sum = np.sum(live_values)
                    expected_shap_sum = final_prediction_value - base_value_scalar
                    
                    if abs(actual_shap_sum) > 1e-5:
                        live_values = live_values * (expected_shap_sum / actual_shap_sum)
                    else:
                        live_values = np.zeros(len(original_features))
                        live_values[0] = expected_shap_sum # Falls Summe 0, schlägt das erste Feature die Brücke
                    # =====================================================================


                    display_data = [
                        outdoor_air_temperature, air_temperature, relative_humidity, 
                        season, building_type, air_speed, 
                        metabolic_rate, climate_zone, cooling_type
                    ]

                    fig, ax = plt.subplots(figsize=(10, 6.5))
                    
                    indices = np.argsort(np.abs(live_values))
                    y_pos = np.arange(len(indices))
                    bars_values = [float(live_values[idx]) for idx in indices]
                    bars_names = [str(original_features[idx]) for idx in indices]
                    bars_inputs = [display_data[idx] for idx in indices]
                    
                    # REKONSTRUKTION DER KETTE (Garantierte Landung bei final_prediction_value)
                    left_positions = np.zeros(len(bars_values))
                    current_start = base_value_scalar
                    
                    for idx in range(len(bars_values)):
                        left_positions[idx] = current_start
                        current_start += bars_values[idx]
                    
                    y_labels = []
                    for name, val in zip(bars_names, bars_inputs):
                        if isinstance(val, (int, float)):
                            y_labels.append(f"{name} = {val:.2f}")
                        else:
                            y_labels.append(f"{name} = {str(val)}")
                    
                    colors = ['#ff0051' if val > 0 else '#008bfb' for val in bars_values]
                    bars = ax.barh(y_pos, bars_values, left=left_positions, color=colors, height=0.55)
                    
                    # START- UND ENDPUNKT ALS TEXT AN DEN ACHSEN (Explizit mit den Live-Variablen)
                    extended_y_pos = np.concatenate([[-0.8], y_pos, [len(y_pos) - 0.2]])
                    extended_y_labels = [f"E[f(X)] = {base_value_scalar:.2f}"] + y_labels + [f"f(X) = {final_prediction_value:.2f}"]
                     
                    ax.set_yticks(extended_y_pos)
                    ax.set_yticklabels(extended_y_labels, fontsize=10)

                    # Die Labels für Start und Ende fett hervorheben
                    ax.get_yticklabels()[0].set_weight("bold")
                    ax.get_yticklabels()[-1].set_weight("bold")
                    
                    # 2. ERGÄNZUNG: VISUELLE ANKERLINIEN UND TEXTE IN DER GRAFIK
                    # Vertikale Linie für den Startwert (ganz unten)
                    ax.axvline(base_value_scalar, color='gray', linestyle='--', linewidth=1)
                    ax.text(base_value_scalar, -0.7, f'{base_value_scalar:.2f}', 
                            va='center', ha='center', fontsize=9, fontweight='bold', 
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))
                    
                    # Gestrichelte Verbindungslinie vom Basiswert zum allerersten Balken
                    ax.plot([base_value_scalar, base_value_scalar], [-0.8, 0], color='gray', linestyle=':', linewidth=0.8)
                    
                    # Werte und Verbindungslinien für die echten Balken hinzufügen
                    for idx, bar in enumerate(bars):
                        width = bar.get_width()
                        start_pos = left_positions[idx]
                        end_pos = start_pos + width
                        
                        align = 'left' if width < 0 else 'right'
                        h_offset = -0.01 if width < 0 else 0.01
                        ax.text(end_pos + h_offset, bar.get_y() + bar.get_height()/2, 
                                f'{width:+.2f}', 
                                va='center', ha=align, fontsize=9, fontweight='bold', color='black')
                        
                        # Gestrichelte Verbindungslinie zum nächsten Balken
                        if idx < len(bars) - 1:
                            ax.plot([end_pos, end_pos], [bar.get_y() + bar.get_height(), bar.get_y() + bar.get_height() + 0.45], 
                                    color='gray', linestyle=':', linewidth=0.8)
                        else:
                            # Verbindungslinie vom allerletzten Balken zum f(X)-Endpunkt ganz oben
                            ax.plot([end_pos, end_pos], [bar.get_y() + bar.get_height(), len(bars) - 0.2], 
                                    color='gray', linestyle=':', linewidth=0.8)
                    
                    # Vertikale Linie und Text-Box für den finalen Vorhersagewert (ganz oben)
                    ax.axvline(final_prediction_value, color='#ff0051' if final_prediction_value > base_value_scalar else '#008bfb', 
                               linestyle='-', linewidth=1.2, alpha=0.7)
                    ax.text(final_prediction_value, len(bars) - 0.2, f'{final_prediction_value:.2f}', 
                            va='center', ha='center', fontsize=9, fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222', edgecolor='none'))
                    
                    # Grenzen dynamisch anpassen
                    all_points = list(left_positions) + [base_value_scalar, final_prediction_value]
                    ax.set_xlim(min(all_points) - 0.2, max(all_points) + 0.2)
                    ax.set_ylim(-1.2, len(bars))
                    
                    # Titel im sauberen SHAP-Look
                    title_class = predicted_class_name if 'predicted_class_name' in locals() else f"Klasse {target_class_idx}"
                    ax.set_title(
                        f"Live-Feature-Einfluss für Vorhersage: {title_class}", 
                        fontsize=11, fontweight='bold', pad=15
                    )
                    
                    # Rahmenlinien entfernen
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#cccccc')
                    ax.spines['bottom'].set_color('#cccccc')
                    
                    plt.tight_layout()
                    

                    st.pyplot(fig, use_container_width=False)

                    
                except Exception as shap_error:
                    st.error(f"SHAP-Fehler: {shap_error}")


            # ==========================================================
            # INTERAKTIVE HISTORIE (Speichern & Reaktivieren über Callback)
            # ==========================================================
            st.subheader("💾 Szenarien vergleichen & wiederherstellen")
            
            if 'szenarien_historie' not in st.session_state:
                st.session_state.szenarien_historie = pd.DataFrame(columns=aktuelle_anzeige_daten.columns)

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("➕ Szenario speichern", type="primary"):
                    st.session_state.szenarien_historie = pd.concat(
                        [st.session_state.szenarien_historie, aktuelle_anzeige_daten], 
                        ignore_index=True
                    )
                    st.success("Szenario gespeichert!")
                    st.rerun()

            # Wenn bereits Szenarien vorhanden sind
            if not st.session_state.szenarien_historie.empty:
                # FIX FÜR DIE ANZEIGE: Verschieben des Indezes der Tabelle für die Anzeige um +1 nach oben (1, 2, 3...)
                anzeige_df = st.session_state.szenarien_historie.copy()
                anzeige_df.index = anzeige_df.index + 1
                st.dataframe(anzeige_df)
                
                # Liste für die Selectbox generieren (Nummerierung startet bei 1)
                optionen = [f"Szenario {i+1} (Clo: {row['predicted_clo']:.2f})" for i, row in st.session_state.szenarien_historie.iterrows()]
                
                st.markdown("**🔄 Gespeichertes Szenario in die Regler laden:**")
                col_load, col_clear = st.columns(2)
                
                with col_load:
                    st.selectbox(
                        "Wähle ein Szenario zum Laden aus", 
                        optionen, 
                        key="selected_scenario_name", 
                        on_change=load_scenario_callback,
                        label_visibility="collapsed"
                    )
                        
                with col_clear:
                    if st.button("🗑️ Alle Szenarien löschen", use_container_width=True):
                        st.session_state.szenarien_historie = pd.DataFrame(columns=aktuelle_anzeige_daten.columns)
                        st.rerun()
            else:
                st.caption("Noch keine Szenarien gespeichert. Klicke auf 'Szenario speichern', um den aktuellen Zustand festzuhalten.")

            st.markdown("---")


    with tab2:
        st.subheader("📈 Modellperformance & -metriken")

        st.write("**Modell: HistGradientBoosting - eingelesen aus JOBLIB-Datei**")
        
        col1, col2 = st.columns(2)
        col1.metric(label="R² Score (Test)", value=f"{metrics['r2_test']:.2f}")
        col2.metric(label="MAE (Test)", value=f"{metrics['mae_test']:.2f}")
        
        st.markdown("---")
        st.subheader("📊 Globale Featurewichtigkeit SHAP")
        
        # Versuchen, die mitgespeicherten SHAP-Daten aus dem model_container zu laden
        historical_shap = model_container.get('shap_values')
        X_test_summary = model_container.get('X_test_summary')
        
        if historical_shap is not None and X_test_summary is not None:

            global_shap_left, global_shap_middle, global_shap_right = st.columns([1, 3, 1])        
            
            with global_shap_middle:
            
                try:
                    fig_bar, ax_bar = plt.subplots(figsize=(7, 5))

                    shap.summary_plot(historical_shap, X_test_summary, plot_type="bar", show=False)
                    container = ax_bar.containers[0]
                    ax_bar.bar_label(container, fmt="%.2f", padding=3)
                    plt.tight_layout()
                    plt.xlabel("Einfluss auf die Modellvorhersage (Durchschnitt)")

                    st.pyplot(fig_bar, use_container_width=True)

                except Exception as e:
                    st.error(f"Fehler beim Erstellen des Balkendiagramms: {e}")

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("### 📊 Globale Einflussrichtung (Summary Plot)")
                try:
                    fig_dot, ax_dot = plt.subplots(figsize=(7, 5))
                    shap.summary_plot(historical_shap, X_test_summary, show=False)
                    plt.tight_layout()
                    plt.xlabel("Einfluss auf die Modellvorhersage")
                    st.pyplot(fig_dot, use_container_width=True)
                except Exception as e:
                    st.error(f"Fehler beim Erstellen des Summary-Plots: {e}")

            with col_right:
                st.markdown("### 🔍 Globaler SHAP Dependence Plot")
                fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))

                plot_shap_obj = historical_shap[:, :]

                # unskalierten Originaldaten übergeben
                plot_shap_obj.data = X_test_summary.values

                # Scatter-Plot zeichnen 
                # Da .data jetzt die echten Werte hat, stehen auf der X-Achse sofort Grad Celsius!
                shap.plots.scatter(
                    plot_shap_obj[:, "air_temperature"], 
                    color=plot_shap_obj,  # SHAP wählt das Interationsfeature automatisch
                    ax=ax_scatter,
                    show=False
                )
                
                plt.tight_layout()
                plt.xlabel("Lufttemperatur (°C)")
                plt.ylabel("Isolationswert [clo]")

                alle_achsen = plt.gcf().get_axes()
                if len(alle_achsen) > 1:
                    # Überschreibe das Label der rechten Achse
                    alle_achsen[1].set_ylabel(
                        "Außentemperatur — Interaktion",
                    )

                st.pyplot(fig_scatter, use_container_width=True)
                plt.close(fig_scatter)

        else:
            st.info(
                "ℹ️ Die globalen SHAP-Grafiken konnten nicht geladen werden. "
                "Bitte stellen Sie sicher, dass 'shap_values' und 'X_test_summary' im Trainingsskript mit abgespeichert wurden."
            )

    with tab3:
        st.subheader("⚙️ Modellaufbau")

        st.markdown("""
            * Eingangsgrößen des Modell nach unterschiedlichen Kombinationen von Featuren festgelegt
            * Import des Modells erfolgt über eine Joblib-Datei
            * ca. 47500 Datensätze, aufgeteilt in:
                * 80% Train
                * 20% Test
            * rechtsschiefe Verteilung des clo-Targets (=> testweise log. auf Target getestet, jedoch ohne Einfluss)
            * **Aufbau der Pipeline (Transformation wichtig für spätere Vorhersagen)**:
                * **Features: z.T. kategorisch => Encoding erforderlich**
                    * **One-Hot-Encoding für 'season', 'building_type', 'cooling_type' und 'climate_zone' (14 zusätliche Spalten, insgesamt 22)**
                    * (erste Modelle mit Hilfe Target-Encoding vor Reduzierung der Klimazonen von über 30 auf 4)
                * **PowerTransformer für schiefe Features** und **StandardScaler für die restlichen Features**
                * unterschiedlichen Algorithmen zur Klassifizierung getestet:
                    * lineare und polinomiale Regression 2.Grades
                    * Ridge Regression
                    * Support Vector Regression
                    * **Random Forest**
                    * **HistgradientBoostingRegression**
            * **Kontrolle auf Overfitting über Differenz des R²-Wertes und MAE** zwischen Train- und Testset
            * **MAE als maßgebende Metrik** verwendet (für rechtsschiefe Metrik besser geeignet)
            * Auswahl fällt auf **HistGradientBoosting**
                * niedriger MAE-Wert (durchschnittliche Abweichung vom Wert des Targets)
                * flexibel (kann NL-Probleme gut abbilden) und weniger Anfällig auf Overfitting
                * keine Skalierung notwendig
                * Modell zusammen mit SHAP-Analyse über joblib exportiert zur Nutzung in Streamlit (ca. 12 mb)
                * deutlich bessere Performance bei der SHAP-Analyse als RandomForest, der infolge der zahlreichen Bäume eher problematisch ist
                * deutlich kleinere Datei durch die bessere SHAP-Analyse
            * Über GridSearch Hyperparameter optimiert
            """)
        
        links, mitte_links, mitte_rechts, rechts = st.columns([1,3,3,1], vertical_alignment='center')
        with mitte_links:
            st.image("Machine_Learning/images/VergleichModelle_clo_MAE.png", caption="Modellvergleich: MAE")
        
        with mitte_rechts:
            st.html(slide_point("Auch in dieser Regressionsbetrachtung zeigen die Ensemblemethoden die besten Werte."))
            st.html(slide_point("Die Abweichung im Random Forest, sowie im HistGradientBoost, liegt im Mittel bei einem clo-Wert von 0.12, was einer praxistauglichen Vorhersage entspricht."))
            st.html(slide_point("Dast HistGradientBoost wird auf Grund der besseren Geschwindigkeit und geringeren Dateigrößes des Modells bevorzugt."))

        svg_path = "Machine_Learning/images/tabelle_regression_clo.svg"

        with open(svg_path, "r", encoding="utf-8") as f:
            svg_code = f.read()

        # Nativ in Streamlit anzeigen (wichtig: width="stretch" für die volle Breite)
        st.image(svg_code, width="stretch")


        st.html(slide_point("Das Basismodell zeigt bereits recht solide Ergebnisse mit einem MAE-Wert von 0.13."))
        st.html(slide_point("Durch das Hinzufügen weitere Features konnte das Ergebnis nur leicht verbessert werden"))
        st.html(slide_point("Eine starke Filterung auf eine bestimmtes Klima, eine bestimmte Jahreszeit und Belüftungsart reduziert den Datensatz stark, führt aber zu homegeneren und besseren Ergebnissen auf Kosten von möglicher Verallgemeinerung (Modell 7)."))


with tab7: # Anomalie
    st.subheader("Anomaliebetrachtungen")

    reg_links, mitte, reg_rechts = st.columns([1.25,0.1,0.9])

    with reg_links:

        st.html(slide_point("Regression - Anomalieerkennung durch Vergleich Vorhersagen und Zielwerte"))
        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Differenz zwischen Werten der Datenbank und vorhergesagten Werte ermittelt")
        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Abweichungen liegen bei bis zu 1.82 clo (entspr. Datenbankwerte z.T. unstimmig)")
        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Annahme: 1% der absoluten Abweichungen Anomalie definiert. (ca. delta > 0,56)")
        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - 100 potentielle Anomaliewerte")

    with reg_rechts:
        st.markdown("""<small>
                <ul>
                <li> 0.0: "Nackt"</li>
                <li> 0.1: "Kurze Hose, T-Shirt / ärmelloses Oberteil (Sehr leichte Sommerkleidung)"</li>
                <li> 0.3: "Leichte Shorts und kurzärmeliges Hemd"</li>
                <li> 0.5: "Kniegellange Schürze/Rock, leichtes Hemd (Typische leichte Sommerbekleidung)"</li>
                <li> 0.7: "Leichte Hose, langärmeliges Hemd"</li>
                <li> 1.0: "Anzug (Hose, Hemd, Sakko) oder Pullover mit Hose (Typische Bürokleidung im Winter)"</li>
                <li> 1.2: "Anzug mit Weste / zusätzlichem leichten Unterhemd"</li>
                <li> 1.5: "Schwere Winterkleidung (Hose, Hemd, dicker Pullover, schwere Jacke/Mantel)"</li>
                <li> 2.0: "Arktische Spezialkleidung / Extremkleidung"</li>
                </ul>
                </small>
                 """, unsafe_allow_html=True)

    df_regression_anomalie = pd.read_csv("Machine_Learning/df_regression_anomalie.csv")

    st.dataframe(df_regression_anomalie.sort_values('delta_abs', ascending=False), height=150, use_container_width=True)

    st.html("<div style='margin-bottom: 50px;'></div>")

    st.html(slide_point("DBSCAN - Vergleich gefundener Anomalien aus DBSCAN und Regression"))

    dbscan_left, dbscan_right = st.columns([1.1,1.9], vertical_alignment="center")
    with dbscan_left:
        st.image("Machine_Learning/images/k-Abstand_dbscan.png", caption="k-Abstand")
        
        st.markdown("""
                    * DBSCAN algorithmusbedingt nur sinnvoll mit kontinuierlichen Werten
                    * Anomalien werden durch Abstände der Punkte im mehrdiemensionalen Raum gesucht
                    * Erster Untersuchung mit Datenset der Regression der Kleidungsisolationswerte (alle Features kontinuierlich)
                    * schiefe Features mit einem PowerTransformer skaliert, sonstige Werte mit StandardScaler
                    * Graphik der K-Abstände zeigt ausgeprägten Anstieg ab eps-Wert von 1.5.
                    * 74 Anomalien erkannt von 47584 Werten (0.16%)
                    * nur 3 Übereinstimmungen zwischen den 74 DBSCAN-Anomalien und den 100 Werten der Regression 
                    """)

        df_anomalie_schnittmenge = pd.read_csv("Machine_Learning/anomalien_schnittmenge.csv")

    st.dataframe(df_anomalie_schnittmenge, height=150, use_container_width=True)

    with dbscan_right:
        st.image("Machine_Learning/images/DBSCAN.png", caption="Anomalieuntersuchung über DBSCAB")

    st.html("<div style='margin-bottom: 50px;'></div>")

    st.html(slide_point("Isolation Forest - Verwendet als Filter von Anomalien aus Basismodell der clo-Regression"))

    forest_left, forest_right = st.columns([1.1,1.9], vertical_alignment="center")
    with forest_left:
        st.image("Machine_Learning/images/anomalie_score_isolation_forest.png", caption="Anomaliescore")
        st.markdown("""
            * Isolation Forest sowohl für kontinuierliche, wie auch für kategorische Werte, geeignet
            * Trainset auf Anomalien geprüft
            * 342 Anomalien im Trainset gefunden (0,79%)
            * Entfernen der 342 Werte aus Trainset verbessert Vorhersagen leicht (MAE-Wert sinkt um 0,01)
            """)
    with forest_right:
        st.image("Machine_Learning/images/isolation_forest.png", caption="Anomalieuntersuchung über Isolation Forest")

    st.html("<div style='margin-bottom: 50px;'></div>")

    st.html(slide_point("Nächste mögliche Schritte"))

    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - SHAP-Analyse der Datenpunkte zur Überprüfung der Anomalien")
    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - Einfluss auf ML-Modelle testen (erste Ergebnisse zeigten leichten Einfluss mit rausgefilterten Anomaliewerten)")

    svg_path = "Machine_Learning/images/tabelle_regression_clo_anomalie.svg"

    # Datei als Text/String einlesen
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_code = f.read()

    # Nativ in Streamlit anzeigen (wichtig: width="stretch" für die volle Breite)
    st.image(svg_code, width="stretch")

    st.markdown("&nbsp;&nbsp;&nbsp;&nbsp; - gelbe und grüne Linien zeigen erste Unterschiede zwischen anomalieungefilterten und -gefilterten Rechnungen")


with tab8: # Fazit

    st.subheader("🏁Fazit")

    links, rechts = st.columns([3,1], vertical_alignment='center')
    with links:
        st.html(slide_point("<b>Subjektive Werte</b> sind im Gegensatz zu physikalischen Werten <b>oft nur schwer über Machine Learning zu bestimmen</b>."))

        st.html(slide_point("Eine Vorhersage der Werte für das thermische Empfinden ist stark geprägt von <b>Label Noise</b> und liefert <b>deshalb keine ausreichend genaue Vorhersagen</b>."))

        st.html(slide_point("Die <b>Kühlsungsstrategie konnte recht gut</b> aus den physichen Daten <b>vorhergesagt werden</b>."))

        st.html(slide_point("Eine durchgeführte Regression zur <b>Bestimmung des Kleidungsisolationswerte</b> zeigte Ergebnisse mit einer <b>durchschnittlichen Genauigkeit von etwa 0.12 clo</b>, was in etwas einer Strickjacke oder einem dünnen Shirt entspricht."))

        st.html(slide_point("Die durchgeführten Anomaliebetrachtungen zeigen auffällige Werte die noch genauer zu betrachten wären."))

        st.html(slide_point(" Ein erster Versuch zeigt kleine Verbesserungen der Vorhersagen durch Ausfiltern der auffälligen Werte."))