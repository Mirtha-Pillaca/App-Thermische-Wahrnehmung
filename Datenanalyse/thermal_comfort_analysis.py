import streamlit as st
import pandas as pd
import pydeck as pdk
import seaborn as sns
import altair as alt
import numpy as np
import matplotlib.pyplot as plt 
from tabulate import tabulate
from PIL import Image
from sklearn.linear_model import LinearRegression
from app_projekt import df


st.set_page_config(page_title="Thermischekomfort Datenanalyse", layout="wide", initial_sidebar_state="expanded")


st.title(" 📊 Analyse der thermischen Wahrnehmung und Einflussgrößen")

# ---------------------------------------------------------
# Tabs definieren
# ---------------------------------------------------------
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊 Physikalische und subjektive Korrelationsanalyse",
    "🏢 Wichtigste Korrelationen & Gebäudebeispiele",
    "🌡️Analyse der Neutraltemperature",
    "🔧 Adaptive Strategien",
    "🌿 Komfort‑Compliance nach ASHRAE 55"
    ])


with tab1:
    
    st.subheader("📊 Physikalische und subjektive Einflussgrößen der thermischen Wahrnehmung")

    st.markdown("""
    Diese Analyse kombiniert **physikalische Messgrößen** und **subjektive Wahrnehmungen**, um ein vollständiges Bild darüber zu erhalten, wie Menschen thermische Bedingungen empfinden und welche Faktoren den Komfort am stärksten beeinflussen.
 
    """, unsafe_allow_html=True)

    # Zwei Spalten
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="
            background-color:#e6f2ff;
            padding:15px;
            border-radius:8px;
            font-size:16px;
            line-height:1.55;
        ">
        <h4>🔥 Physikalische Einflussgrößen (Innenräume) </h3>

        Diese Variablen beeinflussen direkt die Wärmeabgabe und ‑aufnahme des Körpers 
        und bilden die Grundlage für die physikalische Bewertung des thermischen Komforts.

        <ul>
            <li><b>Metabolische Aktivität</b> (wie aktiv ist eine Person) </li> 
             </br>       
            <li><b>Bekleidungsisolation</b> (Art und Dicke der Kleidung)</li>
            </br>
            <li><b>Lufttemperatur</b> (Wärme der Raumluft)</li>
            </br>
            <li><b>Luftgeschwindigkeit</b> (spürbare Luftbewegung oder Luftzug) </li>
            </br>
            <li> <b>Strahlungstemperatur</b> (Wärmeabstrahlung von Wänden, Fenstern und Oberflächen) </li>
            </br>
            <li><b>Relative Luftfeuchtigkeit</b> (Feuchtegehalt der Luft)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background-color:#e6f2ff;
            padding:15px;
            border-radius:8px;
            font-size:16px;
            line-height:1.55;
        ">
        <h4>🙂 Subjektive Einflussgrößen</h3>

        Diese Variablen beschreiben die individuelle Wahrnehmung und Bewertung der Umgebung 
        und zeigen, wie Menschen die physikalischen Bedingungen tatsächlich empfinden.
         ##### 1. Thermische Empfindung (TS) 
        **Kalt  ◄────── Neutral ──────►  Heiß**  
        `-3    -2    -1    0    +1    +2    +3 `

        ##### 2. Thermische Akzeptanz (TA)
        ○ nicht akzeptabel  
        ○ akzeptabel  

        ##### 3. Thermische Präferenz (TP)  
        **Kühler ◄──────── Keine Änderung ────────► Wärmer**  
        `  -1                         0                         +1     `

        ##### 4. Thermischer Komfort (TC, ASHRAE‑Skala 1–6)  
        **Sehr unkomfortabel ◄──────────────────────► Sehr komfortabel**  
               `  1             2            3           4           5            6   `

        </ul>
        </div>
        """, unsafe_allow_html=True)


    st.subheader("🌡️ Wie stark hängen die physikalischen Umweltvariablen tatsächlich mit den vier subjektiven Wahrnehmungsparametern zusammen?")


    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 14px;">

    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 18px; height: 18px; background-color: #e63946; border-radius: 50%;"></div>
        <b>Positive Korrelation</b> – Beide Variablen bewegen sich in die gleiche Richtung.
    </div>

    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 18px; height: 18px; background-color: #457b9d; border-radius: 50%;"></div>
        <b>Negative Korrelation</b> – Die Variablen entwickeln sich gegensätzlich.
    </div>

    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="width: 18px; height: 18px; background-color: #adb5bd; border-radius: 50%;"></div>
        <b>Nahe 0</b> – Kein relevanter Zusammenhang erkennbar.
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)



    col_left, col_right = st.columns([2, 1])

    with col_left:

        # -----------------------------------------------------------
        # 1. Variablen definieren
        # -----------------------------------------------------------
        cols_phys = [
            "metabolic_rate",
            "clothing_ensemble_insulation",
            "air_temperature",
            "air_speed",
            "radiant_temperature",
            "relative_humidity"
        ]

        cols_subj = [
            "thermal_sensation",
            "thermal_acceptability",
            "thermal_preference",
            "thermal_comfort"
        ]

        cols_all = cols_phys + cols_subj

        # -----------------------------------------------------------
        # 2. Kategorische Variablen in Zahlen umwandeln
        # -----------------------------------------------------------

        mapping_acceptability = {
            "acceptable": 1,
            "unacceptable": 0,
            "Unknown": None
        }

        mapping_preference = {
            "cooler": -1,
            "no change": 0,
            "warmer": 1,
            "Unknown": None
        }

        df["thermal_acceptability_num"] = df["thermal_acceptability"].map(mapping_acceptability)
        df["thermal_preference_num"] = df["thermal_preference"].map(mapping_preference)

        # ersetzen die alten Spalten durch die numerischen
        df["thermal_acceptability"] = df["thermal_acceptability_num"]
        df["thermal_preference"] = df["thermal_preference_num"]

        # -----------------------------------------------------------
        # 3. Heatmap erstellen
        # -----------------------------------------------------------

        df_all = df[cols_all].copy()

        # numerisch machen
        for c in cols_all:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

        # Zeilen entfernen, die komplett leer sind
        df_all = df_all.dropna(how="all")

        if df_all.empty:
            st.error("❌ Keine gültigen Daten für die Korrelationsmatrix.")
        else:
            corr_matrix = df_all.corr(method="spearman")

            # deutsche Labels
            german_all_labels = {
                "metabolic_rate": "Metabolische Aktivität",
                "clothing_ensemble_insulation": "Bekleidungsisolation",
                "air_temperature": "Lufttemperatur",
                "air_speed": "Luftgeschwindigkeit",
                "radiant_temperature": "Strahlungstemperatur",
                "relative_humidity": "Relative Luftfeuchtigkeit",
                "thermal_sensation": "Thermisches Empfinden",
                "thermal_acceptability": "Thermische Akzeptanz",
                "thermal_preference": "Thermische Präferenz",
                "thermal_comfort": "Thermischer Komfort"
            }

            corr_matrix = corr_matrix.rename(index=german_all_labels, columns=german_all_labels)

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.set_title("Korrelationsmatrix: Physikalische & subjektive Komfortvariablen")

            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                ax=ax
            )
            plt.xticks(rotation=45, ha='right')   # 🔥 X‑Labels 45° gedreht
            plt.yticks(rotation=0)   
            st.pyplot(fig)

    with col_right:
        st.markdown("""
        <div style="
            font-size:16px;
            line-height:1.55;
        ">

        <h4 style="margin-top:0;">📌 Wichtigste Ergebnisse</h4>

        <p><b>🌡️ Lufttemperatur → Strahlungstemperatur</b><br>
        <b>r = 0.89</b><br>
        Sehr starke positive Beziehung – Beide steigen gemeinsam.</p>

        <hr>

        <p><b>😊 Thermisches Empfinden → ❄️ Thermische Präferenz</b><br>
        <b>r = -0.67</b><br>
        Je wärmer empfunden, desto stärker der Wunsch nach kühleren Bedingungen.</p>

        <hr>

        <p><b>👕 Bekleidungsisolation</b><br>
        <b>r = -0.46</b> zur Lufttemperatur<br>
        <b>r = -0.45</b> zur Strahlungstemperatur<br>
        Höhere Temperaturen → Leichtere Kleidung.</p>

        <hr>

        <p><b>❄️ Lufttemperatur → Thermische Präferenz</b><br>
        <b>r = -0.44</b><br>
        Höhere Lufttemperatur führt zu einer stärkeren Präferenz für kühlere Bedingungen.</p>

        <hr>

        <p><b>☀️ Strahlungstemperatur → Thermische Präferenz</b><br>
        <b>r = -0.41</b><br>
        Warme Oberflächen erzeugen ebenfalls den Wunsch nach einer kühleren Umgebung.</p>

        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # Variablengruppen für Interpretation
    # ============================================================

    phys_vars = [
        "Metabolische Aktivität",
        "Bekleidungsisolation",
        "Lufttemperatur",
        "Luftgeschwindigkeit",
        "Strahlungstemperatur",
        "Relative Luftfeuchtigkeit"
    ]

    subj_vars = [
        "Thermisches Empfinden",
        "Thermische Akzeptanz",
        "Thermische Präferenz",
        "Thermischer Komfort"
    ]

    # ============================================================
    # Interpretationsfunktion
    # ============================================================

    def interpret(corr):
        if corr > 0.6:
            return "sehr starke positive Beziehung 🔥"
        elif corr > 0.3:
            return "moderate positive Beziehung 🙂"
        elif corr > 0.1:
            return "schwache positive Beziehung ➕"
        elif corr < -0.6:
            return "sehr starke negative Beziehung ❄️"
        elif corr < -0.3:
            return "moderate negative Beziehung 😕"
        elif corr < -0.1:
            return "schwache negative Beziehung ➖"
        else:
            return "nahezu kein Zusammenhang ⚪"


    # ============================================================
    # Korrelationen in lange Form bringen + sortieren
    # ============================================================

    corr_long = corr_matrix.stack().reset_index()
    corr_long.columns = ["Variable 1", "Variable 2", "Korrelation"]

    # Selbstkorrelationen entfernen
    corr_long = corr_long[corr_long["Variable 1"] != corr_long["Variable 2"]]

    # Doppelte Paare entfernen (A-B und B-A)
    corr_long["pair"] = corr_long.apply(
        lambda row: tuple(sorted([row["Variable 1"], row["Variable 2"]])),
        axis=1
    )
    corr_long = corr_long.drop_duplicates(subset="pair")

    # Sortieren nach Stärke (absoluter Wert)
    corr_sorted = corr_long.sort_values(
        by="Korrelation",
        key=lambda x: abs(x),
        ascending=False
    )


    # ============================================================
    # EXPANDER 1 — Physikalische Zusammenhänge
    # ============================================================

    with st.expander("🌡️ Physikalische Zusammenhänge"):
        phys_corr = corr_sorted[
            corr_sorted["Variable 1"].isin(phys_vars) &
            corr_sorted["Variable 2"].isin(phys_vars)
        ]

        if phys_corr.empty:
            st.write("Keine physikalischen Zusammenhänge gefunden.")
        else:
            for _, row in phys_corr.iterrows():
                st.markdown(
                    f"- **{row['Variable 1']} ↔ {row['Variable 2']}**: "
                    f"{interpret(row['Korrelation'])} "
                    f"(ρ = {row['Korrelation']:.2f})"
                )

    # ============================================================
    # EXPANDER 2 — Subjektive Zusammenhänge
    # ============================================================

    with st.expander("🙂 Subjektive Zusammenhänge"):
        subj_corr = corr_sorted[
            corr_sorted["Variable 1"].isin(subj_vars) &
            corr_sorted["Variable 2"].isin(subj_vars)
        ]

        if subj_corr.empty:
            st.write("Keine subjektiven Zusammenhänge gefunden.")
        else:
            for _, row in subj_corr.iterrows():
                st.markdown(
                    f"- **{row['Variable 1']} ↔ {row['Variable 2']}**: "
                    f"{interpret(row['Korrelation'])} "
                    f"(ρ = {row['Korrelation']:.2f})"
                )

    # ============================================================
    # EXPANDER 3 — Physikalisch ↔ Subjektiv (Cross-Korrelationen)
    # ============================================================

    with st.expander("🔄 Physikalisch ↔ Subjektiv"):
        cross_corr = corr_sorted[
            (corr_sorted["Variable 1"].isin(phys_vars) & corr_sorted["Variable 2"].isin(subj_vars)) |
            (corr_sorted["Variable 1"].isin(subj_vars) & corr_sorted["Variable 2"].isin(phys_vars))
        ]

        if cross_corr.empty:
            st.write("Keine Beziehungen zwischen physikalischen und subjektiven Variablen gefunden.")
        else:
            for _, row in cross_corr.iterrows():
                st.markdown(
                    f"- **{row['Variable 1']} ↔ {row['Variable 2']}**: "
                    f"{interpret(row['Korrelation'])} "
                    f"(ρ = {row['Korrelation']:.2f})"
                )


with tab2:
    
    st.subheader("🏠 Wichtigste Korrelationen mit Gebäudebeispielen")

    st.markdown("""
    Diese Übersicht zeigt, wie physikalische Komfortparameter (z.B. Lufttemperatur,
    Strahlungstemperatur und Bekleidungsisolation) mit dem Verhalten und Empfinden
    von Personen in verschiedenen Gebäudetypen zusammenhängen.
    """)



    st.markdown("""
    <style>
    .box {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 18px;
        border: 1px solid #e3e6eb;
    }
    .title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .text {
        font-size: 18px;
        line-height: 1.55;
    }
    </style>

    <div class="box">
        <div class="title">🏢 Bürogebäude</div>
        <div class="text">
            • Große Glasflächen können die Luft- und Strahlungstemperatur erhöhen.<br>
            • Deshalb werden häufig kühlere Raumtemperaturen bevorzugt.
        </div>
    </div>

    <div class="box">
        <div class="title">🏠 Mehrfamilienhäuser</div>
        <div class="text">
            • Die Bekleidungsisolation passt sich oft an die Innentemperatur an.<br>
            • Sonneneinstrahlung kann die thermische Wahrnehmung beeinflussen.
        </div>
    </div>

    <div class="box">
        <div class="title">👩‍🏫 Klassenräume</div>
        <div class="text">
            • Eine hohe Personenanzahl erhöht die Wärmebelastung.<br>
            • Deshalb werden häufig Fenster geöffnet oder Ventilatoren genutzt.
        </div>
    </div>

    <div class="box">
        <div class="title">👴 Seniorenzentren</div>
        <div class="text">
            • Das thermische Empfinden spielt eine wichtige Rolle für die thermische Präferenz.<br>
            • Deshalb sind stabile und angenehme Raumtemperaturen besonders wichtig.
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    <div style="font-size:22px; line-height:1.6;">
    <b>Die Temperatur, sowohl Luft- als auch Strahlungstemperatur, ist in allen Gebäudetypen
        der wichtigste Einflussfaktor auf die thermische Wahrnehmung.<b>
    </div>
    """, unsafe_allow_html=True)



with tab3:

    st.markdown("""
    ### 🧊🌡️ Was bedeutet die Neutraltemperatur?

    Die Neutraltemperatur ist eine zentrale Kenngröße der thermischen Wahrnehmungsanalyse. Sie beschreibt die operative Raumtemperatur, bei der Personen weder Wärme‑ noch Kälteempfindungen angeben und sich in einem thermisch neutralen Zustand befinden.
    
    ---  """)

    col_text1, col_text2 = st.columns([1, 1])

    with col_text1:
        st.markdown("""

        #### 🧭 Verständnishilfe (ASCII‑Grafik)

                 zu kalt            neutral         zu warm
                (MTS < 0)          (MTS = 0)       (MTS > 0)
                     \\                |               /
                      \\_______________|______________/
                                  T_neutral

        - **Links:** Personen empfinden die Temperatur als *zu kalt* (negative MTS‑Werte).  
        - **Rechts:** Personen empfinden die Temperatur als *zu warm* (positive MTS‑Werte).  
        - **Mitte:** Die **Neutraltemperatur T_neutral**, bei der die mittlere thermische Empfindung **MTS = 0** ist.""")

    with col_text2:
        st.markdown("""
        #### 📈 Zusammenhang zwischen Temperatur und Wahrnehmung

        Um die Neutraltemperatur zu bestimmen, wird die Beziehung zwischen  
        **operativer Temperatur** und **mittlerer thermischer Empfindung (MTS)** mittels  
        **linearer Regression** modelliert.

        - Eine **höhere Neutraltemperatur** zeigt, dass Personen tendenziell **wärmere Bedingungen bevorzugen**.  
        - Eine **niedrigere Neutraltemperatur** weist auf eine **Präferenz für kühlere Bedingungen** hin.
        """)

    # ---------------------------------------------------------
    # ZWEI SPALTEN: LINKS FILTER, RECHTS HAUPTFIGUR
    # ---------------------------------------------------------
    col_filters, col_main = st.columns([1, 2])

    # -----------------------------
    # FILTER + GRUPPIERUNG (LINKE SPALTE)
    # -----------------------------
    with col_filters:
        st.markdown("### 🔍 Filter für diese Analyse")

        region = st.multiselect("Region", sorted(df["region"].dropna().unique()))
        season = st.multiselect("Jahreszeit", sorted(df["season"].dropna().unique()))
        gender = st.multiselect("Geschlecht", sorted(df["gender"].dropna().unique()))
        building_type = st.multiselect("Gebäudetyp", sorted(df["building_type"].dropna().unique()))
        climate = st.multiselect("Klima", sorted(df["climate"].dropna().unique()))
        cooling_type = st.multiselect("Kühlungsart", sorted(df["cooling_type"].dropna().unique()))

        st.markdown("### 🧩 Gruppierung der Analyse")
        grouping_options = [
            "Keine Gruppierung", "season", "climate", "building_type",
            "cooling_type", "gender", "age", "country", "region"
        ]
        grouping_choice = st.selectbox("Gruppierung der Analyse", grouping_options)



    # -----------------------------
    # FILTER AUF DATEN ANWENDEN
    # -----------------------------
    df_filtered = df.copy()

    if region:
        df_filtered = df_filtered[df_filtered["region"].isin(region)]
    if season:
        df_filtered = df_filtered[df_filtered["season"].isin(season)]
    if gender:
        df_filtered = df_filtered[df_filtered["gender"].isin(gender)]
    if building_type:
        df_filtered = df_filtered[df_filtered["building_type"].isin(building_type)]
    if climate:
        df_filtered = df_filtered[df_filtered["climate"].isin(climate)]
    if cooling_type:
        df_filtered = df_filtered[df_filtered["cooling_type"].isin(cooling_type)]

    # ---------------------------------------------------------
    # Altersklassifizierung (nur wenn age gewählt wird)
    # ---------------------------------------------------------
    if grouping_choice == "age":

        def classify_age(x):
            try:
                x = float(x)
            except:
                return None

            if x < 18:
                return None  # Kinder ignorieren
            elif 18 <= x <= 35:
                return "Jung (18–35)"
            elif 36 <= x <= 55:
                return "Mittel (36–55)"
            else:
                return "Älter (56+)"

        # Neue Spalte erzeugen
        df_filtered["age_group"] = df_filtered["age"].apply(classify_age)

        # Gruppierung ersetzen
        grouping_choice = "age_group"

    if df_filtered.empty:
        col_main.warning("Keine Daten für die ausgewählten Filter.")
        st.stop()

    # ---------------------------------------------------------
    # HAUPTFIGUR (RECHTE SPALTE)
    # ---------------------------------------------------------
    mts_df_main = (
        df_filtered.groupby("operative_temperature")["thermal_sensation"]
        .mean()
        .reset_index()
        .dropna()
    )

    if len(mts_df_main) < 2:
        col_main.warning("Nicht genügend Daten für die Hauptanalyse.")
    else:
        X_main = mts_df_main["operative_temperature"].values.reshape(-1, 1)
        y_main = mts_df_main["thermal_sensation"].values.reshape(-1, 1)

        model_main = LinearRegression()
        model_main.fit(X_main, y_main)

        a_main = model_main.coef_[0][0]
        b_main = model_main.intercept_[0]
        neutral_temp_main = -b_main / a_main

        fig_main, ax_main = plt.subplots(figsize=(10, 5))

        x_range_main = np.linspace(X_main.min(), X_main.max(), 100)
        y_pred_main = a_main * x_range_main + b_main

        # Farben wie gewünscht
        ax_main.scatter(X_main, y_main, color="blue", alpha=0.7)     # Punkte blau
        ax_main.plot(x_range_main, y_pred_main, color="red", linewidth=2)  # Linie rot
        ax_main.axvline(neutral_temp_main, color="green", linestyle="--", linewidth=2)  # Neutraltemp grün
        ax_main.scatter(X_main, y_main, color="blue", alpha=0.7, label="Mittelwert (MTS)")
        ax_main.plot(x_range_main, y_pred_main, color="red", linewidth=2, label="Lineare Regression")
        ax_main.axvline(neutral_temp_main, color="green", linestyle="--", linewidth=2, label="Neutraltemperatur")
        ax_main.legend(loc="upper left")

        ax_main.set_xlabel("Operative Temperatur (°C)")
        ax_main.set_ylabel("Mittlere Thermische Empfindung (MTS)")
        ax_main.set_title("Neutraltemperatur – Gesamtanalyse")
        ax_main.grid(True)

        col_main.pyplot(fig_main)
        col_main.success(f"**Neutraltemperatur (gesamt): {neutral_temp_main:.2f} °C**")

    # ---------------------------------------------------------
    # GRUPPIERTE PLOTS UNTERHALB DER HAUPTFIGUR (MAX. 3 PRO REIHE)
    # ---------------------------------------------------------
    if grouping_choice != "Keine Gruppierung":

        st.markdown("### 🔎 Neutraltemperatur nach Gruppen")

        groups = df_filtered[grouping_choice].dropna().unique()
        groups = [
            g for g in groups
            if str(g).lower() not in ["unknown", "unk", "none", "nan", ""]
        ]

        if len(groups) == 0:
            st.warning("Keine gültigen Gruppen für die gewählte Gruppierung.")
        else:
            cols_per_row = 3
            rows = [groups[i:i + cols_per_row] for i in range(0, len(groups), cols_per_row)]

            for row_groups in rows:
                row_cols = st.columns(cols_per_row)

                for col, g in zip(row_cols, row_groups):

                    sub = df_filtered[df_filtered[grouping_choice] == g]

                    mts_df = (
                        sub.groupby("operative_temperature")["thermal_sensation"]
                        .mean()
                        .reset_index()
                        .dropna()
                    )

                    if len(mts_df) < 2:
                        col.warning(f"Nicht genügend Daten für Gruppe {g}.")
                        continue

                    X = mts_df["operative_temperature"].values.reshape(-1, 1)
                    y = mts_df["thermal_sensation"].values.reshape(-1, 1)

                    model = LinearRegression()
                    model.fit(X, y)

                    a = model.coef_[0][0]
                    b = model.intercept_[0]
                    neutral_temp = -b / a

                    fig, ax = plt.subplots(figsize=(5, 4))

                    x_range = np.linspace(X.min(), X.max(), 100)
                    y_pred = a * x_range + b

                    # Farben wie gewünscht
                    ax.scatter(X, y, color="blue", alpha=0.7)     # Punkte blau
                    ax.plot(x_range, y_pred, color="red", linewidth=2)  # Linie rot
                    ax.axvline(neutral_temp, color="green", linestyle="--", linewidth=2)  # Neutraltemp grün

                    ax.set_xlabel("Operative Temperatur (°C)")
                    ax.set_ylabel("Mittlere Thermische Empfindung (MTS)")
                    ax.set_title(f"Neutraltemperatur – Gruppe: {g}")
                    ax.grid(True)

                    col.pyplot(fig)
                    col.caption(f"Neutraltemperatur: **{neutral_temp:.2f} °C**")


    st.info("""
    ### ℹ️ Zusammenfassung

    Die Neutraltemperatur beschreibt jene operative Raumtemperatur, bei der die mittlere thermische 
    Empfindung (MTS) ausgeglichen ist und weder als warm noch als kalt wahrgenommen wird.

    **Zentrale Erkenntnisse:**
    - Sie dient als belastbarer Referenzwert zur Bewertung des thermischen Komforts.  
    - Höhere Neutraltemperaturen weisen auf eine Präferenz für wärmere Innenraumtemperaturen hin, 
    niedrigere Werte auf kühlere Komfortansprüche.  
    - Einflussgrößen wie Klima, Gebäudetyp, Jahreszeit oder demografische Merkmale führen zu 
    systematischen Unterschieden im thermischen Empfinden.  
    - Gruppierte Analysen zeigen, dass Komfortpräferenzen je nach Kontext und Nutzerprofil variieren.

    **Praktische Relevanz:**
    Die Ergebnisse unterstützen eine evidenzbasierte Anpassung von Innenraumtemperaturen und 
    ermöglichen komfortorientierte Entscheidungen für spezifische Nutzergruppen und Umgebungen.
    """)


with tab4:

    st.subheader("Wie reagieren verschiedene Gruppen auf ihre thermische Wahrnehmung anhand ihrer adaptiven Strategien?")

    st.markdown("""
    Adaptive Komfortstrategien beschreiben, wie Menschen **aktiv** auf ihr **Raumklima** reagieren – 
    etwa durch das **Öffnen von Fenstern**, das **Nutzen von Ventilatoren**, das **Einschalten der Heizung** 
    oder das **Schließen von Türen und Jalousien**.

    Diese Verhaltensweisen stehen in engem Zusammenhang mit der **thermischen Wahrnehmung** und den 
    zentralen **Einflussgrößen**, die im vorherigen Abschnitt analysiert wurden. Faktoren wie 
    **operative Temperatur**, **Luftbewegung**, **Strahlungswärme**, **Kleidung**, 
    **Aktivitätsniveau** sowie **klimatische** und **saisonale Bedingungen** prägen maßgeblich, 
    wie Menschen ihr Umfeld empfinden und welche **adaptiven Strategien** sie einsetzen, um 
    **thermischen Komfort** wiederherzustellen.

    ---
    """)

    st.subheader("📊 Überblick über adaptive Strategien in den ausgewählten Gruppen")

    st.markdown("""
    Die folgenden Heatmaps zeigen, wie häufig diese Strategien in den ausgewählten **Gruppen** 
    angewendet werden. Dadurch lassen sich **Muster** erkennen, die direkt mit den zuvor 
    untersuchten **Einflussgrößen der thermischen Wahrnehmung** verknüpft sind und verdeutlichen, 
    wie unterschiedliche Gruppen auf **thermische Situationen** reagieren.

    """)


    # ---------------------------------------------------------
    # ADAPTIVE VARIABLEN
    # ---------------------------------------------------------
    adaptive_vars = ["window", "door", "fan", "heater", "blind_curtain"]

    for col in adaptive_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---------------------------------------------------------
    # NUR GRUPPIERUNG (KEINE FILTER)
    # ---------------------------------------------------------
    #st.markdown("#### 📊 Gruppierung der Analyse")

    group_param = st.selectbox(
        "Gruppieren der Analyse nach:",
        ["region", "country", "city", "season", "climate_zone", "building_type", "cooling_type", "gender"]
    )

    # ---------------------------------------------------------
    # GRUPPEN ERMITTELN
    # ---------------------------------------------------------
    groups_available = df[group_param].dropna().unique()
    groups_available = [g for g in groups_available if str(g).lower() != "unknown"]

    groups_clean = []
    for g in groups_available:
        df_g = df[df[group_param] == g]
        if df_g[adaptive_vars].mean().sum() > 0:
            groups_clean.append(g)

    # ---------------------------------------------------------
    # HEATMAPS – 3 PRO FIGUR, TITEL IM PLOT
    # ---------------------------------------------------------
    if len(groups_clean) == 0:
        st.warning("Keine Daten für adaptive Strategien in dieser Gruppierung.")
    else:
        for i in range(0, len(groups_clean), 3):

            subset = groups_clean[i:i+3]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # überflüssige Achsen deaktivieren
            for ax in axes[len(subset):]:
                ax.axis("off")

            # Heatmaps zeichnen
            for ax, g in zip(axes, subset):
                df_g = df[df[group_param] == g]
                mean_vals = df_g[adaptive_vars].mean() * 100

                heatmap_df = pd.DataFrame(mean_vals, columns=["Nutzung (%)"])

                sns.heatmap(
                    heatmap_df,
                    annot=True,
                    cmap="coolwarm",
                    cbar=False,
                    fmt=".1f",
                    ax=ax
                )

                ax.set_ylabel("Adaptive Strategien")
                ax.set_xlabel("")

                # TITEL IM PLOT
                ax.set_title(str(g), fontsize=14)

            st.pyplot(fig)

    # ---------------------------------------------------------
    # CAPTION
    # ---------------------------------------------------------
    st.caption("Blau = geringe Nutzung, Rot = hohe Nutzung der Strategien.")

    # ---------------------------------------------------------
    # EXPANDER
    # ---------------------------------------------------------
    with st.expander("🔍 Wichtigste Einflussvariable"):

        st.markdown("""
        **season** → Kleidung ändert sich stark zwischen Sommer / Winter  
        **climate_zone** → Klimazone bestimmt typische Kleidung  
        **gender** → Geschlechtsspezifische Kleidungsgewohnheiten  
        **building_type** → Innenraumumgebung beeinflusst CLO  
        **cooling_type** → AC / natürliche Lüftung beeinflusst Kleidung  
        **metabolic_rate** → Aktivitätsniveau bestimmt Wärmeproduktion  
        **operative_temperature** → Innenraumtemperatur beeinflusst Kleidung  
        **air_temperature** → Außentemperatur beeinflusst Kleidung  
        **radiant_temperature** → Strahlungswärme beeinflusst Komfort  
        **age** → Altersbedingte Unterschiede im Wärmeempfinden

        ---
        ### 🧭 Warum unterscheiden sich die Länder?

        **Klima <--> Kleidung**  
        Heiße Länder → Temperatur dominiert  
        Kalte Länder → Saison dominiert  

        **Gebäude <--> Innenraumklima**  
        Starke Klimaanlagen → operative_temperature ↑  
        Natürliche Lüftung → air_temperature ↑  

        **Kultur <--> Kleidung**  
        Strenge Kleidungsnormen → gender / building_type ↑  

        **Aktivität <--> Wärmeproduktion**  
        Hohe körperliche Aktivität → metabolic_rate ↑  

        **Strahlung <--> Komfort**  
        Starke Sonneneinstrahlung → radiant_temperature ↑  

        Diese Unterschiede sind **normal**:  
        Jedes Land hat **eigenes Klima**, **eigene Gebäude**, **eigene Kultur** und **eigene Datenverteilung**.  
        Darum zeigt die Statistik **verschiedene dominante Variablen**.
        """)


with tab5:

    # ---------------------------------------------------------
    # TITEL UND BESCHREIBUNG
    # ---------------------------------------------------------
    st.header("🌿 Wie gut passen sich verschiedene Gruppen an das adaptive Komfortmodell nach ASHRAE 55 an?")

    st.markdown("""
    Das adaptive Komfortmodell nach **ASHRAE 55** beschreibt, wie sich Menschen an das 
    **Außenklima** anpassen und welche **operativen Innentemperaturen** sie unter 
    verschiedenen Bedingungen akzeptieren. Diese Analyse zeigt, wie gut die gefilterten 
    Daten die Anforderungen des Modells erfüllen und **in welchen Kategorien deutliche 
    Abweichungen vom Komfortbereich auftreten**.

    Das Modell basiert auf der Annahme, dass sich die **thermische Wahrnehmung** dynamisch 
    an die Außentemperatur anpasst:

    - Bei **wärmeren Außentemperaturen** akzeptieren Menschen höhere Innentemperaturen.
    - Bei **kühleren Außentemperaturen** bevorzugen sie niedrigere Innentemperaturen.

    **Die Komfortgleichung lautet:**

    """)

    st.latex(r"T_{\text{comf}} = 0.31 \cdot T_{\text{out}} + 17.8")

    st.markdown("""
    Auf Basis dieser Gleichung ergeben sich die **80 %‑Komfortzone** und die 
    **90 %‑Komfortzone**, die anzeigen, in welchem Temperaturbereich die Mehrheit der 
    Personen **thermischen Komfort** empfindet.

    """)

    # ---------------------------------------------------------
    # ZWEI SPALTEN: LINKS FILTER + GRUPPIERUNG, RECHTS FIGUR
    # ---------------------------------------------------------
    col_left, col_right = st.columns([1, 2])

    # ---------------------------------------------------------
    # FILTER (LINKE SPALTE)
    # ---------------------------------------------------------
    with col_left:
        st.markdown("### 🔍 Filter")

        region = st.multiselect("Region", sorted(df["region"].dropna().unique()), key="t5_region")
        season = st.multiselect("Jahreszeit", sorted(df["season"].dropna().unique()), key="t5_season")
        gender = st.multiselect("Geschlecht", sorted(df["gender"].dropna().unique()), key="t5_gender")
        building_type = st.multiselect("Gebäudetyp", sorted(df["building_type"].dropna().unique()), key="t5_building")
        climate = st.multiselect("Klimazone", sorted(df["climate"].dropna().unique()), key="t5_climate")
        cooling_type = st.multiselect("Kühlungsart", sorted(df["cooling_type"].dropna().unique()), key="t5_cooling")

        st.markdown("### 🧩 Gruppierung der Analyse")
        grouping_choice = st.selectbox(
            "Gruppieren nach:",
            ["Keine Gruppierung", "season", "climate", "building_type",
             "cooling_type", "gender", "age", "region"],
            key="t5_grouping"
        )

    # ---------------------------------------------------------
    # FILTER AUF DATEN ANWENDEN
    # ---------------------------------------------------------
    df_filtered = df.copy()

    if region:
        df_filtered = df_filtered[df_filtered["region"].isin(region)]
    if season:
        df_filtered = df_filtered[df_filtered["season"].isin(season)]
    if gender:
        df_filtered = df_filtered[df_filtered["gender"].isin(gender)]
    if building_type:
        df_filtered = df_filtered[df_filtered["building_type"].isin(building_type)]
    if climate:
        df_filtered = df_filtered[df_filtered["climate"].isin(climate)]
    if cooling_type:
        df_filtered = df_filtered[df_filtered["cooling_type"].isin(cooling_type)]


    # ---------------------------------------------------------
    # Altersklassifizierung (nur wenn age gewählt wird)
    # ---------------------------------------------------------
    if grouping_choice == "age":

        def classify_age(x):
            try:
                x = float(x)
            except:
                return None

            if x < 18:
                return None  # Kinder ignorieren
            elif 18 <= x <= 35:
                return "Jung (18–35)"
            elif 36 <= x <= 55:
                return "Mittel (36–55)"
            else:
                return "Älter (56+)"

        # Neue Spalte erzeugen
        df_filtered["age_group"] = df_filtered["age"].apply(classify_age)

        # Gruppierung ersetzen
        grouping_choice = "age_group"

    if df_filtered.empty:
        col_right.warning("Keine Daten für die ausgewählten Filter.")
        st.stop()

    # ---------------------------------------------------------
    # ADAPTIVE KOMFORTBERECHNUNG
    # ---------------------------------------------------------
    df_sorted = df_filtered.sort_values(by="outdoor_air_temperature")

    T_out = df_sorted["outdoor_air_temperature"]
    T_in = df_sorted["operative_temperature"]

    T_comf = 0.31 * T_out + 17.8
    T_lower_80 = T_comf - 2.5
    T_upper_80 = T_comf + 2.5
    T_lower_90 = T_comf - 3.5
    T_upper_90 = T_comf + 3.5

    # ---------------------------------------------------------
    # AKTIVE FILTERTEXT DARSTELLUNG
    # ---------------------------------------------------------
    active_filters = {
        "Region": region,
        "Jahreszeit": season,
        "Geschlecht": gender,
        "Gebäudetyp": building_type,
        "Klimazone": climate,
        "Kühlungsart": cooling_type
    }

    filter_text = ", ".join(
        f"{k}: {', '.join(v)}" for k, v in active_filters.items() if v
    )

    # ---------------------------------------------------------
    # HAUPTFIGUR (RECHTE SPALTE)
    # ---------------------------------------------------------
    with col_right:
        st.subheader("Adaptive Comfort Chart")
        st.caption(f"Aktive Filter: {filter_text if filter_text else 'Keine Filter aktiv'}")

        fig, ax = plt.subplots(figsize=(12, 7))

        ax.fill_between(T_out, T_lower_90, T_upper_90, color="yellow", alpha=0.15, label="90 % Komfortzone")
        ax.fill_between(T_out, T_lower_80, T_upper_80, color="green", alpha=0.20, label="80 % Komfortzone")

        ax.scatter(
            df_sorted["outdoor_air_temperature"],
            df_sorted["operative_temperature"],
            color="blue",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            label="Messpunkte"
        )

        ax.set_xlabel("Außentemperatur (°C)")
        ax.set_ylabel("Operative Innentemperatur (°C)")
        ax.set_title("ASHRAE 55 – Adaptives Komfortmodell")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

        st.caption(
            "Die farbigen Bereiche markieren die 80 %‑ und 90 %‑Komfortzonen. "
            "Die Punkte zeigen die tatsächlichen Messwerte und verdeutlichen, "
            "wie gut die Daten mit dem adaptiven Modell übereinstimmen."
        )

    # ---------------------------------------------------------
    # GRUPPIERTE PLOTS (WIE TAB 3)
    # ---------------------------------------------------------
    if grouping_choice != "Keine Gruppierung":

        st.markdown("### 🔎 Adaptive Komfortanalyse nach Gruppen")

        groups = df_filtered[grouping_choice].dropna().unique()
        groups = [g for g in groups if str(g).lower() not in ["unknown", "unk", "none", "nan", ""]]

        if len(groups) == 0:
            st.warning("Keine gültigen Gruppen für die gewählte Gruppierung.")
        else:
            cols_per_row = 3
            rows = [groups[i:i + cols_per_row] for i in range(0, len(groups), cols_per_row)]

            for row_groups in rows:
                row_cols = st.columns(cols_per_row)

                for col, g in zip(row_cols, row_groups):

                    sub = df_filtered[df_filtered[grouping_choice] == g]

                    df_sorted_g = sub.sort_values(by="outdoor_air_temperature")

                    if df_sorted_g.empty:
                        col.warning(f"Keine Daten für Gruppe {g}.")
                        continue

                    T_out_g = df_sorted_g["outdoor_air_temperature"]
                    T_in_g = df_sorted_g["operative_temperature"]

                    T_comf_g = 0.31 * T_out_g + 17.8
                    T_lower_80_g = T_comf_g - 2.5
                    T_upper_80_g = T_comf_g + 2.5
                    T_lower_90_g = T_comf_g - 3.5
                    T_upper_90_g = T_comf_g + 3.5

                    fig_g, ax_g = plt.subplots(figsize=(5, 4))

                    ax_g.fill_between(T_out_g, T_lower_90_g, T_upper_90_g, color="yellow", alpha=0.15)
                    ax_g.fill_between(T_out_g, T_lower_80_g, T_upper_80_g, color="green", alpha=0.20)

                    ax_g.scatter(
                        T_out_g,
                        T_in_g,
                        color="blue",
                        alpha=0.7,
                        edgecolor="black",
                        linewidth=0.5
                    )

                    ax_g.set_xlabel("Außentemperatur (°C)")
                    ax_g.set_ylabel("Operative Innentemperatur (°C)")
                    ax_g.set_title(f"Gruppe: {g}")
                    ax_g.grid(True)

                    col.pyplot(fig_g)

    st.info("""
    ### 📌 Zusammenfassung

    Die Analyse zeigt, wie gut sich die untersuchten Gruppen an das **adaptive Komfortmodell 
    nach ASHRAE 55** anpassen. Die Komfortzonen (80 % und 90 %) geben Bereiche an, in denen die 
    Mehrheit der Personen thermischen Komfort empfindet. 

    - **Messpunkte innerhalb der Komfortzonen** weisen auf eine gute Anpassung an das 
    Außenklima hin.
    - **Messpunkte oberhalb der Komfortzonen** deuten auf mögliche **Überhitzung** oder 
    unzureichende Kühlung hin.
    - **Messpunkte unterhalb der Komfortzonen** weisen auf **Unterkühlung**, erhöhte 
    Luftbewegung oder ineffiziente Heizstrategien hin.
    - Die **Streuung der Punkte** zeigt die Auswirkung von unterschiedlichen Gebäudetypen, Klimazonen oder 
    Nutzungsarten auf die Außentemperatur.

    Durch die Gruppierung wird sichtbar, **welche Kategorien besonders gut oder schlecht 
    mit dem adaptiven Modell übereinstimmen** und wo potenzieller Optimierungsbedarf besteht.
    """)
