import streamlit as st
import base64
import os

# ---------------------------------------------------------
# Seitenkonfigurationen
# --------------------------------------------------------- 
st.set_page_config(page_title="Welcome", layout="wide",initial_sidebar_state="expanded")

# ---------------------------------------------------------
# Funktionen definieren
# ---------------------------------------------------------
# --- Funktion für Laden des Bilds ---
def load_image_as_base64(path):
    if not os.path.exists(path):
        st.error(f"❌ Imagen no encontrada: {path}")
        return None
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ---------------------------------------------------------
# Bild
# ---------------------------------------------------------
image_path = "Startseite/introduction_ashrae.png"  
image_base64 = load_image_as_base64(image_path)

if image_base64:
    ext = image_path.split(".")[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    hero_html = f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, #0A2540 0%, #1E88E5 100%);
        padding: 20px 40px;
        border-radius: 25px;
        color: white;
        min-height: 800px;
    ">
    <!-- Texto a la izquierda -->
        <div style="flex: 1.2; padding-right: 10px;">
        <p style="font-size: 1.1em; opacity: 0.85; margin-bottom: 1px;">
            Abschlussprojekt in Data Science & Data Analytics</p>

    <h1 style="font-size: 3em; font-weight: 500; margin-bottom: 10px;">
        Thermische Wahrnehmung in Innenräumen:
    </h1>

    <h1 style="font-size: 3em; font-weight: 500; margin-bottom: 10px;">
        Datenanalyse und Machine Learning Modellierung
    </h1>

    <p style="font-size: 1.15em; margin-top: 25px;">
        <strong>Team:</strong><br>
        Sabrina · Dianela · Mirtha · Daniel
    </p>
    </div>

    <!-- Imagen a la derecha (tamaño fijo) -->
    <div style="flex: 1; text-align: right;">
            <img src="data:{mime};base64,{image_base64}"
            style="
                width: 950px;              /* TAMAÑO FIJO */
                height: auto;
                border-radius: 20px;
                box-shadow: 0px 6px 20px rgba(0,0,0,0.45);
            " />
            Bild: KI-generiert
        </div>
    </div>
        """
    st.markdown(hero_html, unsafe_allow_html=True)
