# App Thermische Wahrnehmung basierend auf ASHRAE-Daten


## 🔍 Projektbeschreibung

### Thermische Wahrnehmung in Innenräumen: Datenanalyse und Machine Learning Modellierung

In diesem Projekt wurde eine interaktive Streamlit application entwickelt, um Daten zur thermischen Wahrnehmung in Innenräumen basierend auf der ASHRAE Global Thermal Comfort Database II zu analysieren und zu visualisieren.
Diese App soll Wissenschaftler:innen, Studierende und Analyst:innen darin unterstützen, Umgebungsbedingungen, Komfortbewertungen und adaptives Verhalten über verschiedene Klima- und Gebäudetypen hinweg zu verstehen.   


---

## Datenquelle

- **ASHRAE Global Thermal Comfort Database v2.1**
- Link: https://datadryad.org/dataset/doi:10.6078/D1F671 (Stand 16.06.2026)

- Umfassende Datenbank zur Untersuchung des thermischen Komforts in Gebäuden weltweit
- Zusammenstellung von Feldstudien aus dem Zeitraum 1995–2016


## 🔍 Projektinhalte

- 📂 Hochladen, zusammenführen und bearbeiten der Tabellen metadata und measurements (CSV)  
- 📊 Datenanalyse und interaktive Visualisierungen für:
  - Klimatische/geografische Variablen
  - Belüftungsart 
  - Physikalische Parameter
- 🤖 Machine Learning  
- 🧭 Filterung u.a. nach Belüftungsart, Klimatypen, Alter etc.
- 🧮 Integration mit Neon oder einer anderen Datenbank möglich 

---

## 🎯 Projektziel

- Das Projekt soll zeigen, wie subjektive thermische Wahrnehmung und physikalische Parameter zusammenwirken und welche globalen Muster sich in großen Datensätzen erkennen lassen
- Die Streamlit‑App dient als interaktive Plattform, um diese Erkenntnisse verständlich und zugänglich zu machen


---

## 🛠️ Verwendete Tools

- **Python**  
- **Streamlit**  
- **Pandas**  
- **Plotly / Matplotlib**  
- **NumPy**  
- **PostgreSQL Neon**  
- **Power BI**
- **Jupyter Notebook**
- **scikit-learn**

---

## 📁 Projektstruktur

```text
ASHRAE_Thermal_Comfort_APP/
│
├── app_projekt.py
├── startseite/
├── einfuehrung/
├── datenbereinigung/
├── datenbank/
├── datenanalyse/
├── machine_learning/
├── dashboard/
├── zusammenfassung_fazit/
├── daten/
├── requirements.txt
└── Readme.md
```

---

## 🚀 Starten der App

1. Installieren der notwendigen Bibliotheken:

pip install -r requirements.txt

2. Starten der Streamlit app:

streamlit run app_projekt.py

---

## 👥 Team

Dieses Projekt wurde im Rahmen der beruflichen Weiterbildung in Data Analytics bzw. Data Science des Data Science Institute DSI Education GmbH entwickelt und wurde in einem interdisziplinären Team durchgeführt:

- Sabrina Hüschenbett     | Data Analyst
- Dianela Mujica          | Data Analyst  
- Mirtha Pillaca Quispe   | Data Scientist  
- Daniel-Jan Schendel     | Data Scientist


---


## Projektzeitraum

06.07. - 24.07.2026


---


## 📄 Lizenz

Dieses Projekt ist privat und ist nur für Bildungszwecke gedacht.

Lizenzen:
- Code: MIT License (see LICENSE)
- Data: Bitte die Originallizenz und Nutzungsbedingungen der Datenquelle beachten.
