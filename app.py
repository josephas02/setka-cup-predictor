import datetime
import glob
import json
import os
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Configuración de la página
st.set_page_config(
    page_title="Setka Cup Predictor Pro", page_icon="🏓", layout="centered"
)


@st.cache_resource
def cargar_modelo_y_datos():
  # Buscar archivos zip del dump completo
  archivos_zip = glob.glob("setka_players_dump_full_*.zip")
  if not archivos_zip:
    archivos_zip = glob.glob("setka_players_dump_*.zip")
  if not archivos_zip:
    archivos_zip = glob.glob("**/setka_players_dump_*.zip", recursive=True)

  jugadores = []

  if archivos_zip:
    zip_path = max(archivos_zip, key=os.path.getmtime)
    st.info(f"[*] Cargando base de datos desde {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
      json_names = [name for name in z.namelist() if name.endswith(".json")]
      if json_names:
        with z.open(json_names[0]) as f:
          jugadores = json.load(f)
  else:
    archivos_json = glob.glob("setka_players_dump_*.json")
    if archivos_json:
      archivo_reciente = max(archivos_json, key=os.path.getmtime)
      with open(archivo_reciente, "r", encoding="utf-8") as f:
        jugadores = json.load(f)

  if not jugadores:
    return None, []

  # Entrenamiento del modelo
  features_list = [
      "diff_sc",
      "diff_uttf",
      "diff_win_rate",
      "diff_set_ratio",
      "diff_momentum",
      "diff_fatigue",
      "diff_puntos",
      "diff_exp",
  ]

  filas = []
  limite_entreno = min(len(jugadores), 300)
  for i in range(limite_entreno):
    for j in range(i + 1, min(i + 5, limite_entreno)):
      f1 = extraer_features(jugadores[i])
      f2 = extraer_features(jugadores[j])

      if f1["rating_sc"] == 0 or f2["rating_sc"] == 0:
        continue

      filas.append({
          "diff_sc": f1["rating_sc"] - f2["rating_sc"],
          "diff_uttf": f1["rating_uttf"] - f2["rating_uttf"],
          "diff_win_rate": f1["win_rate"] - f2["win_rate"],
          "diff_set_ratio": f1["set_ratio"] - f2["set_ratio"],
          "diff_momentum": f1["momentum"] - f2["momentum"],
          "diff_fatigue": f1["fatigue"] - f2["fatigue"],
          "diff_puntos": f1["promedio_puntos"] - f2["promedio_puntos"],
          "diff_exp": f1["experiencia"] - f2["experiencia"],
          "target": 1 if f1["rating_sc"] >= f2["rating_sc"] else 0,
      })

  df = pd.DataFrame(filas)
  model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
  if not df.empty:
    model.fit(df[features_list], df["target"])

  return model, jugadores


def extraer_features(j):
  stats = j.get("tournamentStats", [])
  total_w = sum(s.get("matchWin", 0) for s in stats)
  total_l = sum(s.get("matchLost", 0) for s in stats)
  total_partidos = total_w + total_l
  win_rate = (total_w / total_partidos) if total_partidos > 0 else 0.5

  total_sw = sum(s.get("setWin", 0) for s in stats)
  total_sl = sum(s.get("setLost", 0) for s in stats)
  set_ratio = (
      (total_sw / (total_sw + total_sl)) if (total_sw + total_sl) > 0 else 0.5
  )

  ultimos_torneos = stats[:3]
  w_reciente = sum(s.get("matchWin", 0) for s in ultimos_torneos)
  l_reciente = sum(s.get("matchLost", 0) for s in ultimos_torneos)
  momentum = (
      (w_reciente / (w_reciente + l_reciente))
      if (w_reciente + l_reciente) > 0
      else win_rate
  )

  partidos_ult = (
      (ultimos_torneos[0].get("matchWin", 0) + ultimos_torneos[0].get("matchLost", 0))
      if ultimos_torneos
      else 0
  )
  fatigue = partidos_ult * 2
  puntos = np.mean([s.get("points", 0) for s in stats]) if stats else 0

  return {
      "rating_sc": j.get("ratingSc", 0) or 0,
      "rating_uttf": j.get("ratingUttf", 0) or 0,
      "win_rate": win_rate,
      "set_ratio": set_ratio,
      "momentum": momentum,
      "fatigue": fatigue,
      "promedio_puntos": puntos,
      "experiencia": total_partidos,
  }


def calcular_edad(birth_year):
  if not birth_year:
    return "N/D"
  try:
    current_year = datetime.datetime.now().year
    return current_year - int(birth_year)
  except:
    return "N/D"


def mostrar_tarjeta_jugador(j):
  # Extraer datos de la estructura del JSON
  nombre = f"{j.get('firstName', '')} {j.get('lastName', '')}"
  genero = j.get("gender", "")
  genero_icon = "♂️" if genero in ["M", "male", "1"] else ("♀️" if genero else "")

  birth_year = j.get("birthYear") or j.get("yearOfBirth")
  edad = calcular_edad(birth_year)

  ciudad = j.get("city", "Desconocida")
  pais = j.get("country", "")
  ubicacion = f"{ciudad}, {pais}".strip(", ")

  # Imagen de perfil (si viene en el JSON)
  foto_url = j.get("photoUrl") or j.get("avatar") or j.get("imageUrl")

  # Medallas (oro, plata, bronce)
  gold = j.get("goldMedals", 0) or j.get("medals", {}).get("gold", 0)
  silver = j.get("silverMedals", 0) or j.get("medals", {}).get("silver", 0)
  bronze = j.get("bronzeMedals", 0) or j.get("medals", {}).get("bronze", 0)

  stats = j.get("tournamentStats", [])
  total_torneos = len(stats)
  total_w = sum(s.get("matchWin", 0) for s in stats)
  total_l = sum(s.get("matchLost", 0) for s in stats)
  total_partidos = total_w + total_l

  rating_sc = j.get("ratingSc", 0)
  rating_uttf = j.get("ratingUttf", 0)

  # Renderizado visual estilo tarjeta
  col_img, col_info = st.columns([1, 2])
  with col_img:
    if foto_url:
      st.image(foto_url, width=120)
    else:
      st.markdown("👤 *Sin foto*")

  with col_info:
    st.markdown(f"### {nombre} {genero_icon}")
    st.markdown(f"**Edad:** {edad} años")
    st.markdown(f"📍 {ubicacion}")

  st.markdown(f"🏆 **Rank Setka Cup:** `{rating_sc}` | 🏅 **Rank UTTF:** `{rating_uttf}`")
  st.markdown(
      f"🥇 **{gold}** &nbsp;&nbsp;&nbsp; 🥈 **{silver}** &nbsp;&nbsp;&nbsp; 🥉"
      f" **{bronze}**"
  )
  st.markdown(f"📌 **Total torneos:** {total_torneos}")
  st.markdown(
      f"🎮 **Total partidos:** {total_partidos} &nbsp;|&nbsp; <span"
      f" style='color:green;'>**Wins:** {total_w}</span> &nbsp;|&nbsp; <span"
      f" style='color:red;'>**Losses:** {total_l}</span>",
      unsafe_allow_html=True,
  )


# --- UI PRINCIPAL ---
st.title("🏓 Setka Cup Predictor Pro")
st.markdown("Sistema inteligente de predicción y estadísticas de Tenis de Mesa.")

model, jugadores = cargar_modelo_y_datos()

if not jugadores:
  st.error("[!] No se encontró la base de datos completa de jugadores.")
else:
  nombres_jugadores = [
      f"{j.get('firstName', '')} {j.get('lastName', '')} (ID: {j.get('id')}, SC: {j.get('ratingSc', 0)})"
      for j in jugadores
  ]

  st.sidebar.header("Selección de Jugadores")
  j1_nombre = st.sidebar.selectbox("Seleccione al Jugador A", nombres_jugadores, index=0)
  j2_nombre = st.sidebar.selectbox(
      "Seleccione al Jugador B",
      nombres_jugadores,
      index=1 if len(nombres_jugadores) > 1 else 0,
  )

  idx_a = nombres_jugadores.index(j1_nombre)
  idx_b = nombres_jugadores.index(j2_nombre)

  jugador_a = jugadores[idx_a]
  jugador_b = jugadores[idx_b]

  col_a, col_b = st.columns(2)

  with col_a:
    st.subheader("Jugador A")
    mostrar_tarjeta_jugador(jugador_a)

  with col_b:
    st.subheader("Jugador B")
    mostrar_tarjeta_jugador(jugador_b)

  st.markdown("---")
  if st.button(
      "🔮 Predecir Enfrentamiento", type="primary", use_container_width=True
  ):
    f1 = extraer_features(jugador_a)
    f2 = extraer_features(jugador_b)

    vector = pd.DataFrame([{
        "diff_sc": f1["rating_sc"] - f2["rating_sc"],
        "diff_uttf": f1["rating_uttf"] - f2["rating_uttf"],
        "diff_win_rate": f1["win_rate"] - f2["win_rate"],
        "diff_set_ratio": f1["set_ratio"] - f2["set_ratio"],
        "diff_momentum": f1["momentum"] - f2["momentum"],
        "diff_fatigue": f1["fatigue"] - f2["fatigue"],
        "diff_puntos": f1["promedio_puntos"] - f2["promedio_puntos"],
        "diff_exp": f1["experiencia"] - f2["experiencia"],
    }])

    probs = model.predict_proba(vector)[0]
    prob_a = probs[1] * 100
    prob_b = probs[0] * 100

    st.subheader("📊 Resultado de la Predicción")
    col_res1, col_res2 = st.columns(2)
    col_res1.metric(
        label=f"{jugador_a.get('firstName')} {jugador_a.get('lastName')}",
        value=f"{prob_a:.1f}%",
    )
    col_res2.metric(
        label=f"{jugador_b.get('firstName')} {jugador_b.get('lastName')}",
        value=f"{prob_b:.1f}%",
    )

    st.progress(int(prob_a))
