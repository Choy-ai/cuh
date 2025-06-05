# app_auto_predictor.py

import streamlit as st
import pandas as pd
import datetime
import requests
import joblib
from sklearn.preprocessing import StandardScaler

def safe_api_call(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error en llamada a API: {e}")
        return None

@st.cache_data(ttl=3600)
def obtener_stats_equipos(season='2024'):
    url = f'https://statsapi.mlb.com/api/v1/teams?sportId=1&season={season}'
    data = safe_api_call(url)
    if not data:
        return pd.DataFrame()
    equipos = []
    for team in data.get('teams', []):
        team_id = team['id']
        name = team['name']
        stats_url = f'https://statsapi.mlb.com/api/v1/teams/stats?season={season}&group=hitting,pitching&teamId={team_id}'
        stats_resp = safe_api_call(stats_url)
        if not stats_resp or not stats_resp.get('stats'):
            continue
        hitting = stats_resp['stats'][0]['splits'][0]['stat'] if stats_resp['stats'][0]['splits'] else {}
        pitching = stats_resp['stats'][1]['splits'][0]['stat'] if stats_resp['stats'][1]['splits'] else {}
        equipos.append({
            'team_id': team_id,
            'team_name': name,
            'avg': float(hitting.get('avg', 0) or 0),
            'ops': float(hitting.get('ops', 0) or 0),
            'runs': float(hitting.get('runs', 0) or 0),
            'gamesPlayed': int(hitting.get('gamesPlayed', 1) or 1),
            'era': float(pitching.get('era', 0) or 0)
        })
    return pd.DataFrame(equipos)

@st.cache_data(ttl=300)
def obtener_partidos_hoy():
    hoy = datetime.datetime.now().strftime('%Y-%m-%d')
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={hoy}&hydrate=probablePitcher(lineups),team,linescore'
    data = safe_api_call(url)
    if not data or 'dates' not in data or not data['dates']:
        return []
    partidos = []
    for fecha in data['dates']:
        for game in fecha.get('games', []):
            equipos = game['teams']
            partidos.append({
                'game_id': game['gamePk'],
                'home_id': equipos['home']['team']['id'],
                'home_name': equipos['home']['team']['name'],
                'away_id': equipos['away']['team']['id'],
                'away_name': equipos['away']['team']['name'],
                'home_pitcher': equipos['home'].get('probablePitcher', {}).get('fullName', 'N/A'),
                'away_pitcher': equipos['away'].get('probablePitcher', {}).get('fullName', 'N/A'),
            })
    return partidos

@st.cache_data(ttl=3600)
def obtener_stats_pitcher(nombre):
    if nombre == 'N/A' or not nombre:
        return {'era': 4.50, 'whip': 1.30}
    search_url = f'https://search-api.mlb.com/svc/search/v2/mlb?query={nombre.replace(" ", "%20")}&limit=1'
    r = safe_api_call(search_url)
    try:
        player_id = r['docs'][0]['player_id']
        stats_url = f'https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&season=2024&group=pitching'
        stats_resp = safe_api_call(stats_url)
        stats = stats_resp['stats'][0]['splits'][0]['stat'] if stats_resp and stats_resp['stats'] and stats_resp['stats'][0]['splits'] else {}
        return {'era': float(stats.get('era', 4.5)), 'whip': float(stats.get('whip', 1.3))}
    except:
        return {'era': 4.50, 'whip': 1.30}

@st.cache_resource(ttl=3600)
def cargar_modelos():
    scaler = joblib.load("scaler.pkl")
    model_win = joblib.load("model_win.pkl")
    model_runs = joblib.load("model_runs.pkl")
    model_hits = joblib.load("model_hits.pkl")
    return scaler, model_win, model_runs, model_hits

def predecir_partido(row, scaler, model_win, model_runs, model_hits):
    X = scaler.transform(row)
    prob_win = model_win.predict_proba(X)[0][1]
    pred_runs = model_runs.predict(X)[0]
    pred_hits = model_hits.predict(X)[0]
    return prob_win, pred_runs, pred_hits

def recolectar_datos_y_entrenar(dias=30):
    from mlb_data_collector import recolectar_partidos_desde_fecha, entrenar_modelo
    hoy = datetime.date.today()
    inicio = hoy - datetime.timedelta(days=dias)
    df_hist = recolectar_partidos_desde_fecha(inicio.strftime('%Y-%m-%d'), hoy.strftime('%Y-%m-%d'))
    if df_hist.empty:
        st.warning("No se encontraron datos históricos para entrenar.")
        return False
    entrenar_modelo(df_hist)
    return True

def main():
    st.set_page_config(page_title="MLB Auto Predictor", layout="wide")
    st.title("⚾ MLB Auto Predictor: Predicción y Reentrenamiento Automático")

    if st.button("🔄 Reentrenar modelos con últimos datos históricos (30 días)"):
        with st.spinner("Recolectando datos y entrenando modelos..."):
            exito = recolectar_datos_y_entrenar(dias=30)
            if exito:
                st.success("Modelos reentrenados correctamente.")
            else:
                st.error("Fallo en reentrenamiento.")

    st.markdown("---")

    df_teams = obtener_stats_equipos()
    if df_teams.empty:
        st.error("No se pudieron cargar datos de equipos.")
        return

    partidos = obtener_partidos_hoy()
    if not partidos:
        st.warning("No hay partidos programados para hoy.")
        return

    partido_sel = st.selectbox("Selecciona un partido del día:", [f"{p['away_name']} @ {p['home_name']}" for p in partidos])
    sel = next(p for p in partidos if f"{p['away_name']} @ {p['home_name']}" == partido_sel)

    team_A = df_teams[df_teams['team_name'] == sel['away_name']].iloc[0]
    team_B = df_teams[df_teams['team_name'] == sel['home_name']].iloc[0]

    pitch_A = obtener_stats_pitcher(sel['away_pitcher'])
    pitch_B = obtener_stats_pitcher(sel['home_pitcher'])

    input_row = pd.DataFrame([{
        'home_runs': team_B['runs'] / max(team_B['gamesPlayed'], 1),  # usando home stats en home_runs feature
        'away_runs': team_A['runs'] / max(team_A['gamesPlayed'], 1),
        'home_hits': team_B['avg'] * max(team_B['gamesPlayed'], 1),  # aproximación
        'away_hits': team_A['avg'] * max(team_A['gamesPlayed'], 1),
        'home_pitcher_era': pitch_B['era'],
        'away_pitcher_era': pitch_A['era'],
        'home_pitcher_whip': pitch_B['whip'],
        'away_pitcher_whip': pitch_A['whip'],
    }])

    # Para nuestro modelo actual que usa home_runs, away_runs, home_hits, away_hits:
    # Solo dejamos las 4 columnas que usa el modelo (aunque añadimos otras, el modelo solo lee esas)
    input_model = input_row[['home_runs', 'away_runs', 'home_hits', 'away_hits']]

    scaler, model_win, model_runs, model_hits = cargar_modelos()

    prob_win, pred_runs, pred_hits = predecir_partido(input_model, scaler, model_win, model_runs, model_hits)

    st.subheader(f"{sel['away_name']} vs {sel['home_name']}")
    st.write(f"🧠 **Probabilidad de victoria para {sel['away_name']}:** {prob_win:.2%}")
    st.write(f"⚾ **Total de carreras esperado:** {pred_runs:.2f}")
    st.write(f"🔥 **Total de hits esperado:** {pred_hits:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Lanzador Visitante", sel['away_pitcher'])
        st.metric("ERA", pitch_A['era'])
        st.metric("WHIP", pitch_A['whip'])
    with col2:
        st.metric("Lanzador Local", sel['home_pitcher'])
        st.metric("ERA", pitch_B['era'])
        st.metric("WHIP", pitch_B['whip'])

if __name__ == "__main__":
    main()
