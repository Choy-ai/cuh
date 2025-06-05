# mlb_data_collector.py
import requests
import pandas as pd
import datetime
import time
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

def safe_api_call(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error en llamada a API: {e}")
        return None

def obtener_resultados_juego(game_pk):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    data = safe_api_call(url)
    if data is None:
        return None

    linescore = data.get("liveData", {}).get("linescore", {})
    teams = data.get("gameData", {}).get("teams", {})

    return {
        "game_id": game_pk,
        "home_team": teams.get("home", {}).get("name", "N/A"),
        "away_team": teams.get("away", {}).get("name", "N/A"),
        "home_runs": linescore.get("teams", {}).get("home", {}).get("runs", 0),
        "away_runs": linescore.get("teams", {}).get("away", {}).get("runs", 0),
        "home_hits": linescore.get("teams", {}).get("home", {}).get("hits", 0),
        "away_hits": linescore.get("teams", {}).get("away", {}).get("hits", 0)
    }

def recolectar_partidos_desde_fecha(inicio: str, fin: str):
    fechas = pd.date_range(start=inicio, end=fin).strftime('%Y-%m-%d')
    juegos = []
    for fecha in fechas:
        print(f"Revisando fecha: {fecha}")
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={fecha}"
        data = safe_api_call(url)
        if data is None or not data.get('dates'):
            continue

        for game in data['dates'][0].get('games', []):
            game_pk = game['gamePk']
            resultado = obtener_resultados_juego(game_pk)
            if resultado:
                juegos.append(resultado)
            time.sleep(0.3)  # para no saturar la API

    return pd.DataFrame(juegos)

def entrenar_modelo(df):
    df = df.copy()
    df["total_runs"] = df["home_runs"] + df["away_runs"]
    df["total_hits"] = df["home_hits"] + df["away_hits"]
    df["away_win"] = (df["away_runs"] > df["home_runs"]).astype(int)

    features = ["home_runs", "away_runs", "home_hits", "away_hits"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model_win = LogisticRegression(max_iter=1000)
    model_win.fit(X_scaled, df["away_win"])

    model_runs = LinearRegression()
    model_runs.fit(X_scaled, df["total_runs"])

    model_hits = LinearRegression()
    model_hits.fit(X_scaled, df["total_hits"])

    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(model_win, "model_win.pkl")
    joblib.dump(model_runs, "model_runs.pkl")
    joblib.dump(model_hits, "model_hits.pkl")

    print("Modelos entrenados y guardados exitosamente.")

if __name__ == "__main__":
    hoy = datetime.date.today()
    hace_30 = hoy - datetime.timedelta(days=30)
    df_hist = recolectar_partidos_desde_fecha(hace_30.strftime('%Y-%m-%d'), hoy.strftime('%Y-%m-%d'))
    df_hist.to_csv("historical_mlb_games.csv", index=False)
    print("Datos guardados en historical_mlb_games.csv")

    entrenar_modelo(df_hist)
