import streamlit as st
import datetime
from mlb_data_collector import recolectar_partidos_desde_fecha

st.title("📦 Recolector de Datos Históricos MLB")

fecha_inicio = st.date_input("Fecha de inicio", datetime.date.today() - datetime.timedelta(days=7))
fecha_fin = st.date_input("Fecha de fin", datetime.date.today())

if st.button("Recolectar datos"):
    if fecha_inicio > fecha_fin:
        st.error("La fecha de inicio no puede ser posterior a la de fin.")
    else:
        with st.spinner("Recolectando datos..."):
            df = recolectar_partidos_desde_fecha(str(fecha_inicio), str(fecha_fin))
            df.to_csv("historical_mlb_games.csv", index=False)
        st.success("✅ Datos guardados en historical_mlb_games.csv")
        st.dataframe(df)
