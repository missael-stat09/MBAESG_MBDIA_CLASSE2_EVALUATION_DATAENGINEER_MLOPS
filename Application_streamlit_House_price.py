#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from snowflake.snowpark.context import get_active_session

st.cache_data.clear()  # Vider le cache

session = get_active_session()

@st.cache_data
def load_scaler_stats():
    df = session.table("HOUSE_PRICE_DB.PUBLIC.SCALER_STATS").to_pandas()
    means = dict(zip(df['FEATURE'].str.upper(), df['MEAN']))
    stds  = dict(zip(df['FEATURE'].str.upper(), df['STD']))
    return means, stds

means, stds = load_scaler_stats()

st.title("🏠 Estimation du Prix d'une Maison")
st.write("Renseignez les caractéristiques de la maison pour obtenir une estimation du prix.")

col1, col2, col3 = st.columns(3)

with col1:
    area      = st.number_input("Surface (m²)", min_value=33, max_value=324, value=100)
    bedrooms  = st.slider("Chambres", 1, 6, 3)
    bathrooms = st.slider("Salles de bain", 1, 4, 1)
    stories   = st.slider("Étages", 1, 4, 2)

with col2:
    parking         = st.slider("Places de parking", 0, 3, 1)
    mainroad        = st.selectbox("Route principale", ["yes", "no"])
    guestroom       = st.selectbox("Chambre d'amis", ["yes", "no"])
    basement        = st.selectbox("Sous-sol", ["yes", "no"])

with col3:
    airconditioning  = st.selectbox("Climatisation", ["yes", "no"])
    prefarea         = st.selectbox("Zone privilégiée", ["yes", "no"])
    furnishingstatus = st.selectbox("Ameublement", 
                                    ["furnished", "semi-furnished", "unfurnished"])

if st.button("Estimer le prix", type="primary"):
    start_time = time.time()
    
    with st.spinner("Calcul en cours..."):
        
        # Encodage
        input_data = {
    'AREA'                               : area,
    'BEDROOMS'                           : bedrooms,
    'BATHROOMS'                          : bathrooms,
    'STORIES'                            : stories,
    'PARKING'                            : parking,
    'MAINROAD'                           : 1 if mainroad == "yes" else 0,
    'GUESTROOM'                          : 1 if guestroom == "yes" else 0,
    'BASEMENT'                           : 1 if basement == "yes" else 0,
    'AIRCONDITIONING'                    : 1 if airconditioning == "yes" else 0,
    'PREFAREA'                           : 1 if prefarea == "yes" else 0,
    'FURNISHINGSTATUS_SEMI_FURNISHED'    : 1 if furnishingstatus == "semi-furnished" else 0,
    'FURNISHINGSTATUS_UNFURNISHED'       : 1 if furnishingstatus == "unfurnished" else 0,
}

        # Normalisation manuelle sans sklearn
        input_scaled = {col: (val - means[col]) / stds[col] 
                        for col, val in input_data.items()}
        
        input_df = pd.DataFrame([input_scaled])

        try:
            # Prédiction via SQL
            input_snowpark = session.create_dataframe(input_df)
            input_snowpark.create_or_replace_temp_view("TEMP_HOUSE_PREDICTION")

            result = session.sql("""
                SELECT HOUSE_PRICE_XGBOOST!PREDICT(*) AS predicted_price 
                FROM TEMP_HOUSE_PREDICTION
            """).collect()

            pred_log   = json.loads(result[0]['PREDICTED_PRICE'])
            pred_value = float(list(pred_log.values())[0])
            pred_euros = int(np.exp(pred_value))

            end_time = time.time()

            st.success(f"💰 Prix estimé : **{pred_euros:,} €**")
            st.caption(f"⏱️ Prédiction effectuée en {(end_time - start_time):.2f} secondes")

        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")


# In[ ]:




