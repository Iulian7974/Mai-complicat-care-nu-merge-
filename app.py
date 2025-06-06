import streamlit as st
import pandas as pd
from loto_xgb_utils import load_data, train_model, predict_next_draw

st.set_page_config(page_title="Loto 6/49 - XGBoost", layout="centered")
st.title("🎯 Predicții Loto 6/49 cu XGBoost")

uploaded_file = st.file_uploader("Încarcă fișierul Excel cu extrageri", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.success("✅ Date încărcate!")

    if st.button("🔄 Antrenează modelul și prezice"):
        model = train_model(df)
        pred_6, pred_10 = predict_next_draw(model, df)
        st.subheader("🔮 Predicții")
        st.write("🎯 Cele mai probabile 6 numere:", sorted(pred_6))
        st.write("🎯 Cele mai probabile 10 numere:", sorted(pred_10))
