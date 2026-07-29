import streamlit as st
import eda 
import inf

st.set_page_config(
    page_title='Prediksi Customer Churn E-Commerce',
    page_icon='📊',
    layout='wide'
)
 
# sidebar navigasi untuk berpindah halaman
with st.sidebar:
    st.title('Navigasi')
    page = st.radio('Pilih Halaman:', ['EDA', 'Prediksi Churn'])
 
    st.markdown('---')
    st.caption(
        'Milestone 2 - Prediksi Customer Churn E-Commerce. '
        'Model: XGBoost (tuned) dengan Recall sebagai metric utama. '
        'Dataset: Ecommerce Customer Churn (Kaggle).'
    )
 
if page == 'EDA':
    eda.run()
else:
    inf.run()
 