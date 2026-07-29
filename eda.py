import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px


def run():
    st.title('📊Exploratory Data Analysis')

    data = pd.read_csv('Churnrate.csv', sep=";")
    st.dataframe(data)

    st.write('## 1. Distribusi Churn')
    churn_pct = data['Churn'].value_counts(normalize=True) * 100
 
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.countplot(x='Churn', data=data, ax=ax1)
    ax1.set_xlabel('Churn (0 = Bertahan, 1 = Churn)')
    ax1.set_title('Distribusi Churn')
    st.pyplot(fig1)

    st.write('## 3. Tenure vs Churn')
    med_tenure = data.groupby('Churn')['Tenure'].median()
 
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Churn', y='Tenure', data=data, ax=ax2)
    ax2.set_title('Tenure vs Churn')
    st.pyplot(fig2)

    st.write('## 4. Riwayat Komplain vs Churn')
    complain_churn = data.groupby('Complain')['Churn'].mean() * 100
 
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.barplot(x=complain_churn.index, y=complain_churn.values, ax=ax3)
    ax3.set_xlabel('Complain (0 = Tidak, 1 = Pernah)')
    ax3.set_ylabel('Churn Rate (%)')
    ax3.set_title('Churn Rate berdasarkan Riwayat Komplain')
    st.pyplot(fig3)

    st.write('## 5. Kategori Produk Favorit vs Churn')
    cat_churn = (data.groupby('PreferedOrderCat')['Churn'].mean() * 100).sort_values(ascending=False)
 
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sns.barplot(x=cat_churn.values, y=cat_churn.index, ax=ax4)
    ax4.set_xlabel('Churn Rate (%)')
    ax4.set_title('Churn Rate berdasarkan Kategori Produk Favorit')
    st.pyplot(fig4)

    st.write('## 6. Jumlah Device Terdaftar vs Churn')
    device_churn = data.groupby('NumberOfDeviceRegistered')['Churn'].mean() * 100
 
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=device_churn.index, y=device_churn.values, ax=ax6)
    ax6.set_xlabel('Jumlah Device Terdaftar')
    ax6.set_ylabel('Churn Rate (%)')
    ax6.set_title('Churn Rate berdasarkan Jumlah Device Terdaftar')
    st.pyplot(fig6)

    st.write('## 7. City Tier vs Churn')
    tier_churn = data.groupby('CityTier')['Churn'].mean() * 100
 
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    sns.barplot(x=tier_churn.index, y=tier_churn.values, ax=ax5)
    ax5.set_xlabel('City Tier')
    ax5.set_ylabel('Churn Rate (%)')
    ax5.set_title('Churn Rate berdasarkan City Tier')
    st.pyplot(fig5)

    st.write('## 8. Korelasi Fitur Numerik')
    num_df = data.select_dtypes(include=np.number)
 
    fig7, ax7 = plt.subplots(figsize=(11, 8))
    sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax7)
    ax7.set_title('Correlation Heatmap - Fitur Numerik')
    st.pyplot(fig7)
 
    corr_churn = num_df.corr()['Churn'].drop('Churn')
    fitur_terkuat = corr_churn.abs().idxmax()

    


if __name__ == "__main__":
    run()

