import streamlit as st
import pandas as pd
import pickle

with open('best_model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

def run():
    st.title('Prediksi Risiko Churn Pelanggan')

    with st.form('form_churn'):
         col1, col2, col3 = st.columns(3)
    with col1:
            st.subheader('Profil')
            gender = st.selectbox('Gender', ['Male', 'Female'])
            marital = st.selectbox('Status Pernikahan', ['Married', 'Single', 'Divorced'])
            city_tier = st.selectbox('City Tier', [1, 2, 3])
            warehouse = st.number_input('Jarak Gudang ke Rumah (km)', min_value=1, max_value=50, value=15)
            n_address = st.number_input('Jumlah Alamat Terdaftar', min_value=1, max_value=25, value=3)
            n_device = st.number_input('Jumlah Device Terdaftar', min_value=1, max_value=10, value=4)
    with col2:
            st.subheader('Perilaku')
            tenure = st.number_input('Tenure (bulan)', min_value=0, max_value=70, value=9)
            login_device = st.selectbox('Device Login Favorit', ['Phone', 'Computer'])
            hour_app = st.number_input('Jam di App per Hari', min_value=0, max_value=10, value=3)
            order_cat = st.selectbox('Kategori Produk Favorit',
                                     ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'])
            satisfaction = st.slider('Skor Kepuasan (1-5)', min_value=1, max_value=5, value=3)
            complain = st.selectbox('Pernah Komplain Bulan Lalu?', ['Tidak', 'Ya'])
 
    with col3:
            st.subheader('Transaksi')
            payment = st.selectbox('Metode Pembayaran Favorit',
                                   ['Debit Card', 'Credit Card', 'E wallet', 'UPI', 'Cash on Delivery'])
            order_count = st.number_input('Jumlah Order Bulan Lalu', min_value=0, max_value=30, value=2)
            coupon = st.number_input('Kupon Dipakai Bulan Lalu', min_value=0, max_value=30, value=1)
            hike = st.number_input('Kenaikan Nilai Order dari Tahun Lalu (%)', min_value=0, max_value=50, value=15)
            cashback = st.number_input('Rata-rata Cashback Bulan Lalu', min_value=0.0, max_value=500.0, value=165.0)
 
            submitted = st.form_submit_button('Prediksi')

    if submitted:
        data_baru = pd.DataFrame([{
            'Tenure': tenure,
            'PreferredLoginDevice': login_device,
            'CityTier': city_tier,
            'WarehouseToHome': warehouse,
            'PreferredPaymentMode': payment,
            'Gender': gender,
            'HourSpendOnApp': hour_app,
            'NumberOfDeviceRegistered': n_device,
            'PreferedOrderCat': order_cat,
            'SatisfactionScore': satisfaction,
            'MaritalStatus': marital,
            'NumberOfAddress': n_address,
            'Complain': 1 if complain == 'Ya' else 0,
            'OrderAmountHikeFromlastYear': hike,
            'CouponUsed': coupon,
            'OrderCount': order_count,
            'CashbackAmount': cashback
        }])

        with st.expander('Lihat data yang dimasukkan'):
            st.dataframe(data_baru)
 
        # pipeline otomatis menangani imputasi, scaling, dan encoding
        pred = model.predict(data_baru)[0]
        proba = model.predict_proba(data_baru)[0][1]
 
        st.markdown('---')
        st.subheader('Hasil Prediksi')
        st.metric('Probabilitas Churn', f'{proba * 100:.1f}%')
        st.progress(float(proba))
 
        if pred == 1:
            st.error('Pelanggan ini terprediksi **CHURN** (berisiko berhenti bertransaksi).')
 
            # rekomendasi dasar untuk semua pelanggan berisiko
            rekomendasi = (
                '**Rekomendasi:** masukkan pelanggan ini ke daftar prioritas tim retention - '
                'berikan intervensi seperti voucher/promo personal atau penawaran cashback.'
            )
            # rekomendasi tambahan yang menyesuaikan profil input
            if complain == 'Ya':
                rekomendasi += (
                    ' Pelanggan memiliki **riwayat komplain** - prioritaskan follow-up customer service '
                    'untuk menyelesaikan keluhannya terlebih dahulu.'
                )
            if tenure <= 3:
                rekomendasi += (
                    ' Pelanggan masih dalam **fase onboarding** (tenure sangat pendek) - segmen dengan '
                    'risiko churn tertinggi menurut hasil EDA.'
                )
            st.markdown(rekomendasi)
        else:
            st.success('Pelanggan ini terprediksi **BERTAHAN**.')
            st.markdown(
                '**Rekomendasi:** tidak perlu alokasi budget promo retensi khusus - cukup dijaga '
                'melalui engagement reguler agar anggaran retensi tetap efisien.'
            )
           
if __name__ == "__main__":
    run()