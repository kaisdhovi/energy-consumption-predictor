import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

st.set_page_config(page_title="Analisis Konsumsi Daya Rumah Tangga", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
    }
    .stDataFrame th, .stDataFrame td {
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    .stDataFrame th {
        background-color: #e6f0fa;
        color: #2c3e50;
    }
    .stDataFrame tr:hover {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Analisis Konsumsi Daya Rumah Tangga")
st.markdown("""
    Aplikasi ini memprediksi daya rumah tangga dengan agregasi per jam menggunakan SVR.
""")

# Upload File
st.subheader("Unggah Dataset")
dataset_file = st.file_uploader(
    "Pilih file household_power_consumption.txt",
    type=["txt"],
    help="Unggah file dengan format .txt dan pemisah ';'"
)

progress_bar = st.progress(0)
progress_text = st.empty()

# Outlier
def handle_outliers_iqr(df, columns):
    df_out = df.copy()
    outlier_info = []
    
    for column in columns:
        Q1 = df_out[column].quantile(0.25)
        Q3 = df_out[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        original_min = df_out[column].min()
        original_max = df_out[column].max()
        df_out[column] = np.clip(df_out[column], lower_bound, upper_bound)
        clipped_min = df_out[column].min()
        clipped_max = df_out[column].max()
        clipped_count = np.sum((df[column] < lower_bound) | (df[column] > upper_bound))
        outlier_info.append({
            'Column': column,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Original Range': f"[{original_min:.4f}, {original_max:.4f}]",
            'Clipped Range': f"[{clipped_min:.4f}, {clipped_max:.4f}]",
            'Values Clipped': clipped_count
        })
    
    return df_out, pd.DataFrame(outlier_info)

# Load dan Preprocess
def load_and_preprocess_data(file):
    progress_bar.progress(10)
    progress_text.text("Memuat dataset...")
    
    dtypes = {
        'Global_active_power': float,
        'Global_reactive_power': float,
        'Voltage': float,
        'Global_intensity': float,
        'Sub_metering_1': float,
        'Sub_metering_2': float,
        'Sub_metering_3': float
    }
    
    try:
        energy_df_raw = pd.read_csv(
            file,
            sep=';',
            dtype=dtypes,
            na_values='?',
            parse_dates={'Date_Time': ['Date', 'Time']},
            dayfirst=True
        )
        progress_bar.progress(20)
        progress_text.text("Menangani nilai yang hilang...")
        
        energy_df_filled = energy_df_raw.copy()
        energy_df_filled.fillna(method='ffill', inplace=True)
        energy_df_indexed = energy_df_filled.copy()
        energy_df_indexed.set_index('Date_Time', inplace=True)
        return energy_df_indexed
    
    except Exception as e:
        st.error(f"Error saat memuat data: {e}")
        return None

# Agregasi Per Jam dan Feature Engineering
def aggregate_and_engineer_features(df):
    progress_bar.progress(30)
    progress_text.text("Melakukan agregasi per jam (sum untuk energi, median untuk lainnya)...")
    
    aggregation_cols = [
        'Global_active_power',
        'Global_reactive_power',
        'Global_intensity',
        'Voltage',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]
    
    aggregation_dict = {
        'Global_active_power': 'sum',
        'Global_reactive_power': 'median',
        'Global_intensity': 'median',
        'Voltage': 'median',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum'
    }
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    energy_hourly_agg = df.resample('h').agg(aggregation_dict)
    energy_hourly_agg.fillna(method='ffill', inplace=True)
    
    # Convert summed power to total energy in kWh
    energy_hourly_agg['Total_Energy_kWh'] = energy_hourly_agg['Global_active_power'] / 60.0
    energy_hourly_agg['Sub_metering_1_kWh'] = energy_hourly_agg['Sub_metering_1'] / 1000.0
    energy_hourly_agg['Sub_metering_2_kWh'] = energy_hourly_agg['Sub_metering_2'] / 1000.0
    energy_hourly_agg['Sub_metering_3_kWh'] = energy_hourly_agg['Sub_metering_3'] / 1000.0
    
    # Drop original summed columns
    energy_hourly_agg.drop(columns=['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], inplace=True, errors='ignore')
    
    return energy_hourly_agg, aggregation_cols

# Handling Outlier GUI
def process_outliers(df, aggregation_cols):
    progress_bar.progress(40)
    progress_text.text("Menangani outlier...")
    
    numerical_cols_for_outliers = ['Total_Energy_kWh', 'Global_reactive_power', 'Global_intensity', 'Voltage', 'Sub_metering_1_kWh', 'Sub_metering_2_kWh', 'Sub_metering_3_kWh']
    df_no_outliers, outlier_info = handle_outliers_iqr(df, numerical_cols_for_outliers)
    return df_no_outliers, outlier_info

# Feature Engineering GUI
def add_time_features(df):
    progress_bar.progress(50)
    progress_text.text("Menambahkan fitur waktu dan siklikal...")
    
    df_final = df.copy()
    df_final['Year'] = df_final.index.year
    df_final['Month'] = df_final.index.month
    df_final['Day'] = df_final.index.weekday
    df_final['Hour'] = df_final.index.hour
    df_final['DayOfYear'] = df_final.index.dayofyear
    
    df_final['Hour_sin'] = np.sin(2 * np.pi * df_final['Hour'] / 24.0)
    df_final['Hour_cos'] = np.cos(2 * np.pi * df_final['Hour'] / 24.0)
    df_final['Day_sin'] = np.sin(2 * np.pi * df_final['Day'] / 7.0)
    df_final['Day_cos'] = np.cos(2 * np.pi * df_final['Day'] / 7.0)
    df_final['Month_sin'] = np.sin(2 * np.pi * df_final['Month'] / 12.0)
    df_final['Month_cos'] = np.cos(2 * np.pi * df_final['Month'] / 12.0)
    df_final['DayOfYear_sin'] = np.sin(2 * np.pi * df_final['DayOfYear'] / 365.0)
    df_final['DayOfYear_cos'] = np.cos(2 * np.pi * df_final['DayOfYear'] / 365.0)
    
    # fitur lag 1 jam
    progress_text.text("Menambahkan fitur lag 1 jam...")
    df_final['Total_Energy_kWh_lag1'] = df_final['Total_Energy_kWh'].shift(1)
    initial_rows = df_final.shape[0]
    df_final.dropna(subset=['Total_Energy_kWh_lag1'], inplace=True)
    final_rows = df_final.shape[0]
    st.write(f"Dropped {initial_rows - final_rows} row(s) due to NaN in lag feature.")
    
    return df_final

# Visualisasi GUI
def visualize_preprocessed_data(df):
    progress_bar.progress(60)
    progress_text.text("Membuat visualisasi data...")
    
    st.subheader("Distribusi Total Energi Per Jam (Setelah Penanganan Outlier)")
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(df['Total_Energy_kWh'], kde=True, bins=50, color='blue')
    plt.title('Distribusi Total Energi Per Jam (kWh, Outlier Ditangani)', fontsize=16)
    plt.xlabel('Total Energi Per Jam (kWh)', fontsize=12)
    plt.ylabel('Frekuensi', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)
    
    st.subheader("Distribusi Total Energi terhadap Komponen Waktu")
    fig = plt.figure(figsize=(14, 12))
    time_features = ["Year", "Month", "Day", "Hour"]
    feature_labels = ["Tahun", "Bulan", "Hari dalam Minggu (0=Senin)", "Jam dalam Hari"]
    
    for i, feature in enumerate(time_features):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(data=df, x=feature, y="Total_Energy_kWh", color=sns.color_palette()[i])
        plt.title(f"Total Energi vs {feature_labels[i]} (Outlier Ditangani)", fontsize=14)
        plt.xlabel(feature_labels[i], fontsize=12)
        plt.ylabel("Total Energi Per Jam (kWh)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# === Model Training Setup (80:20 Split) ===
def prepare_model_data(df):
    progress_bar.progress(70)
    progress_text.text("Menyiapkan data untuk pelatihan model...")
    
    feature_cols = [
        'Global_reactive_power',
        'Global_intensity',
        'Voltage',
        'Sub_metering_1_kWh',
        'Sub_metering_2_kWh',
        'Sub_metering_3_kWh',
        'Year',
        'Hour_sin',
        'Hour_cos',
        'Day_sin',
        'Day_cos',
        'Month_sin',
        'Month_cos',
        'DayOfYear_sin',
        'DayOfYear_cos',
        'Total_Energy_kWh_lag1'
    ]
    target_col = "Total_Energy_kWh"
    
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train_index = X_train.index
    X_test_index = X_test.index
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_test_index, feature_cols

class SVR:
    def __init__(self, C=1.0, epsilon=0.1, learning_rate=0.001, epochs=100, batch_size=64, random_state=None):
        self.C = C
        self.epsilon = epsilon
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = None
        self.bias = None
        
        if self.random_state:
            np.random.seed(self.random_state)

    def _loss(self, y_true, y_pred):
        error = y_true - y_pred
        loss_part = np.maximum(0, np.abs(error) - self.epsilon)
        reg_part = 0.5 * np.dot(self.weights, self.weights)
        total_cost = reg_part + self.C * np.sum(loss_part)
        return total_cost

    def _gradient(self, X_batch, y_batch):
        n_samples = X_batch.shape[0]
        y_pred = self._predict_batch(X_batch)
        error = y_batch - y_pred
        grad_loss_w = np.zeros_like(self.weights)
        grad_loss_b = 0
        
        loss_indices = np.abs(error) > self.epsilon
        if np.any(loss_indices):
            sign_error = -np.sign(error[loss_indices])
            grad_loss_w = np.dot(X_batch[loss_indices].T, sign_error)
            grad_loss_b = np.sum(sign_error)
        
        grad_reg_w = self.weights
        dw = grad_reg_w + self.C * grad_loss_w / n_samples
        db = self.C * grad_loss_b / n_samples
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = np.random.randn() * 0.01
        y = y.values if isinstance(y, pd.Series) else y
        
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            if self.random_state is None:
                np.random.seed(int(time.time()))
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                dw, db = self._gradient(X_batch, y_batch)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            y_pred_epoch = self.predict(X)
            current_loss = self._loss(y, y_pred_epoch)
            progress = (epoch + 1) / self.epochs
            progress_bar.progress(int(80 + progress * 20))
            progress_text.text(f"Melatih Model SVR Manual... ({int(progress * 100)}%) | Loss: {current_loss:.4f}")

    def _predict_batch(self, X_batch):
        return np.dot(X_batch, self.weights) + self.bias

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise Exception("Model not trained yet.")
        return np.dot(X, self.weights) + self.bias

# Train SVR
def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, X_test_index, start_date, end_date):
    progress_bar.progress(80)
    progress_text.text("Melatih Model SVR Manual... (0%)")
    
    manual_svr = SVR(
        C=10,
        epsilon=0.1,
        learning_rate=0.0001,
        epochs=200,
        batch_size=128,
        random_state=42
    )
    
    start_time = time.time()
    manual_svr.fit(X_train_scaled, y_train)
    manual_train_time = time.time() - start_time
    
    y_pred_manual = manual_svr.predict(X_test_scaled)
    mse_manual = mean_squared_error(y_test, y_pred_manual)
    rmse_manual = np.sqrt(mse_manual)
    mae_manual = mean_absolute_error(y_test, y_pred_manual)
    r2_manual = r2_score(y_test, y_pred_manual)
    
    st.subheader("Metrik Evaluasi Model SVR Manual")
    metrics_df = pd.DataFrame({
        'Metrik': ['MSE', 'RMSE', 'MAE', 'R²', 'Waktu Pelatihan (detik)'],
        'Nilai': [
            f"{mse_manual:.4f}",
            f"{rmse_manual:.4f}",
            f"{mae_manual:.4f}",
            f"{r2_manual:.4f}",
            f"{manual_train_time:.4f}"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True)
    
    st.subheader(f"Perbandingan Prediksi vs Aktual ({start_date.date()} - {end_date.date()})")
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_manual}, index=X_test_index)
    results_df.sort_index(inplace=True)
    results_df = results_df[(results_df.index >= start_date) & (results_df.index <= end_date)]
    
    if results_df.empty:
        st.error("Tidak ada data dalam rentang tanggal yang dipilih.")
        return
    
    fig = plt.figure(figsize=(15, 7))
    plt.plot(
        results_df.index,
        results_df["Actual"],
        label="Nilai Aktual",
        alpha=0.8,
        linewidth=1,
        color='#1f77b4'
    )
    plt.plot(
        results_df.index,
        results_df["Predicted"],
        label="Nilai Prediksi (SVR Manual)",
        alpha=0.8,
        linestyle="--",
        linewidth=1,
        color='#ff7f0e'
    )
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()
    plt.gca().tick_params(axis='x', rotation=45)
    plt.gca().set_xticks(results_df.index[::len(results_df.index)//5])
    plt.title("SVR Manual: Aktual vs Prediksi Total Energi Per Jam (kWh)", fontsize=16)
    plt.xlabel("Tanggal", fontsize=12)
    plt.ylabel("Total Energi Per Jam (kWh)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close(fig)
    
    progress_bar.progress(100)
    progress_text.text("Pelatihan dan evaluasi model selesai.")

if dataset_file is not None:
    with st.spinner("Memproses data..."):
        # 1. Memuat dan memproses data
        energy_df_indexed = load_and_preprocess_data(dataset_file)
        if energy_df_indexed is not None:
            st.subheader("Data Awal")
            st.dataframe(energy_df_indexed.head(), use_container_width=True)
            
            # 2. Agregasi per jam dan fitur awal
            energy_hourly_summary, aggregation_cols = aggregate_and_engineer_features(energy_df_indexed)
            st.subheader("Data Setelah Agregasi per Jam")
            st.dataframe(energy_hourly_summary.head(), use_container_width=True)
            
            # 3. Penanganan outlier
            energy_hourly_no_outliers, outlier_info = process_outliers(energy_hourly_summary, aggregation_cols)
            st.subheader("Data Setelah Penanganan Outlier")
            st.dataframe(energy_hourly_no_outliers.head(), use_container_width=True)
            
            st.subheader("Informasi Penanganan Outlier")
            st.dataframe(outlier_info, use_container_width=True)
            
            # 4. Fitur waktu dan siklikal
            df_final = add_time_features(energy_hourly_no_outliers)
            st.subheader("Data Setelah Fitur Waktu dan Siklikal")
            st.dataframe(df_final.head(), use_container_width=True)
            
            # Tambahan: Simpan data preprocessed
            preprocessed_path = 'preprocessed_hourly_data_v4_lag.csv'
            df_final.to_csv(preprocessed_path)
            st.write(f"Final preprocessed data saved to {preprocessed_path}")
            
            # 5. Visualisasi data
            visualize_preprocessed_data(df_final)
            
            # 6. Input rentang waktu untuk prediksi
            st.subheader("Pilih Rentang Waktu untuk Prediksi")
            min_date = df_final.index.min().date()
            max_date = df_final.index.max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Tanggal Mulai",
                    min_value=min_date,
                    max_value=max_date,
                    value=min_date
                )
            with col2:
                end_date = st.date_input(
                    "Tanggal Selesai",
                    min_value=min_date,
                    max_value=max_date,
                    value=max_date
                )
            
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # 7. Jalankan analisis
            st.markdown("---")
            if st.button("Jalankan Analisis", use_container_width=True):
                X_train_scaled, X_test_scaled, y_train, y_test, X_test_index, feature_cols = prepare_model_data(df_final)
                
                # 9. Pelatihan dan evaluasi model
                train_and_evaluate_model(
                    X_train_scaled,
                    X_test_scaled,
                    y_train,
                    y_test,
                    X_test_index,
                    start_date,
                    end_date
                )

else:
    st.info("Silakan unggah file dataset untuk memulai analisis.")
