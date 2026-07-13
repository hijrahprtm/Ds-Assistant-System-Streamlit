import streamlit as st
import pandas as pd
import plotly.express as px
import json
import re
from groq import Groq

# 1. Set halaman Streamlit agar responsif dan estetik
st.set_page_config(page_title="Alex Data Science Workspace", layout="wide")

# 2. Hubungkan ke Groq Cloud API Client
@st.cache_resource
def get_groq_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return Groq(api_key=api_key)
    except Exception:
        return None

client = get_groq_client()

st.title("🤖 Alex - Data Science Workspace (Cloud Llama)")
st.caption("Zero-cost UI, Automated Dataset Generation, and Instant Custom CSV Analytics")
st.write("---")

# Validasi API Key Groq awal
if client is None:
    st.error("API Key belum disetel! Harap masukkan variabel `GROQ_API_KEY` di Advanced Settings -> Streamlit Secrets.")
    st.stop()

# ==========================================================
# 💬 AREA UTAMA: INPUT CHAT & TOMBOL "ADD CSV"
# ==========================================================
user_query = st.text_area(
    "Masukkan perintah atau instruksi analisis data kamu di sini:", 
    height=120,
    placeholder="Contoh: Buatkan data tren penjualan 5 produk selama 6 bulan terakhir beserta grafiknya.\n"
                "Atau aktifkan opsi di bawah untuk mengunggah file CSV milikmu sendiri."
)

# Fitur Interaktif: Tombol Toggle untuk Membuka Fitur "Add CSV"
show_upload_option = st.toggle("📎 Tambahkan File CSV (Add CSV untuk Analisis)", value=False)

csv_context_prompt = ""
uploaded_df = None

# Jika Fitur "Add CSV" diaktifkan oleh user, tampilkan uploader secara dinamis
if show_upload_option:
    st.markdown("---")
    st.subheader("📁 Unggah Dataset CSV")
    uploaded_file = st.file_uploader("Pilih file CSV dari komputer Anda", type=["csv"])
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Berhasil memuat: `{uploaded_file.name}` ({uploaded_df.shape[0]} baris, {uploaded_df.shape[1]} kolom)")
            
            with st.expander("🔍 Lihat Pratinjau Data (5 Baris Teratas)"):
                st.dataframe(uploaded_df.head(5), use_container_width=True)
            
            # Ekstraksi metadata skema untuk disuntikkan ke konteks Llama
            columns_schema = {col: str(dtype) for col, dtype in zip(uploaded_df.columns, uploaded_df.dtypes)}
            data_sample_head = uploaded_df.head(3).to_dict(orient='records')
            
            csv_context_prompt = (
                f"\n\n[USER UPLOADED DATA CONTEXT]\n"
                f"The user has uploaded a local CSV dataset named: '{uploaded_file.name}'.\n"
                f"Dataset Dimensions: {uploaded_df.shape[0]} rows and {uploaded_df.shape[1]} columns.\n"
                f"Columns & Data Types Schema: {json.dumps(columns_schema)}\n"
                f"Sample Data (First 3 rows): {json.dumps(data_sample_head)}\n"
                f"INSTRUCTION: Use this dataset schema and context to answer the user's analytical questions accurately."
            )
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")
    st.markdown("---")

# Tombol Eksekusi Utama
process_btn = st.button("Proses Data & Visualisasikan 🚀", type="primary", use_container_width=True)

# ==========================================================
# 🧠 ALUR PEMROSESAN UTAMA (PROMPTING & RENDER VISUALISASI)
# ==========================================================
if process_btn:
    if user_query.strip() == "":
        st.warning("Ketik instruksi analisis Anda terlebih dahulu ya!")
    else:
        with st.spinner("Alex sedang memikirkan hasil analisis terbaik untuk Anda..."):
            
            system_prompt = (
                "You are Alex, a Senior Data Scientist. Answer the user's question naturally in the language they use (Indonesian or English).\n"
                "CRITICAL INSTRUCTION: If the user requests new data generation, or asks for statistical transformation of the uploaded data, you MUST append a valid JSON array of objects representing the data at the very end of your response, wrapped inside ```json and ``` blocks. "
                "The JSON must be tabular-friendly (flat key-value pairs). Do not mix prose inside the JSON block."
            )
            
            final_user_prompt = f"{system_prompt}{csv_context_prompt}\n\nUser Question: {user_query}"
            
            try:
                # Panggil model Llama terbaru (Llama 3.3 70B Versatile)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": final_user_prompt}],
                    model="llama-3.3-70b-versatile",
                )
                full_reply = chat_completion.choices[0].message.content
                
                # Ekstrak teks narasi penjelasan dan blok kode data JSON menggunakan Regex
                json_match = re.search(r'```json\s*(.*?)\s*```', full_reply, re.DOTALL)
                explanation_text = re.sub(r'```json\s*.*?\s*```', '', full_reply, flags=re.DOTALL).strip()
                
                # Tampilkan Penjelasan Utama Hasil Analisis Alex
                st.write("---")
                st.subheader("💡 Analisis & Jawaban Alex")
                st.markdown(explanation_text)
                
                # RENDER DATASET BARU / OLAHAN (Jika LLM merespon dengan format JSON)
                if json_match:
                    json_data_str = json_match.group(1)
                    data = json.loads(json_data_str)
                    df_to_render = pd.DataFrame(data)
                    
                    st.write("---")
                    st.subheader("📊 Dataset Hasil Olahan / Generasi (CSV)")
                    st.dataframe(df_to_render, use_container_width=True)
                    
                    csv_bytes = df_to_render.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Dataset Olahan (CSV)",
                        data=csv_bytes,
                        file_name='alex_processed_dataset.csv',
                        mime='text/csv',
                    )
                    
                    # Pembuatan Visualisasi Otomatis menggunakan Plotly
                    st.write("---")
                    st.subheader("📈 Visualisasi Grafik Otomatis")
                    
                    numeric_cols = df_to_render.select_dtypes(include=['number']).columns.tolist()
                    string_cols = df_to_render.select_dtypes(include=['object']).columns.tolist()
                    
                    if len(string_cols) >= 1 and len(numeric_cols) >= 1:
                        fig = px.bar(df_to_render, x=string_cols[0], y=numeric_cols[0], 
                                     title=f"Grafik Analisis: {string_cols[0]} vs {numeric_cols[0]}",
                                     template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(numeric_cols) >= 2:
                        fig = px.line(df_to_render, x=df_to_render.index, y=numeric_cols[0], 
                                      title=f"Tren Analisis Data: {numeric_cols[0]}", 
                                      template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                
                # RENDER DATASET UNGGAHAN (Jika tidak ada JSON baru dari LLM, namun ada file yang aktif diunggah)
                elif uploaded_df is not None:
                    st.write("---")
                    st.info("💡 Tip: Alex memberikan analisis langsung pada berkas Anda. Di bawah ini adalah visualisasi cepat dari distribusi data mentah yang Anda unggah:")
                    
                    num_cols = uploaded_df.select_dtypes(include=['number']).columns.tolist()
                    str_cols = uploaded_df.select_dtypes(include=['object']).columns.tolist()
                    
                    if len(str_cols) >= 1 and len(num_cols) >= 1:
                        fig = px.histogram(uploaded_df, x=str_cols[0], y=num_cols[0], 
                                           title=f"Distribusi Data Unggahan: {str_cols[0]} vs {num_cols[0]}",
                                           template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan pemrosesan pada sistem AI: {e}")
