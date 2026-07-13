import streamlit as st
import pandas as pd
import plotly.express as px
import json
import re
import io
import contextlib
from groq import Groq

# 1. Konfigurasi Halaman Utama
st.set_page_config(page_title="Alex Advanced Thinking Workspace", layout="wide")

@st.cache_resource
def get_groq_client():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception:
        pass
    return None

client = get_groq_client()

st.title("🧠 Alex - Advanced Thinking Data Science Workspace")
st.caption("Powered by Llama 3.3 Reasoning & Live Python Execution Engine")
st.write("---")

if client is None:
    st.error("⚠️ API Key Groq Belum Terdeteksi di Streamlit Secrets!")
    st.stop()

# ==========================================================
# 💬 INPUT & UPLOAD CSV
# ==========================================================
user_query = st.text_area(
    "Masukkan instruksi data science dan visualisasi yang kamu mau:", 
    height=120,
    placeholder="Contoh: Lakukan clustering dengan K-Means menjadi 3 cluster, lalu buatkan grafik scatter plot antara Tenure vs Monthly Charges dengan warna berdasarkan Cluster."
)

show_upload_option = st.toggle("📎 Tambahkan File CSV", value=True)

csv_context_prompt = ""
uploaded_df = None

if show_upload_option:
    uploaded_file = st.file_uploader("Unggah berkas CSV Anda", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state['uploaded_df'] = pd.read_csv(uploaded_file)
            uploaded_df = st.session_state['uploaded_df']
            st.success(f"✓ Terbaca: `{uploaded_file.name}` ({uploaded_df.shape[0]} baris)")
            
            with st.expander("🔍 Pratinjau Data"):
                st.dataframe(uploaded_df.head(5), use_container_width=True)
            
            csv_context_prompt = (
                f"\n\n[CONTEXT] User uploaded a file named '{uploaded_file.name}' with shape {uploaded_df.shape}.\n"
                f"Columns and Types: {str(uploaded_df.dtypes.to_dict())}\n"
                f"The dataframe is already loaded in the environment as a variable named `uploaded_df`.\n"
            )
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")

process_btn = st.button("Jalankan Deep Thinking Analysis 🚀", type="primary", use_container_width=True)

# ==========================================================
# 🧠 REASONING & LIVE EXECUTION ENGINE
# ==========================================================
if process_btn:
    if user_query.strip() == "":
        st.warning("Mohon isi instruksi atau pertanyaan Anda terlebih dahulu.")
    else:
        with st.spinner("Alex sedang memikirkan perhitungan & grafik terbaik..."):
            
            # Mengubah instruksi agar AI yang membuat objek fig dari plotly express langsung di kodenya
            system_prompt = (
                "You are Alex, an expert Data Scientist. You have access to a pandas DataFrame named `uploaded_df`.\n"
                "Your task is to write a clean Python script to perform the requested data science task perfectly.\n"
                "CRITICAL CODE RULES:\n"
                "1. You MUST wrap the executable Python code inside a ```python ... ``` block.\n"
                "2. The code must perform the calculation and finally assign the resulting/transformed dataframe to a variable named `output_df`.\n"
                "3. DYNAMIC VISUALIZATION RULE: You MUST create a custom Plotly figure object based on the user's chart preference (e.g., scatter, box, histogram, etc.) and assign it to a variable named `fig`. Make sure it uses `template='plotly_dark'`.\n"
                "4. SAFETY RULE: Do not use inline lambda logic that references the dataframe while creating it (calculate stats into separate variables first).\n"
                "Do not use external libraries that require installation outside pandas, numpy, and scikit-learn."
            )
            
            final_prompt = f"{system_prompt}{csv_context_prompt}\n\nUser Request: {user_query}"
            
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": final_prompt}],
                    model="llama-3.3-70b-versatile", 
                )
                raw_reply = chat_completion.choices[0].message.content
                
                code_match = re.search(r'```python\s*(.*?)\s*```', raw_reply, re.DOTALL)
                explanation_text = re.sub(r'```python\s*.*?\s*```', '', raw_reply, flags=re.DOTALL).strip()
                
                st.subheader("💡 Analisis & Rekomendasi Alex")
                st.markdown(explanation_text)
                
                if code_match:
                    python_code = code_match.group(1)
                    
                    # Inisialisasi environment eksekusi lokal
                    local_env = {
                        'pd': pd,
                        'px': px,
                        'uploaded_df': st.session_state.get('uploaded_df', None),
                        'output_df': None,
                        'fig': None # Menyiapkan penampung untuk grafik kustom
                    }
                    
                    stdout_buffer = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(stdout_buffer):
                            exec(python_code, globals(), local_env)
                        
                        real_output_df = local_env.get('output_df', None)
                        if real_output_df is None and isinstance(local_env.get('uploaded_df'), pd.DataFrame):
                            real_output_df = local_env.get('uploaded_df')
                        
                        # Render Dataframe Hasil Perhitungan Nyata
                        if isinstance(real_output_df, pd.DataFrame):
                            st.write("---")
                            st.subheader("📊 Dataset Hasil Eksekusi Nyata Backend")
                            st.dataframe(real_output_df, use_container_width=True)
                            
                            csv_bytes = real_output_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Hasil Analisis (CSV)",
                                data=csv_bytes,
                                file_name='alex_real_executed_data.csv',
                                mime='text/csv',
                            )
                        
                        # ==========================================================
                        # 📈 RENDER GRAFIK DINAMIS (Bukan Bar Chart Paksaan Lagi)
                        # ==========================================================
                        custom_fig = local_env.get('fig', None)
                        if custom_fig is not None:
                            st.write("---")
                            st.subheader("📈 Visualisasi Grafik Analisis Kustom")
                            st.plotly_chart(custom_fig, use_container_width=True)
                        else:
                            print_output = stdout_buffer.getvalue()
                            if print_output:
                                st.info("🖥️ **Hasil Perhitungan Konsol:**")
                                st.code(print_output)
                                
                    except Exception as exec_err:
                        st.error(f"Eror saat mengeksekusi kode Python buatan AI: {exec_err}")
                        with st.expander("Lihat Kode yang Gagal"):
                            st.code(python_code)
                            
            except Exception as e:
                st.error(f"Terjadi kendala pada API Groq: {e}")
