import streamlit as st
import pandas as pd
import plotly.express as px
import json
import re
from groq import Groq  # Jalankan 'pip install groq'

st.set_page_config(page_title="Alex Data Science Workspace", layout="wide")

# Ambil API key secara aman dari Streamlit Secrets
@st.cache_resource
def get_groq_client():
    # Pastikan Anda memasukkan GROQ_API_KEY di settings Streamlit Cloud nanti
    api_key = st.secrets["GROQ_API_KEY"]
    return Groq(api_key=api_key)

try:
    client = get_groq_client()
except Exception as e:
    st.error("API Key belum disetel di Streamlit Secrets!")
    st.stop()

st.title("🤖 Alex - Data Science Workspace (Cloud Llama)")
st.caption("Zero-cost UI, Automated Dataset Generation, and Instant Visualization")
st.write("---")

user_query = st.text_area(
    "Masukkan perintah / pertanyaan task Data Science kamu di sini:", 
    placeholder="Contoh: Buatkan data tren penjualan 5 produk teratas selama 6 bulan terakhir beserta grafiknya."
)

if st.button("Proses Data & Visualisasikan 🚀", type="primary"):
    if user_query.strip() == "":
        st.warning("Ketik sesuatu dulu ya!")
    else:
        with st.spinner("Alex sedang memproses data dan membuat visualisasi..."):
            system_prompt = (
                "You are Alex, a Senior Data Scientist. Answer the user's question naturally in the language they use (Indonesian or English).\n"
                "CRITICAL INSTRUCTION: If the user requests data, generation, or statistics, you MUST append a valid JSON array of objects representing the data at the very end of your response, wrapped inside ```json and ``` blocks. "
                "The JSON must be tabular-friendly (flat key-value pairs). Do not mix prose inside the JSON block."
            )
            
            try:
                # Panggil Llama 3 via Groq (Gratis & Sangat Cepat)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    model="llama3-8b-8192",  # Bisa diganti sesuai model Llama yang tersedia di Groq
                )
                full_reply = chat_completion.choices[0].message.content
                
                json_match = re.search(r'```json\s*(.*?)\s*```', full_reply, re.DOTALL)
                explanation_text = re.sub(r'```json\s*.*?\s*```', '', full_reply, flags=re.DOTALL).strip()
                
                st.subheader("💡 Analisis & Jawaban Alex")
                st.write(explanation_text)
                
                if json_match:
                    json_data_str = json_match.group(1)
                    data = json.loads(json_data_str)
                    df = pd.DataFrame(data)
                    
                    st.write("---")
                    st.subheader("📊 Dataset Hasil Generasi (CSV)")
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Dataset (CSV)",
                        data=csv,
                        file_name='alex_generated_dataset.csv',
                        mime='text/csv',
                    )
                    
                    st.write("---")
                    st.subheader("📈 Visualisasi Grafik Otomatis")
                    
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    string_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if len(string_cols) >= 1 and len(numeric_cols) >= 1:
                        fig = px.bar(df, x=string_cols[0], y=numeric_cols[0], 
                                     title=f"Grafik Hubungan {string_cols[0]} vs {numeric_cols[0]}",
                                     template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(numeric_cols) >= 2:
                        fig = px.line(df, x=df.index, y=numeric_cols[0], title="Tren Data")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Data berhasil dibuat, namun format kolom terlalu unik untuk grafik otomatis.")
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan sistem: {e}")