# Ds-Assistant-System-Streamlit
Helpin u to learn about Data Science

https://dsassistanttolearn.streamlit.app/

**TASK**
Alex, please group the customers in this CSV dataset into 3 clusters using the K-Means algorithm based on the 'Tenure_Months' and 'Monthly_Charges' columns. Save the results into output_df with a new column named 'Cluster'. After that, create a scatter plot using Plotly Express and save it to the variable fig. Set the X-axis to 'Tenure_Months', the Y-axis to 'Monthly_Charges', and color the dots based on the 'Cluster' column. Make sure to convert the 'Cluster' column data type to string so that the legend colors are clearly separated.

**SOAL:**
"Alex, tolong kelompokkan pelanggan di data CSV ini menjadi 3 cluster menggunakan algoritma K-Means berdasarkan kolom 'Tenure_Months' dan 'Monthly_Charges'. Simpan hasilnya ke dalam output_df dengan kolom baru bernama 'Cluster'. Setelah itu, buatlah grafik sebaran (scatter plot) menggunakan Plotly Express yang disimpan ke variabel fig. Atur agar sumbu X adalah 'Tenure_Months', sumbu Y adalah 'Monthly_Charges', dan warnai titiknya (color) berdasarkan kolom 'Cluster'. Pastikan tipe data kolom Cluster diubah menjadi string agar warna legendanya terpisah dengan jelas."

<img width="629" height="355" alt="image" src="https://github.com/user-attachments/assets/b1716355-a382-4815-a7b2-9e3dba5cdb36" />

<img width="629" height="395" alt="image" src="https://github.com/user-attachments/assets/21e75159-5b79-4faa-a904-885285eb8e08" />

**Explanation (English):**

**1. Red Group (Cluster 1) – Crisis Zone 🚨**
Profile: New customers (joined less than 10 months) who are immediately charged at a premium rate (ranging from $75 to $110).

Business Impact: This is our highest churn-risk segment. They lack brand loyalty and are currently burdened with high costs.

Recommended Action: We must urgently implement loyalty promotions, exclusive discounts, or proactive Customer Service outreach to improve their experience and prevent them from switching to competitors.

**2. Green Group (Cluster 0) – Stable/Budget Zone 🪙**
Profile: Mid-tenure customers (approximately 10 to 25 months) who prefer low-cost plans (under $50).

Business Impact: This segment is generally stable and secure, as they feel their current plan offers good value for their budget.

Recommended Action: This group is an ideal target for upselling campaigns (gradually introducing them to slightly higher-tier plans).

**3. Blue Group (Cluster 2) – Loyal "Premium" Customers 💎**
Profile: Our most loyal customers! They have been with us for a long time (30+ months, up to 60 months). Their spending varies, with many willing to pay premiums of up to $90+.

Business Impact: This is our primary revenue driver (cash cow) and must be prioritized for retention at all costs.

Recommended Action: Implement appreciation programs such as Loyalty Rewards, VIP access, or exclusive features to ensure they feel valued and continue their long-term contracts.

🎯 Executive Summary
"Our primary focus for the next month is to rescue the Red Group. By optimizing their costs or enhancing their service experience, we can transition them into long-term, loyal customers similar to our Blue Group."

**Explanation (Bahasa Indonesia):**

**1. Kelompok Merah (Cluster 1) – Zona Bahaya / Krisis 🚨**
**Siapa mereka? **
Pelanggan baru (baru bergabung kurang dari 10 bulan) tapi langsung dikenakan biaya bulanan yang sangat mahal (berkisar antara $75 hingga $110).

**Dampak Bisnis:** Ini adalah kelompok paling rentan kabur (high-risk churn). Mereka belum loyal, tapi sudah dibebani biaya tinggi.

**Rekomendasi Aksi:** Kita perlu segera memberikan promo loyalty, diskon khusus, atau treatment ekstra dari Customer Service agar mereka tidak kapok dan pindah ke kompetitor.

**2. Kelompok Hijau (Cluster 0) – Zona Hemat / Stabil 🪙**
**Siapa mereka? **
Pelanggan dengan masa langganan jangka menengah (sekitar 10 hingga 25 bulan) yang memilih paket dengan biaya bulanan rendah (di bawah $50).

**Dampak Bisnis:** Kelompok ini cenderung stabil dan aman karena mereka merasa mendapatkan harga yang murah dan pas di kantong.

**Rekomendasi Aksi:** Kelompok ini bisa menjadi target empuk untuk program upselling (ditawari naik ke paket yang sedikit lebih mahal secara bertahap).

**3. Kelompok Biru (Cluster 2) – Kelompok "Sultan" Loyal 💎**
**Siapa mereka? **
Pelanggan paling setia kita! Mereka sudah bertahan sangat lama (di atas 30 bulan, bahkan ada yang sampai 60 bulan). Pengeluaran mereka bervariasi, dan banyak yang rela membayar mahal hingga $90+.

**Dampak Bisnis:** Ini adalah sumber pendapatan utama (cash cow) perusahaan yang wajib kita pertahankan mati-matian.

**Rekomendasi Aksi:** Berikan program apresiasi seperti Loyalty Rewards, akses VIP, atau fitur eksklusif agar mereka tetap merasa dihargai dan terus memperpanjang kontrak.

