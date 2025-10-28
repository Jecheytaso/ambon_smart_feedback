# Gunakan base image Python (versi 3.9 atau lebih baru umumnya bagus)
FROM python:3.9-slim

# Set direktori kerja di dalam container
WORKDIR /code

# Salin file requirements dulu agar Docker bisa cache layer ini
COPY ./requirements.txt /code/requirements.txt

# Install dependencies Python dari requirements.txt
# --no-cache-dir: Tidak menyimpan cache download pip
# --upgrade pip: Memastikan pip versi terbaru
RUN pip install --no-cache-dir --upgrade pip -r /code/requirements.txt

# Salin semua file aplikasi Anda (app.py, html, js, css, dll.) ke container
COPY . /code/

# Beri tahu Docker port mana yang akan digunakan aplikasi Anda
# Hugging Face Spaces biasanya menggunakan port 7860
EXPOSE 7860

# =========================================================
# BARIS INI PENTING DAN SEBELUMNYA HILANG:
# Jalankan aplikasi menggunakan Uvicorn
# app:app -> Cari file bernama app.py, di dalamnya cari objek bernama app
# --host 0.0.0.0 -> Agar bisa diakses dari luar container
# --port 7860 -> Port standar yang biasanya didengarkan oleh HF Spaces
# =========================================================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

