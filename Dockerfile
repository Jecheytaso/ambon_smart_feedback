# Gunakan base image Python (versi 3.9 atau lebih baru umumnya bagus)
FROM python:3.9-slim

# Set direktori kerja di dalam container
WORKDIR /code

# ---> UBAH BARIS INI <---
# Set lokasi cache Hugging Face ke /tmp (direktori sementara yang pasti bisa ditulis)
ENV HF_HOME=/tmp/hf_cache
# ------------------------

# Salin file requirements dulu agar Docker bisa cache layer ini
COPY ./requirements.txt /code/requirements.txt

# Install dependencies Python dari requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin semua file aplikasi Anda (app.py, html, js, css, dll.) ke container
COPY . /code/

# Beri tahu Docker port mana yang akan digunakan aplikasi Anda
# Hugging Face Spaces biasanya menggunakan port 7860
EXPOSE 7860

# Perintah untuk menjalankan aplikasi Anda menggunakan Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

