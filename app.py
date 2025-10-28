from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# === 1. Inisialisasi Aplikasi FastAPI ===
app = FastAPI()

# === 2. Pemuatan Model (Global) ===
# Model dimuat sekali saat aplikasi/Space dimulai.
# Ini JAUH lebih efisien daripada memuatnya di setiap request.

MODEL_NAME = "Jechey/best-indobert-model" 

# Cek apakah GPU tersedia (jika Anda menggunakan GPU di Space)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Memuat tokenizer dari: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Memuat model dari: {MODEL_NAME} ke {device}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval() # Set model ke mode evaluasi (penting untuk inferensi)
print("Model dan tokenizer berhasil dimuat.")


# === 3. Mapping ID Label ke Nama Label ===
# !!! INI BAGIAN PALING PENTING !!!
# Model Anda mengeluarkan angka (misal: 0, 1, 2, 3, 4).
# Kita perlu memetakannya kembali ke nama label (misal: "Dinas Kominfo").
# Urutan di sini HARUS SAMA PERSIS dengan urutan saat Anda melatih LabelEncoder.
#
# Untuk menemukan urutan yang benar, buka notebook Colab Anda
# dan jalankan:
# print(label_encoder.classes_)
#
# Ganti daftar di bawah ini dengan output dari perintah tersebut.
# (Saya hanya menebak urutannya berdasarkan fungsi dummy Anda)

label_map = [
    "Dinas Kominfo",
    "Dinas LHP",
    "Dinas PUPR",
    "Dinas Pendidikan",
    "Dinas Perhubungan"
]
print(f"Label map berhasil dimuat: {label_map}")


# === 4. Pydantic Model untuk Input ===
class InputText(BaseModel):
    text: str

# === 5. Fungsi Prediksi (Pengganti Dummy) ===
def get_model_prediction(text_input: str) -> str:
    """
    Fungsi ini mengambil teks input mentah,
    menjalankannya melalui model, dan mengembalikan prediksi label (string).
    """
    try:
        # 1. Tokenisasi: Ubah teks menjadi angka
        # return_tensors="pt" berarti kita ingin hasil berupa PyTorch Tensors
        inputs = tokenizer(
            text_input, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128 # Sesuaikan max_length jika perlu
        )

        # 2. Pindahkan data token ke device (GPU/CPU)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3. Jalankan Model
        # torch.no_grad() memberitahu model untuk tidak menghitung gradien
        # (ini menghemat memori dan mempercepat inferensi)
        with torch.no_grad():
            outputs = model(**inputs)

        # 4. Dapatkan Logits (skor mentah dari model)
        logits = outputs.logits

        # 5. Dapatkan ID Prediksi
        # Terapkan argmax untuk menemukan indeks (ID) dengan skor tertinggi
        # .item() mengubah tensor (misal: tensor[3]) menjadi angka (misal: 3)
        predicted_class_id = logits.argmax(dim=-1).item()

        # 6. Petakan ID ke Nama Label
        if 0 <= predicted_class_id < len(label_map):
            return label_map[predicted_class_id]
        else:
            print(f"Error: Model memprediksi ID di luar jangkauan: {predicted_class_id}")
            return "Error: ID Prediksi Tidak Dikenal"

    except Exception as e:
        print(f"Terjadi error saat prediksi: {e}")
        return "Error: Proses prediksi gagal"

# === 6. FastAPI Endpoint ===

@app.get("/")
async def root():
    # Endpoint sederhana untuk mengecek apakah server berjalan
    return {"message": "Server FastAPI berjalan. Gunakan endpoint /predict untuk klasifikasi."}


@app.post("/predict")
async def predict(data: InputText):
    """
    Endpoint utama yang menerima teks dan mengembalikan prediksi.
    """
    print(f"Menerima teks untuk diprediksi: {data.text}")
    
    # Panggil fungsi prediksi model
    hasil_prediksi = get_model_prediction(data.text)
    
    print(f"Hasil prediksi: {hasil_prediksi}")
    return {"prediction": hasil_prediksi}
