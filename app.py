# Tambahkan ini DI AWAL SANGAT ATAS
print("--- app.py execution started ---")

try:
    # Import libraries satu per satu untuk isolasi error
    print("Importing FastAPI...")
    from fastapi import FastAPI
    print("Importing BaseModel...")
    from pydantic import BaseModel
    print("Importing transformers...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print("Importing torch...")
    import torch
    print("--- Imports successful ---")

    # === 1. Inisialisasi Aplikasi FastAPI ===
    print("Initializing FastAPI app...")
    app = FastAPI()
    print("FastAPI app initialized.")

    # === 2. Pemuatan Model (Global) ===
    MODEL_NAME = "Jechey/best-indobert-model"
    print(f"Checking for CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Attempting to load tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("--- Tokenizer loaded successfully ---")

    print(f"Attempting to load model from: {MODEL_NAME}")
    # Muat model ke memori utama dulu sebelum dipindah ke device
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("--- Model loaded successfully (to default device/RAM) ---")
    print(f"Moving model to device: {device}")
    model.to(device)
    print("--- Model moved to device successfully ---")
    model.eval()
    print("--- Model set to eval mode ---")

    # === 3. Mapping ID Label ke Nama Label ===
    # Pastikan urutan ini sudah benar sesuai output label_encoder.classes_
    label_map = [
        "Dinas Kominfo",
        "Dinas LHP",
        "Dinas PUPR",
        "Dinas Pendidikan",
        "Dinas Perhubungan"
    ]
    print(f"Label map loaded: {label_map}")

    # === 4. Pydantic Model untuk Input ===
    class InputText(BaseModel):
        text: str
    print("InputText model defined.")

    # === 5. Fungsi Prediksi ===
    def get_model_prediction(text_input: str) -> str:
        print(f"get_model_prediction called with text: '{text_input[:50]}...'") # Log input pendek
        try:
            print("Tokenizing input...")
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128 # Sesuaikan jika berbeda saat training
            )
            print("Moving inputs to device...")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            print("Performing inference...")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            print("Inference complete.")

            predicted_class_id = logits.argmax(dim=-1).item()
            print(f"Predicted class ID: {predicted_class_id}")

            if 0 <= predicted_class_id < len(label_map):
                result = label_map[predicted_class_id]
                print(f"Mapped prediction: {result}")
                return result
            else:
                print(f"Error: Predicted ID out of range: {predicted_class_id}")
                return "Error: ID Prediksi Tidak Dikenal"
        except Exception as e_inner:
            print(f"!!! EXCEPTION in get_model_prediction: {e_inner}")
            # Tambahkan traceback jika perlu detail lebih lanjut
            import traceback
            print(traceback.format_exc()) # Cetak detail error internal prediksi
            return "Error: Proses prediksi gagal internal"

    # === 6. FastAPI Endpoint ===
    @app.get("/")
    async def root():
        print("Root endpoint '/' accessed.")
        return {"message": "Server FastAPI berjalan. Gunakan endpoint /predict untuk klasifikasi."}

    @app.post("/predict")
    async def predict(data: InputText):
        print(f"Endpoint '/predict' accessed with text: '{data.text[:50]}...'") # Log input pendek
        hasil_prediksi = get_model_prediction(data.text)
        print(f"Returning prediction: {hasil_prediksi}")
        return {"prediction": hasil_prediksi}

    print("--- Endpoint definitions complete ---")
    print("--- app.py setup seems complete. Uvicorn should take over now. ---")

except Exception as e_outer:
    # Tangkap error yang terjadi DI LUAR fungsi/endpoint (saat startup, misal load model)
    print(f"!!! FATAL STARTUP EXCEPTION: {e_outer}")
    # Tambahkan traceback untuk detail error startup
    import traceback
    print(traceback.format_exc())
    # Kita bisa coba raise lagi agar HF tahu ada masalah serius saat startup
    raise e_outer

