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
    print("Importing StaticFiles...")
    from fastapi.staticfiles import StaticFiles
    print("Importing FileResponse...")
    from fastapi.responses import FileResponse
    print("--- Imports successful ---")

    # === 1. Inisialisasi Aplikasi FastAPI ===
    print("Initializing FastAPI app...")
    app = FastAPI()
    print("--- FastAPI app initialized ---")

    # === 2. Pemuatan Model (Global) ===
    MODEL_NAME = "Jechey/best-indobert-model"
    print(f"Setting device (checking CUDA)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Attempting to load tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("--- Tokenizer loaded successfully ---")

    print(f"Attempting to load model from: {MODEL_NAME} to {device}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval() # Set model ke mode evaluasi
    print("--- Model loaded successfully and set to eval mode ---")

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
    # Catatan: print ini seharusnya tidak di dalam class, dipindahkan ke luar
    print("--- InputText model defined ---")

    # === 5. Fungsi Prediksi ===
    def get_model_prediction(text_input: str) -> str:
        print(f"--- get_model_prediction called with input: '{text_input[:50]}...' ---") # Log input awal
        try:
            print("Starting tokenization...")
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            print("Tokenization complete.")

            print("Moving inputs to device...")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print("Inputs moved to device.")

            print("Starting model inference...")
            with torch.no_grad():
                outputs = model(**inputs)
            print("Model inference complete.")

            logits = outputs.logits
            # Perbaiki typo: predicted_class_id bukan redicted_class_id
            predicted_class_id = logits.argmax(dim=-1).item()
            print(f"Predicted class ID: {predicted_class_id}")

            if 0 <= predicted_class_id < len(label_map):
                prediction = label_map[predicted_class_id]
                print(f"Mapped prediction: {prediction}")
                return prediction
            else:
                print(f"!!! ERROR: Predicted ID out of range: {predicted_class_id}")
                return "Error: ID Prediksi Tidak Dikenal"

        except Exception as e_pred:
            print(f"!!! EXCEPTION in get_model_prediction: {e_pred}")
            # Log traceback jika perlu detail lebih lanjut
            import traceback
            traceback.print_exc()
            return "Error: Proses prediksi gagal"

    # === 6. Konfigurasi Static Files ===
    # Mount direktori root ('.') ke path URL '/static'
    # Pastikan path di HTML (css, js, img) menggunakan /static/
    print("Mounting static files...")
    app.mount("/static", StaticFiles(directory="."), name="static")
    print("--- Static files mounted to /static ---")


    # === 7. FastAPI Endpoints ===
    @app.get("/")
    async def serve_home():
        print("Root endpoint '/' accessed, serving deteksi.html")
        # Ganti "deteksi.html" dengan "index.html" jika itu halaman utama Anda
        return FileResponse("deteksi.html")

    @app.post("/predict")
    async def predict(data: InputText):
        print(f"Predict endpoint '/predict' accessed with text: '{data.text[:50]}...'")
        hasil_prediksi = get_model_prediction(data.text)
        print(f"Returning prediction: {hasil_prediksi}")
        return {"prediction": hasil_prediksi}

    print("--- app.py setup seems complete. Uvicorn should take over now. ---")

except Exception as e_global:
    # Tangkap error fatal saat startup (misal gagal load model)
    print(f"!!! FATAL STARTUP EXCEPTION: {e_global}")
    # Log traceback untuk detail
    import traceback
    traceback.print_exc()
    # Coba exit agar HF tahu ada masalah (mungkin tidak selalu efektif)
    import sys
    sys.exit(1)

