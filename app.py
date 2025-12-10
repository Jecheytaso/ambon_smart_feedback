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
    label_map = [
        "Dinas Kominfo",
        "Dinas Lingkungan Hidup dan Persampahan",
        "Dinas Pekerjaan Umum dan Penataan Ruangan",
        "Dinas Pendidikan",
        "Dinas Perhubungan"
    ]
    print(f"Label map loaded: {label_map}")

    # === 4. Pydantic Model untuk Input ===
    class InputText(BaseModel):
        text: str
    print("--- InputText model defined ---") # Pindahkan print ke luar class

    # === 5. Fungsi Prediksi ===
    def get_model_prediction(text_input: str) -> str:
        print(f"--- get_model_prediction called with input: '{text_input[:50]}...' ---")
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
                predicted_class_id = logits.argmax(dim=-1).item() # Typo diperbaiki
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
            import traceback
            traceback.print_exc()
            return "Error: Proses prediksi gagal"

    # === 6. Konfigurasi Static Files ===
    # Mount direktori root ('.') ke path URL '/static'
    print("Mounting static files...")
    app.mount("/static", StaticFiles(directory="."), name="static")
    print("--- Static files mounted to /static ---")

    # === 7. FastAPI Endpoints ===

    # Endpoint untuk halaman Home (index.html)
    @app.get("/", response_class=FileResponse)
    async def serve_index():
        print("Root endpoint '/' accessed, serving index.html")
        return FileResponse("index.html")

    @app.get("/index", response_class=FileResponse)
    async def serve_index_explicit():
        print("Endpoint '/index' accessed, serving index.html")
        return FileResponse("index.html")

    # Endpoint untuk halaman Deteksi (deteksi.html)
    @app.get("/deteksi", response_class=FileResponse)
    async def serve_deteksi():
        print("Endpoint '/deteksi' accessed, serving deteksi.html")
        return FileResponse("deteksi.html")

    # Endpoint untuk halaman Tentang (about.html)
    @app.get("/about", response_class=FileResponse)
    async def serve_about():
        print("Endpoint '/about' accessed, serving about.html")
        return FileResponse("about.html")

    # Endpoint untuk halaman Bantuan (bantuan.html)
    @app.get("/bantuan", response_class=FileResponse)
    async def serve_bantuan():
        print("Endpoint '/bantuan' accessed, serving bantuan.html")
        return FileResponse("bantuan.html")

    # Endpoint API untuk prediksi (tetap sama)
    @app.post("/predict")
    async def predict(data: InputText):
        print(f"Predict endpoint '/predict' accessed with text: '{data.text[:50]}...'")
        hasil_prediksi = get_model_prediction(data.text)
        print(f"Returning prediction: {hasil_prediksi}")
        return {"prediction": hasil_prediksi}

    print("--- app.py setup seems complete. Uvicorn should take over now. ---")

except Exception as e_global:
    # Tangkap error fatal saat startup
    print(f"!!! FATAL STARTUP EXCEPTION: {e_global}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

