    # Hanya import yang paling dasar
    from fastapi import FastAPI
    
    print("--- Minimal app.py started ---") # Pesan debug sederhana
    
    app = FastAPI()
    
    @app.get("/")
    def read_root():
        print("Root endpoint '/' accessed (minimal app).")
        return {"message": "Minimal FastAPI app is running!"}
    
    # Endpoint predict sederhana tanpa model
    @app.post("/predict")
    async def predict_simple():
        print("Predict endpoint '/predict' accessed (minimal app).")
        return {"prediction": "Test Berhasil (Server Minimal)"}
    
    print("--- Minimal app setup complete. Uvicorn should take over. ---")
    

