from fastapi import FastAPI, UploadFile
from app.inference import predict_nifti

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    # Save uploaded file temporarily
    with open("temp.nii", "wb") as f:
        f.write(await file.read())
    pred = predict_nifti("temp.nii")
    return {"prediction": pred}
