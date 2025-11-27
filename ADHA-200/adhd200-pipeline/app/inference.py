import joblib
import nibabel as nib
import numpy as np

# Path to your trained logistic regression model
MODEL_PATH = "models/model_v1.pkl"

# Load the model once at startup
model = joblib.load(MODEL_PATH)

def extract_features(img):
    """
    Extract features from the NIfTI image.
    Replace this with the same feature extraction logic you used in training.
    For now, we use a simple global mean intensity.
    """
    data = img.get_fdata()
    features = [np.mean(data)]
    return np.array(features).reshape(1, -1)

def predict_nifti(nifti_file):
    """
    Run prediction on a NIfTI file.
    """
    img = nib.load(nifti_file)
    X = extract_features(img)
    pred = model.predict(X)[0]
    return int(pred)
