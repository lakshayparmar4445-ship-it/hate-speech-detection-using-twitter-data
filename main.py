from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.responses import FileResponse

app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once when server starts
model = joblib.load(r"C:\Users\admin\OneDrive\Desktop\hate speech detection using ML proj internship\hate_speech_model.pkl")
vectorizer = joblib.load(r"C:\Users\admin\OneDrive\Desktop\hate speech detection using ML proj internship\tfidf_vectorizer.pkl")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/predict")
def predict(data: TextInput):

    text_vector = vectorizer.transform([data.text])
    prediction = model.predict(text_vector)[0]
    result = "Hate Speech" if prediction == 1 else "Not Hate Speech"

    return {
        "prediction": int(prediction),
        "result": result
        }

