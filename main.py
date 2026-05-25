# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
from scorer import score_dataframe

app = FastAPI(title="Data Quality Scorer API", version="1.0")

@app.post("/score")
async def score_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not parse CSV.")
    if df.empty or len(df.columns) == 0:
        raise HTTPException(status_code=422, detail="CSV is empty.")
    return score_dataframe(df)

@app.get("/health")
def health():
    return {"status": "ok"}