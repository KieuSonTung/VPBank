import sys
sys.path.append('..')

# from src.model.lgbm import LGBM
from src.preprocess import preprocess_pipeline

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import os
import joblib


app = FastAPI()

# Load the pre-trained LightGBM model
model = joblib.load('../weights/best_lgbm1.pkl')

# Load Label Encoders
loaded_label_encoders = joblib.load('../weights/label_encoders/label_encoders.pkl')


@app.post("/infer")
async def infer(file: UploadFile = File(...), download: bool = False):
    
    # Save the uploaded file to a temporary location
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No file selected for uploading")
    
    # Read the Excel file
    try:
        df = pd.read_excel(file.file.read())

        df = preprocess_pipeline(df)

        for column, le in loaded_label_encoders.items():
            df[column] = le.transform(df[column])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")
    
    ids = df['id']
    
    # Perform inference
    probs = model.predict(df)

    # Add predictions to the DataFrame
    result_df = pd.DataFrame({
        'id': ids,
        'probability': probs
    })

    # Save the updated DataFrame to a new Excel file
    output_path = 'output.xlsx'
    result_df.to_excel(output_path, index=False)

    # Return the updated file
    if download:
        return FileResponse(output_path, filename='output.xlsx')
    else:
        return JSONResponse(content={"message": "Inference completed", "output_path": output_path})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
