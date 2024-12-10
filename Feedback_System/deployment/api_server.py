from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
from pathlib import Path
from inference import SingleImageInference

app = FastAPI(title="Technical Drawing Analysis API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
TEMP_DIR = Path("temp_uploads").absolute()
TEMP_DIR.mkdir(exist_ok=True)

MODEL_PATH = "/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/LVmodule/Multifusion/checkpoints04/best_model.pt"
inference_model = SingleImageInference(MODEL_PATH)


@app.post("/analyze")
async def analyze_drawing(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = TEMP_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        results = inference_model.predict(str(file_path))

        # Generate visualization
        vis_filename = f"vis_{file.filename}"
        vis_path = TEMP_DIR / vis_filename
        inference_model.visualize_results(str(file_path), results, str(vis_path))

        return JSONResponse({
            "status": "success",
            "predictions": results,
            "visualization_path": f"/visualization/{vis_filename}"
        })

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.get("/visualization/{filename}")
async def get_visualization(filename: str):
    file_path = TEMP_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    return FileResponse(path=str(file_path), media_type="image/jpeg")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)