from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
import tempfile
import shutil
from pathlib import Path
from processor import process_imu_data

app = FastAPI(title="Brokkr's Eye", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process IMU data CSV file
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    try:
        # Create temporary file for input
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_input:
            # Save uploaded file
            content = await file.read()
            temp_input.write(content)
            temp_input_path = temp_input.name

        # Generate output filename
        base_name = os.path.splitext(file.filename)[0]
        output_filename = f"{base_name}_processed.csv"
        output_path = UPLOAD_DIR / output_filename

        # Process the data
        result_path = process_imu_data(
            temp_input_path,
            str(output_path),
            progress_callback=None  # No progress callback for web
        )

        # Clean up temporary input file
        os.unlink(temp_input_path)

        return {
            "status": "success",
            "message": "File processed successfully",
            "filename": output_filename,
            "download_url": f"/download/{output_filename}"
        }

    except Exception as e:
        # Clean up on error
        if 'temp_input_path' in locals():
            try:
                os.unlink(temp_input_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download processed file
    """
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='text/csv'
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Brokkr's Eye API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
