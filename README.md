# Brokkr's Eye - Web Application

A FastAPI-based web application for processing IMU (Inertial Measurement Unit) sensor data and generating yaw rate classifications using a trained neural network model.

## Features

- **Modern Web Interface**: Clean, responsive design with drag-and-drop file upload
- **Real-time Processing**: Upload CSV files and get processed results instantly
- **RESTful API**: FastAPI backend with automatic API documentation
- **Download Results**: Processed files available for immediate download
- **No Installation Required for Users**: Access via web browser

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   # Windows
   run.bat

   # Or manually
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the Application**:
   - Open your browser to: `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`

## Usage

### Web Interface

1. Navigate to `http://localhost:8000`
2. Drag and drop your CSV file or click "Select File"
3. Click "Process Data"
4. Wait for processing to complete
5. Download your processed file

### API Endpoints

#### Upload and Process File
```http
POST /upload
Content-Type: multipart/form-data

file: <your-csv-file>
```

Response:
```json
{
  "status": "success",
  "message": "File processed successfully",
  "filename": "data_processed.csv",
  "download_url": "/download/data_processed.csv"
}
```

#### Download Processed File
```http
GET /download/{filename}
```

#### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "Brokkr's Eye API"
}
```

## Input File Format

The input CSV file must contain the following columns:
- `Timestamp`: Date and time (format: YYYY-MM-DD HH:MM:SS)
- `Milliseconds`: Milliseconds component
- `LinLeft`: Left wheel linear encoder reading
- `LinRight`: Right wheel linear encoder reading
- `AccX`, `AccY`, `AccZ`: Accelerometer readings (g)
- `GyroX`, `GyroY`, `GyroZ`: Gyroscope readings (deg/s)

Example:
```csv
Timestamp,Milliseconds,LinLeft,LinRight,AccX,AccY,AccZ,GyroX,GyroY,GyroZ
2025-10-05 17:32:52,513,20285,20244,0.992,-0.037,-0.097,0.488,0.122,-0.305
```

## Output File Format

The output CSV contains all original columns plus:
- `time(s)`: Time in seconds from start
- `delta_yaw(deg)`: Change in yaw angle
- `dt(s)`: Time delta between measurements
- `yaw_rate(deg/s)`: Yaw rate
- `turn_type`: Classification (straight, slight left/right, normal left/right, major left/right)

## Project Structure

```
webapp/
├── main.py                      # FastAPI application
├── processor.py                 # Core processing logic
├── model.py                     # Neural network model
├── Model_A_B500_E300_V2.hdf5   # Trained model weights
├── requirements.txt             # Python dependencies
├── run.bat                      # Startup script
├── templates/
│   └── index.html              # Web interface
├── static/
│   ├── style.css               # Styling
│   └── script.js               # Client-side logic
└── uploads/                     # Processed files (auto-created)
```

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload
```

### Running in Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment

### Docker (Optional)
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t brokkrs-eye .
docker run -p 8000:8000 brokkrs-eye
```

## Technologies

- **Backend**: FastAPI, Uvicorn
- **ML/AI**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## License

Part of the MjolnirNN research project.

## Version

v1.0.0 - Initial web application release
