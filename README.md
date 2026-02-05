# AI Voice Detection API

Simple Robust Classifier v1.0 with 90% accuracy for detecting AI-generated vs human voices.

## Features
- FastAPI-based REST API
- Simple Robust CNN model (90% accuracy)
- Multi-language support (English, Hindi, Tamil, Malayalam, Telugu)
- Real-time audio classification

## Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /api/v1/health` - API v1 health check
- `POST /api/v1/detect` - Audio classification endpoint

## Deployment
This API is deployed on Railway with automatic scaling and HTTPS.

## Usage
Send a POST request to `/api/v1/detect` with:
```json
{
  "audio_base64": "base64_encoded_audio_data"
}
```

Returns:
```json
{
  "classification": "ai_generated" | "human",
  "confidence_score": 0.85,
  "confidence_level": "high",
  "explanation": "Audio classified as AI-generated voice...",
  "language_detected": "en",
  "processing_time_ms": 250
}
```