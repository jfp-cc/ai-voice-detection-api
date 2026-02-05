# AI Voice Detection API

Simple Robust Classifier v1.0 - 90% accuracy

## Deployment on Railway

This API is deployed on Railway using GitHub integration.

### Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check
- `GET /api/v1/health` - API v1 health check
- `POST /api/v1/detect` - AI voice detection endpoint
- `GET /docs` - FastAPI documentation

### Usage

```bash
curl -X POST "https://your-app.railway.app/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "your_base64_audio_data"}'
```

### Model Performance

- **Accuracy**: 90% balanced accuracy
- **Languages**: English, Hindi, Tamil, Malayalam, Telugu
- **Model**: Simple Robust CNN (64x64 input)