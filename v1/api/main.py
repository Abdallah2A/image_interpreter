from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from api.inference import generate_caption
from constants import DEVICE, WORKING_DIR
import os

app = FastAPI(title="Image Captioning API", description="Generate captions for images using a trained model.")

# Load model and artifacts (simplified for demo; in practice, load from ZenML artifact store)
model_path = os.path.join(WORKING_DIR, 'best_model.pth')
features_path = os.path.join(WORKING_DIR, 'features.pkl')
tokenizer_path = os.path.join(WORKING_DIR, 'tokenizer.pkl')


@app.post("/predict", summary="Generate caption for an uploaded image")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Generate caption
        caption = generate_caption(image, model_path, tokenizer_path, DEVICE)

        return JSONResponse(content={"caption": caption}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health", summary="Check API health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
