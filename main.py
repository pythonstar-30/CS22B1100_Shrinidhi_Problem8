from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import json

# Import the logic functions from your logic.py file
try:
    from fin_logic import (
        extract_contextual_amounts, 
        extract_contextual_amounts_from_text, 
        label_amounts_with_llm
    )
except ImportError:
    print("CRITICAL: Could not import from logic.py. Ensure the file exists and has no syntax errors.")
    # Define placeholder functions if import fails, so the server can start and report errors
    extract_contextual_amounts = None
    extract_contextual_amounts_from_text = None
    label_amounts_with_llm = None

# --- Part 1: FastAPI App and Request Models ---

app = FastAPI(
    title="Intelligent Invoice Processor API",
    description="An API that uses OCR and an LLM to extract labeled monetary amounts from images and text."
)

class TextRequest(BaseModel):
    """Defines the structure for incoming text requests, e.g., {"text": "Your invoice text here..."}"""
    text: str

# --- Part 2: API Endpoints ---

@app.get("/", summary="Check API Status")
def read_root():
    """A simple endpoint to confirm that the API is running."""
    return {"status": "ok", "message": "Invoice Processor API is running."}

@app.post("/process-image/", summary="Process an Invoice Image")
async def process_invoice_image(file: UploadFile = File(..., description="The invoice image (e.g., PNG, JPG) to process.")):
    """
    Accepts an image file, performs OCR to find amounts and their context,
    then uses an LLM to generate a final, labeled JSON output.
    """
    if not all([extract_contextual_amounts, label_amounts_with_llm]):
        raise HTTPException(status_code=503, detail="Server logic is not available or failed to load.")
        
    # Read the image file from the request
    image_bytes = await file.read()
    
    # Step 1: Extract contextual data from the image
    contextual_data = extract_contextual_amounts(image_bytes)
    
    # Step 2: Use the LLM to get the final labeled JSON
    labeled_data = label_amounts_with_llm(contextual_data)
    
    return labeled_data

@app.post("/process-text/", summary="Process Invoice Text")
async def process_invoice_text(request: TextRequest):
    """
    Accepts a JSON object with a 'text' field, finds amounts and their context,
    then uses an LLM to generate a final, labeled JSON output.
    """
    if not all([extract_contextual_amounts_from_text, label_amounts_with_llm]):
        raise HTTPException(status_code=503, detail="Server logic is not available or failed to load.")

    # Step 1: Extract contextual data from the provided text
    contextual_data = extract_contextual_amounts_from_text(request.text)
    
    with open("process_text.json", 'w') as f:
        json.dump(contextual_data, f, indent=4)
    # Step 2: Use the LLM to get the final labeled JSON
    labeled_data = label_amounts_with_llm(contextual_data)
    with open("labelled_text.json", 'w') as f:
        json.dump(labeled_data, f, indent=4)
    
    return labeled_data

if __name__ == "__main__":
    # This makes the script runnable with "python main.py"
    uvicorn.run(app, host="0.0.0.0", port=8000)

