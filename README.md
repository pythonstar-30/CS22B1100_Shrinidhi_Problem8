# CS22B1100_Shrinidhi_Problem8

# Intelligent Invoice Processor API

This project is a Python-based API built with FastAPI that uses Optical Character Recognition (OCR) and a Large Language Model (LLM) to extract and label monetary amounts from images and text.

---
##Features

* **Process Images:** Upload an invoice image (`.png`, `.jpg`, etc.) to extract labeled amounts like `sub_total`, `tax`, and `amount_due`.
* **Process Text:** Send raw text from an invoice to get the same structured JSON output.
* **AI-Powered Labeling:** Uses `easyocr` for text extraction and a local LLM (`phi3:3.8b` via Ollama) to intelligently label the extracted values.

---
##  Setup and Installation

1.  **Prerequisites:**
    * Python 3.8+
    * Ollama installed and running with the `phi3:3.8b` model (`ollama run phi3:3.8b`).
    * Git (for cloning the repository).

2.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/pythonstar-30/CS22B1100_Shrinidhi_Problem8>
    cd <CS22B1100_Shrinidhi_Problem8>
    ```

3.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

4.  **Run the Server:**
    ```bash
    python main.py
    ```
    The API will be available at `http://127.0.0.1:8000`.

---
##  API Usage

The API exposes two main endpoints for processing invoices.

### **1. Process an Invoice Image**

This endpoint accepts an image file and returns a structured JSON of the amounts found.

* **URL:** `/process-image/`
* **Method:** `POST`
* **Body:** `multipart/form-data`

#### **Example using `curl`:**
```bash
curl -X POST -F "file=@/path/to/your/invoice.png" "http://127.0.0.1:8000/process-image/"

```
### **2. Process an invoice text**
This endpoint accepts a JSON object with raw text and returns the same structured JSON output.


* **URL:** `/process-text/`
* **Method:** `POST`
* **Body:** `raw (JSON)`

#### **Example using `curl`:**
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"text": "The subtotal is $50.00 and the tax is $5.00, making the total amount $55.00"}' \
"http://127.0.0.1:8000/process-text/"

```
