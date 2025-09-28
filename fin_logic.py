import json
import re
import easyocr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --- Part 1: AI and Model Configuration ---
print("Loading EasyOCR and LLM models...")
reader = easyocr.Reader(['en'])
# IMPORTANT: format="json" helps ensure the LLM provides valid JSON
llm = ChatOllama(model="phi3:3.8b", temperature=0, format="json")
print("Models loaded.")


# --- Part 2: Core Logic ---

def clean_monetary_value(value: str) -> str:
    """Corrects common OCR errors in a string that represents a number."""
    symbol = value[0]
    numeric_part = value[1:]
    corrections = {
        'O': '0', 'o': '0', 'l': '1', 'I': '1', 'Z': '2',
        'g': '9', 'q': '9', 'B': '8'
    }
    for error, correction in corrections.items():
        numeric_part = numeric_part.replace(error, correction)
    return f"{symbol}{numeric_part}"

def extract_contextual_amounts(image_bytes: bytes):
    """
    Performs OCR, finds monetary amounts, cleans them, and identifies their context.
    """
    all_fragments = reader.readtext(image_bytes, paragraph=False)
    
    money_pattern = r"(\$|S|€|£|₹|INR|USD|EUR|GBP)\s?([\d,OolISZgqB\.]+)"
    
    amount_fragments = []
    other_fragments = []

    for bbox, text, conf in all_fragments:
        if re.fullmatch(money_pattern, text):
            amount_fragments.append({"bbox": bbox, "text": text})
        else:
            other_fragments.append({"bbox": bbox, "text": text})

    contextual_amounts = []
    for amount in amount_fragments:
        ay_center = (amount["bbox"][0][1] + amount["bbox"][2][1]) / 2
        ax_left = amount["bbox"][0][0]
        best_candidate = None
        min_distance = float('inf')

        for other in other_fragments:
            oy_center = (other["bbox"][0][1] + other["bbox"][2][1]) / 2
            ox_right = other["bbox"][1][0]
            if abs(ay_center - oy_center) < 20 and ox_right < ax_left:
                distance = ax_left - ox_right
                if distance < min_distance:
                    min_distance = distance
                    best_candidate = other["text"]
        
        cleaned_amount = clean_monetary_value(amount["text"])
        contextual_amounts.append({
            "amount": cleaned_amount,
            "context": best_candidate or "Unknown"
        })
    print(contextual_amounts)
    return contextual_amounts

def extract_contextual_amounts_from_text(text_input: str):
    """Processes a TEXT STRING to find amounts and their sequential context."""
    contextual_amounts = []
    money_pattern = re.compile(
        r"((?:(?:\$|S|€|£|₹|INR|USD|EUR|GBP)\s?)?[\d,]+(?:\.\d{1,2})?%?)"
    )

    # Use re.finditer to get the position of each match
    for match in re.finditer(money_pattern, text_input):
        amount_text = match.group(0)
        
        # Look at the text immediately before the match
        context_window_start = max(0, match.start() - 30) # Look back 30 chars
        context_window = text_input[context_window_start : match.start()]
        
        # The last couple of words in the window are the best candidates for the label
        words_in_window = context_window.strip().split()
        best_candidate = " ".join(words_in_window[-2:]) if words_in_window else "Unknown"

        cleaned_amount = clean_monetary_value(amount_text)
        contextual_amounts.append({
            "amount": cleaned_amount,
            "context": best_candidate
        })
        
    return contextual_amounts

def label_amounts_with_llm(contextual_data: list):
    """
    Uses an LLM to assign clean labels and format the final JSON output.
    """
    if not contextual_data:
        return {"status": "no_amounts_found", "reason": "OCR found no amounts to process."}
        
    context_string = "\n".join([f"- Amount: {item['amount']}, Nearby Text: {item['context']}" for item in contextual_data])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly intelligent data extraction bot. Your task is to analyze the user's text, which contains monetary amounts and their nearby context, and transform it into a specific JSON format.

        Follow these rules precisely:
        1.  **Top-Level Currency:** Determine the single currency for the entire document (e.g., USD, INR, EUR) and place it in the top-level "currency" key. if the identified currency is S, treat it as USD.
        2.  **Amounts List:** Process each line item to create a list of objects for the "amounts" key.
        3.  **Object Structure:** Each object in the "amounts" list MUST have three keys: "type", "value", and "source".
        4.  **"type" Key:** The "type" must be a clean, concise, snake_case label based on the 'Nearby Text'. (e.g., 'SUB TOTAL' becomes 'sub_total', 'Amount DUE' becomes 'amount_due').
        5.  **"value" Key:** The "value" MUST be a JSON number (integer or float), not a string. Extract this from the amount text.
        6.  **"source" Key:** The "source" must be a string reconstructing the original context in the format: "text: 'Context': 'Amount'".
        7.  **Status:** The final JSON must include a top-level "status" key with the value "ok".

        **EXAMPLE:**
        Human Input:
        - Amount: ₹1200, Nearby Text: Total
        - Amount: ₹1000, Nearby Text: Paid
        - Amount: ₹200, Nearby Text: Due

        Your JSON Output: 
       {{
    "currency": "INR",
    "amounts": [
        {{"type": "total", "value": 1200, "source": "text: 'Total: ₹1200'"}},
        {{"type": "paid", "value": 1000, "source": "text: 'Paid: ₹1000'"}},
        {{"type": "due", "value": 200, "source": "text: 'Due: ₹200'"}}
    ],
    "status": "ok"
    }}
         
        """),
        ("human", "{context_string}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response_str = chain.invoke({"context_string": context_string})
    
    try:
        start_index = response_str.find('{')
        end_index = response_str.rfind('}')
        if start_index != -1 and end_index != -1:
            clean_json_str = response_str[start_index : end_index + 1]
            return json.loads(clean_json_str)
        else:
            raise json.JSONDecodeError("No JSON object found in the response.", response_str, 0)
    except json.JSONDecodeError:
        return {"status": "error", "reason": "LLM returned invalid JSON.", "raw_output": response_str}
    


















'''
# --- Part 3: FastAPI App and Local Testing Block ---
app = FastAPI(title="Intelligent OCR Service")
class TextRequest(BaseModel):
    text: str

@app.post("/process-image")
async def process_image_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return extract_contextual_amounts(image_bytes)

@app.post("/process-text")
async def process_text_endpoint(request: TextRequest):
    return extract_contextual_amounts_from_text(request.text)


if __name__ == '__main__':
    print("\n--- Running IMAGE Test Mode ---")
    image_path = 'r2.png'
    output_filename = 'final_labeled_output.json'
    text_output_filename = 'final_labeled_output_text.json'
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        contextual_data = extract_contextual_amounts(image_bytes)
        final_results = label_amounts_with_llm(contextual_data)
        with open(output_filename, 'w') as f:
            json.dump(final_results, f, indent=4)
       
    except FileNotFoundError:
        print(f"ERROR: Test file '{image_path}' not found.")

    print("\n--- Running TEXT Test Mode ---")
    sample_text = "The total for the Full Check Up was S745.00. The final invoice TOTAL is S1,902.05."
    text_results = extract_contextual_amounts_from_text(sample_text)
    text_final_results = label_amounts_with_llm(text_results)
    with open(text_output_filename, 'w') as f:
        json.dump(text_final_results, f, indent=4)
'''