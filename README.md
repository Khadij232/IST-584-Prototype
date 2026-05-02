[README.md](https://github.com/user-attachments/files/27295521/README.md)
# IST-584-Prototype
AI-powered privacy policy analyzer that detects privacy risks and recommends tailored settings using Legal-BERT and EasyOCR.
# 🔐 Privacy Policy Analyzer

An AI-powered web application that detects privacy risks in app privacy policies and provides tailored setting recommendations based on a user's device screenshots — no technical knowledge required.

Built as a capstone project for IST 584, this tool addresses the **privacy paradox**: users care about their data but rarely act on it due to the complexity and length of privacy policies. This system reduces that burden by doing the reading for them.

---

## 📋 Features

- **Privacy Policy Classification** — paste any privacy policy and the app identifies risks across 5 categories using a fine-tuned Legal-BERT model
- **Privacy Risk Score** — an overall 0–100 risk score with color-coded feedback (green / orange / red)
- **Detected Data Practices** — per-label confidence scores with detailed info popovers
- **Riskiest Sentences** — sentence-level breakdown grouped by label in collapsible dropdowns, color-coded by category
- **Screenshot Analysis** — upload a settings screenshot and the app uses EasyOCR to extract text and identify enabled privacy-risk settings
- **Priority Alerts** — cross-references policy findings with screenshot settings to flag the most urgent risks

---

## AI & ML Stack

| Layer | Technology | Purpose |
|---|---|---|
| OCR | EasyOCR (deep learning CNN + LSTM) | Extracts text from settings screenshots |
| Classification | Legal-BERT (fine-tuned transformer) | Multi-label privacy risk classification |
| Inference | PyTorch + HuggingFace Transformers | Model loading and inference |
| UI | Streamlit | Web interface |

---

## Label Taxonomy

The model classifies text across five privacy risk categories:

| Label | Description |
|---|---|
| `DATA_COLLECTION` | Collects personal data such as name, email, device info, and usage behavior |
| `THIRD_PARTY_SHARING` | Shares user data with external companies like advertisers or analytics providers |
| `LOCATION_DATA` | Tracks physical location using GPS or network-based methods |
| `ADVERTISING_DATA` | Uses user behavior to deliver targeted or personalized advertisements |
| `OPT_OUT` | Privacy settings are enabled by default, requiring users to manually opt out |

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/privacy-policy-analyzer.git
cd privacy-policy-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the model files
The fine-tuned Legal-BERT model is not included in this repository due to file size. Place your model files in a folder called `model_folder` in the root of the project:

```
privacy-policy-analyzer/
├── app.py
├── requirements.txt
├── README.md
└── model_folder/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── vocab.txt
    └── ...
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Project Structure

```
privacy-policy-analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── model_folder/           # Fine-tuned Legal-BERT model (not included)
```

---

## Model Training

The Legal-BERT model (`nlpaueb/legal-bert-base-uncased`) was fine-tuned as a multi-label classifier using a custom-labeled dataset of privacy policy sentences. Training details:

- **Base model:** `nlpaueb/legal-bert-base-uncased`
- **Task:** Multi-label sequence classification
- **Loss function:** BCEWithLogitsLoss
- **Epochs:** 8
- **Learning rate:** 2e-5
- **Batch size:** 8
- **Threshold:** 0.3 (for positive label prediction)
- **Metrics:** Micro F1, Precision, Recall

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

```
streamlit
torch
transformers
easyocr
Pillow
numpy
```

---

## Limitations

- Screenshot OCR accuracy depends on image quality — low-contrast or partially visible text may produce unreliable results
- Recommendations are label-based and should be treated as a starting point for privacy awareness, not definitive legal advice

---

## Author

**Khadijah Akinsanmi** — IST 584 Capstone Project  
The Pennsylvania State University
