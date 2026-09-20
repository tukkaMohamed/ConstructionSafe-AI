# 🦺 ConstructionSafe AI

A multimodal construction safety assistant that combines:

- 📚 Retrieval-Augmented Generation (RAG)
- 👁️ YOLO object detection
- 🤖 Ollama local LLM
- ⚡ FastAPI backend
- 🎨 Streamlit frontend

ConstructionSafe AI answers construction safety questions using a PPE safety document and can optionally analyze uploaded construction images using a trained YOLO model.

---

## 📌 Project Overview

The system can:

1. Retrieve relevant safety information from the construction PPE document.
2. Generate grounded answers using a local LLM.
3. Detect construction safety-related objects in uploaded images.
4. Combine document context with visual observations.
5. Display answers, sources, and YOLO detections through a Streamlit interface.

### Important Design Principle

> YOLO detections are treated as **visual observations only**.
> They are not automatically interpreted as proof of safety compliance or non-compliance.

---

## 🏗️ Architecture

The system follows a multimodal architecture that combines document retrieval, computer vision, and local language generation.

```text
                    ┌──────────────────────┐
                    │   Streamlit Frontend │
                    │      Port 8501       │
                    └──────────┬───────────┘
                               │ HTTP
                               ▼
                    ┌──────────────────────┐
                    │    FastAPI Backend   │
                    │      Port 8000       │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
              ▼                ▼                 ▼
       ┌────────────┐   ┌─────────────┐   ┌────────────┐
       │   Chroma   │   │    YOLO     │   │   Ollama   │
       │ Vector DB  │   │  Detection  │   │  Llama 3.2 │
       └─────┬──────┘   └──────┬──────┘   └──────┬─────┘
             │                 │                  │
             ▼                 ▼                  │
      Retrieved chunks   Visual observations       │
             │                 │                  │
             └────────────────┬┴──────────────────┘
                               ▼
                         Grounded Answer
```

---

## 📁 Project Structure

```text
Construction-Hazard-Detection.v97i.yolov11/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── routes/
│   │   │       └── query.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── schemas/
│   │   │   └── query.py
│   │   ├── services/
│   │   │   ├── retrieval.py
│   │   │   ├── generation.py
│   │   │   └── yolo.py
│   │   └── utils/
│   │
│   ├── data/
│   │   └── vector_store/
│   │
│   ├── tests/
│   │   └── test_query.py
│   │
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── .env
│   └── requirements.txt
│
├── notebooks/
│   └── rag_pipeline.ipynb
│
├── runs/
│   └── detect/
│       └── runs/
│           └── construction_yolo11n/
│               └── weights/
│                   └── best.pt
│
├── ingest_documents.py
├── chunk_documents.py
├── embedding_store.py
├── evaluate_model.py
├── predict_image.py
├── split_dataset.py
├── data.yaml
├── .gitignore
└── README.md
```

---

## 🧠 RAG Pipeline

The Retrieval-Augmented Generation pipeline follows these main stages:

### 1. Document Loading

The construction PPE PDF is loaded and inspected before processing. The document contains safety information covering:

- Eye and face protection
- Foot protection
- Hand protection
- Head protection
- Hearing protection

### 2. Chunking Strategy

The document is split into overlapping text chunks.

**Current configuration:**

```text
Chunk size: 1000 words
Overlap: 150 words
```

Each chunk is stored with metadata such as:

- Source
- Page
- Chunk ID

### 3. Embeddings

The system uses the following Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

The model converts document chunks and user questions into vector embeddings.

### 4. Vector Store

ChromaDB is used as the persistent vector database.

**Collection:** `construction_safety`

### 5. Retrieval

For each user question, the system retrieves the most relevant document chunks.

**Current configuration:** `Top K = 2`

### 6. Generation

The retrieved context is passed to the local Ollama LLM:

**Model:** `llama3.2`

The generated answer is grounded in the retrieved safety information and includes the relevant document source.

---

## 👁️ Computer Vision Component

The Extended Track includes a YOLO-based computer vision component for detecting construction safety-related objects in uploaded images.

### Model

The system uses a trained **YOLO11n** model. The model detects the following classes:

- Hardhat
- Mask
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest
- machinery
- utility pole
- vehicle

### Detection Configuration

```text
Model: YOLO11n
Confidence threshold: 0.30
```

### Detection Output

For each detected object, the system returns:

- Class name
- Confidence score
- Bounding box coordinates

### Multimodal Integration

The YOLO detections are converted into computer vision observations and provided to the generation component together with the retrieved document context.

```text
Uploaded Image
      ↓
YOLO11n
      ↓
Object Detections
      ↓
Computer Vision Context
      ↓
RAG + LLM Prompt
      ↓
Multimodal Answer
```

### Safety Interpretation

YOLO detections are treated as **visual observations only**. A detection such as `NO-Hardhat` is reported as a detection and is not automatically interpreted as proof that a worker is unsafe, non-compliant, or violating a safety requirement.

The safety document defines the safety requirements, while YOLO describes visual observations from the uploaded image.

---

## 📊 Evaluation

### YOLO Model Evaluation

The YOLO model was evaluated on a held-out test set containing **670 images**.

**Overall test metrics:**

| Metric | Score |
|---|---|
| Precision | 0.778 |
| Recall | 0.535 |
| mAP@50 | 0.602 |
| mAP@50–95 | 0.358 |

The model was trained for **30 epochs**.

### RAG and Multimodal Evaluation

The multimodal assistant was evaluated using 10 test questions covering:

- Document-based construction safety questions
- Image-based questions
- Multimodal questions combining document requirements with image observations
- Questions where the provided document does not contain enough information

**Evaluation results:**

| Metric | Result |
|---|---|
| Total tests | 10 |
| Passed tests | 8 |
| Partial tests | 2 |
| Pass rate | 80% |

### Failure Cases and Mitigation

Several failure cases were identified during evaluation.

**1. Missing Image Observations**
In an early test, the model answered the document-based question correctly but did not mention the available YOLO detections.
*Mitigation:* The generation prompt was updated to require relevant computer vision observations to be explicitly mentioned when detections are available.

**2. Inferring Compliance from YOLO**
The model initially interpreted `NO-Hardhat` detections as evidence of non-compliance.
*Mitigation:* The generation prompt was strengthened to explicitly separate visual observations from safety requirements and prohibit compliance judgments based only on YOLO detections.

**3. Irrelevant Retrieved Information**
For a question about respiratory protection, the document did not provide enough information. The model avoided inventing a respiratory requirement but added unrelated hearing-protection information.
*Mitigation:* The prompt was updated to focus the answer only on information relevant to the user's question.

**4. Image-Focused Questions**
For a question specifically asking what YOLO detects, the model initially referenced the document unnecessarily.
*Mitigation:* The multimodal prompting was adjusted so image-focused questions prioritize computer vision observations.

---

## ⚙️ Backend API

The backend is implemented using FastAPI.

### Health Check

**`GET /health`**

Checks whether the backend service is running.

Example response:

```json
{
  "status": "healthy",
  "service": "ConstructionSafe AI API"
}
```

### Query Endpoint

**`POST /query`**

Accepts a construction safety question and an optional image.

**Request** — the endpoint uses `multipart/form-data`:

| Field | Required | Description |
|---|---|---|
| `question` | ✅ | Text question |
| `image` | optional | Construction image |

**Response:**

```json
{
  "answer": "Workers should wear hard hats where there is a potential for falling objects.",
  "sources": [
    "CONSTRUCTION_PPE.pdf - Page 1"
  ],
  "detections": [
    {
      "class_name": "Hardhat",
      "confidence": 0.85,
      "bbox": [10.0, 20.0, 100.0, 150.0]
    }
  ]
}
```

---

## 🚀 Running the Backend

From the project root:

```bash
uvicorn backend.app.main:app --reload
```

The backend will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 🎨 Running the Frontend

Install the frontend dependencies:

```bash
pip install -r frontend/requirements.txt
```

Start the Streamlit application:

```bash
streamlit run frontend/app.py
```

The frontend will be available at:

```text
http://localhost:8501
```

---

## 🧪 Running Tests

From the project root:

```bash
python -m pytest backend/tests/test_query.py -v
```

Expected result:

```text
2 passed
```

---

## 🛠️ Technologies

**Backend**
- Python
- FastAPI
- Pydantic
- Uvicorn
- Ollama

**Retrieval & RAG**
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- Retrieval-Augmented Generation (RAG)

**Computer Vision**
- YOLO11n
- Ultralytics
- PyTorch
- Torchvision

**Frontend**
- Streamlit
- Requests

**Development & Testing**
- Jupyter Notebook
- Pytest
- Git & GitHub

---

## 📦 Dataset Attribution

The computer vision dataset used in this project is **Construction-Hazard-Detection**.

**Source:** The dataset was obtained from Roboflow Universe and used in accordance with its CC BY 4.0 license.

The dataset is provided under the **CC BY 4.0** license.

The dataset was split into:

```text
Train:      5,350 images
Validation:   668 images
Test:         670 images
```

The safety knowledge base was created from `CONSTRUCTION_PPE.pdf`.

---

## ⚠️ Limitations

- The YOLO model provides visual detections and should not be treated as proof of safety compliance.
- Detection performance can vary depending on image quality, lighting, camera angle, and object visibility.
- The RAG knowledge base currently contains a limited construction PPE document.
- The assistant can only provide document-grounded answers based on the available knowledge base.
- Ollama is used locally for LLM generation.
- The current system is intended as an AI-assisted safety information tool and does not replace professional safety inspection or official safety procedures.

---

## 🔮 Future Improvements

- Expanding the construction safety document collection.
- Adding OCR for scanned safety documents.
- Improving retrieval with hybrid search and reranking.
- Fine-tuning the YOLO model with additional construction-site data.
- Adding more safety-related computer vision classes.
- Adding user authentication and conversation history.
- Deploying the backend and frontend to a cloud environment.
- Adding more comprehensive automated RAG evaluation.