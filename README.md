# 📝 Text Summarizer & (FastAPI + T5)

A simple and elegant **Text Summarization Web App** built using
**FastAPI** and **Hugging Face Transformers**.\
This project uses a fine-tuned **T5-small** model to generate summaries
from dialogue-based text.

------------------------------------------------------------------------

## 🚀 Features

-   ✨ Dialogue-based text summarization\
-   ⚡ FastAPI backend for fast processing\
-   🎨 Clean and responsive UI\
-   🤖 Transformer-based NLP model\

------------------------------------------------------------------------

## 🧠 Model Details

-   **Model:** T5-small\
-   **Fine-tuned on:** SAMSum Dataset (Dialogue Dataset)\
-   **Training Samples:** 5000 dialogues\
-   **Task:** Abstractive Text Summarization

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   **Backend:** FastAPI\
-   **Frontend:** HTML, CSS, JavaScript\
-   **ML Framework:** PyTorch\
-   **Transformers:** Hugging Face Transformers

------------------------------------------------------------------------

## 📂 Project Structure

    ├── templates/
    │   └── index.html
    ├── saved_summary_model/
    │   └── (fine-tuned T5 model files)
    ├── app.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Clone the Repository

``` bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

------------------------------------------------------------------------

### 2. Create Virtual Environment

``` bash
python -m venv env
```

Activate environment:

-   **Windows**

``` bash
env\Scripts\activate
```

-   **Linux / Mac**

``` bash
source env/bin/activate
```

------------------------------------------------------------------------

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### 4. Run the Application

``` bash
uvicorn app:app --reload
```

------------------------------------------------------------------------

### 5. Open in Browser

    http://127.0.0.1:8000

------------------------------------------------------------------------

## 🔌 API Endpoint

### POST `/summarize/`

**Request Body:**

``` json
{
  "dialogue": "Your input text here"
}
```

**Response:**

``` json
{
  "summary": "Generated summary..."
}
```

------------------------------------------------------------------------

## ⚠️ Important Notes

-   The model is trained on only **5000 samples** from the SAMSum
    dataset.\
-   This is a **lightweight and experimental setup**.

### For better accuracy:

-   Train on the **full dataset**\
-   Increase the **number of epochs**\
-   Use larger models like **t5-base** or **t5-large**

------------------------------------------------------------------------

## 📌 Future Improvements

-   Integrate translation feature in UI\
-   Improve model accuracy with more data\
-   Optimize performance for deployment\
-   Deploy on cloud (AWS / GCP / Render)\
-   Improve mobile responsiveness

------------------------------------------------------------------------


