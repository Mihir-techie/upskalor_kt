# InnerAlign Wellness AI

A comprehensive wellness tracking and emotional intelligence system that helps users monitor their emotional baseline, detect mood drifts, and receive personalized wellness advice using AI-powered RAG (Retrieval-Augmented Generation).

## 🌟 Features

- **Emotional Baseline Calculation**: Analyzes historical wellness data to establish personalized emotional baselines
- **Mood Drift Detection**: Identifies deviations from your normal emotional patterns
- **AI-Powered Wellness Advice**: Provides personalized recommendations using RAG technology
- **Knowledge Base Management**: Maintains structured wellness information and policies
- **REST API**: FastAPI-based web service for integration

## 📁 Project Structure

```
├── day2/
│   ├── baseline_engine.py      # Emotional baseline calculation engine
│   └── fitlife_emotional_dataset.csv  # Sample wellness dataset
├── day3/
│   └── drift_detector.py      # Mood drift detection algorithms
├── day6/
│   ├── rag.py                 # RAG system for wellness advice
│   ├── rag_documentation.md   # RAG system documentation
│   ├── knowledge_base/        # Wellness knowledge base
│   └── document_index.faiss/  # Vector index for semantic search
├── day7/
│   ├── app.py                 # FastAPI application for the wellness assistant
│   └── faiss_index/           # Copied vector index for the API
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mihir-techie/upskalor_kt.git
cd upskalor_kt
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional for OpenAI):
```bash
cp .env.example .env

### Running All Components

To run the entire system successfully:

1. **Calculate Baseline**:
```bash
cd day2
python baseline_engine.py
```

2. **Run Drift Detection**:
```bash
cd day3
copy ..\day2\user_baseline_stats.csv .
python drift_detector.py
```

3. **Build Knowledge Base**:
```bash
cd ..\day6
python rag.py
```

4. **Start the API Server**:
```bash
cd ..\day7
# Default port
python -m uvicorn app:app --host 0.0.0.0 --port 8000
# If 8000 is in use, switch to 8001
# python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

The API will be available at `http://localhost:8000`

### API Endpoints

- Upload knowledge (PDF or text):
```bash
curl -X POST -F "file=@path\to\document.pdf" http://localhost:8000/admin/knowledge/upload
```
- Chat with the assistant:
```bash
curl -X POST "http://localhost:8000/chat?question=How%20can%20I%20reduce%20stress&drift=Neutral&guidance=General%20wellness"
```

## 📊 Data Format

The system expects wellness data in CSV format with the following columns:
- ID: User identifier
- Date: Timestamp of the entry
- Primary Emotion: Main emotional state
- Energy Level (1-10): Self-reported energy
- Mood After (1-10): Post-activity mood
- Stress Level (1-10): Current stress level
- Additional demographic and activity columns

## 🤖 AI Components

- **Embeddings**: Uses sentence-transformers for semantic understanding
- **Vector Store**: FAISS for efficient similarity search
- **Document Processing**: LangChain for text splitting and management
- **LLM**: OpenAI GPT-3.5-turbo for response generation

## 🔧 Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning preprocessing
- **LangChain**: RAG framework
- **FAISS**: Vector similarity search
- **HuggingFace Transformers**: Embeddings generation
- **FastAPI**: Web framework for the API
- **Uvicorn**: ASGI server

## 📈 Key Metrics

The system tracks several wellness metrics:
- **Energy Level**: Physical and mental energy
- **Mind Clarity**: Cognitive focus and clarity
- **Stress Level**: Current stress intensity
- **Emotion Score**: Normalized emotional state


## Acknowledgments

- Built as part of the Upskolor KT program
- Uses open-source AI and wellness technologies
- Community-driven wellness insights


