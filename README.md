# InnerAlign Wellness AI

A comprehensive wellness tracking and emotional intelligence system that helps users monitor their emotional baseline, detect mood drifts, and receive personalized wellness advice using AI-powered RAG (Retrieval-Augmented Generation).

## 🌟 Features

- **Emotional Baseline Calculation**: Analyzes historical wellness data to establish personalized emotional baselines
- **Mood Drift Detection**: Identifies deviations from your normal emotional patterns
- **AI-Powered Wellness Advice**: Provides personalized recommendations using RAG technology
- **Knowledge Base Management**: Maintains structured wellness information and policies

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mihir-techie/upskalor_kt.git
cd upskalor_kt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env

### Usage

#### Calculate Emotional Baseline
```bash
cd day2
python baseline_engine.py
```

#### Detect Mood Drifts
```bash
cd day3
python drift_detector.py
```

#### Run RAG Wellness System
```bash
cd day6
python rag.py
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

## 🔧 Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning preprocessing
- **LangChain**: RAG framework
- **FAISS**: Vector similarity search
- **HuggingFace Transformers**: Embeddings generation

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


