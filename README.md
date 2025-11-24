# nava2105/ValkeySearchDemo

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Valkey](https://img.shields.io/badge/valkey-%23DC382D.svg?style=for-the-badge&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Bootstrap](https://img.shields.io/badge/bootstrap-%23563D7C.svg?style=for-the-badge&logo=bootstrap&logoColor=white)

## Table of Contents
1. [General Info](#general-info)
2. [Features](#features)
3. [Technologies](#technologies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Endpoints](#endpoints)
7. [Configuration and Notes](#configuration-and-notes)
8. [Database Schema](#database-schema)

---

## General Info
This project is a **Multilingual Semantic Dictionary** with vector search capabilities and an interactive word connection game. It uses multilingual embeddings to find conceptually similar words across different languages, enabling semantic search beyond traditional translation.

- **Key Features**:
  - Semantic search for words across 7 languages
  - Vector-based similarity using sentence transformers
  - Interactive "Connect Words" game with semantic challenges
  - Valkey (Redis OSS compatible) as vector database
  - Web interface with Bootstrap 5

Designed for language learning and exploring semantic relationships across languages.

---

## Features
The project includes the following capabilities:
- **Semantic Dictionary**:
  - Search for conceptually similar words in any language
  - Filter results by specific language or search across all
  - Adjustable number of results (1-20)
  - Real-time similarity scoring

- **Word Connection Game**:
  - Interactive gameplay: find the most conceptually identical word
  - 4 options per round with varied languages
  - Immediate feedback with correct answers
  - Unlimited rounds with random word selection

- **Vector Visualizer**:
  - Interactive 2D visualization of semantic vector space
  - PCA dimensionality reduction for real-time rendering
  - Color-coded by language
  - Hover to see word details
  - Filter by language and adjust sample size
  - Performance-optimized with batch loading

- **Multilingual Support**:
  - 7 languages: Spanish, English, French, Russian, Italian, German, Portuguese
  - 15,000 most frequent words per language
  - Vector embeddings generated via multilingual models

- **Vector Database**:
  - Valkey with Search module for HNSW vector indexing
  - Efficient cosine similarity search
  - Persistent storage with Docker volumes

---

## Technologies
Technologies used in this project:
- [Python](https://www.python.org/): Version 3.8+
- [Flask](https://flask.palletsprojects.com/): Web framework
- [Valkey](https://valkey.io/): Vector database with Search module
- [SentenceTransformers](https://www.sbert.net/): Multilingual embeddings
- [scikit-learn](https://scikit-learn.org/): PCA for dimensionality reduction
- [Docker](https://www.docker.com/): Containerization
- [Bootstrap](https://getbootstrap.com/): Frontend styling
- [Plotly.js](https://plotly.com/javascript/): Interactive visualizations
- [wordfreq](https://pypi.org/project/wordfreq/): Word frequency lists

---

## Installation
### Prerequisites
- **Python 3.8+**: Ensure Python is installed with pip
- **Docker**: Required for running Valkey with Search module
- **Git**: For cloning the repository

### Steps to Run the Application:
1. Clone this repository:
   ```bash
   git https://github.com/nava2105/ValkeySearchDemo.git
   cd ValkeySearchDemo
   ```

2. Start Valkey with Search module using Docker:
    ```bash
    docker run --name valkey-search-container -d -p 6379:6379 -v valkey-data:/data valkey/valkey-bundle:8.1.0 valkey-server --save 60 1 --loglevel warning
    ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Create requirements.txt with: flask, valkey, sentence-transformers, wordfreq, numpy)

4. Load the word embeddings into Valkey (one-time setup):
   ```bash
   python main.py
   ```
   This process takes ~5-10 minutes and loads 105,000 words (15k per language).

5. Run the Flask application: 
   ```bash
   python app.py
   ```
   
6. Access the application:
* Web Interface: http://localhost:5000
* Dictionary tab: Search semantically similar words
* Game tab: Play "Connect Words"

---

## Usage
Once the application is running, you can interact with it through the web interface:

### Using the Dictionary:
1. Enter a word in any supported language (e.g., "gatto", "love", "liberté")
2. Optionally select a source language filter 
3. Choose number of results (default: 5)
4. Click "Buscar" to see conceptually similar words across languages

#### Example Dictionary Search:
* Input: "gatto" (Italian)
* Results: "cat" (en), "gato" (es), "chat" (fr), "Katze" (de) - all semantically similar

### Playing the Game:
1. The game displays a target word in its original language
2. Four options appear in different languages
3. Select the word most conceptually identical or similar
4. Get immediate feedback with the correct answer
5. Click "Nueva Ronda" for a new challenge

#### Example Game Round:
* Target: "libertad" (Spanish)
* Options: "freedom" (en), "liberté" (fr), "book" (en), "freiheit" (de)
* Correct: "freedom" or "liberté" (conceptually closest)

### Using the Vector Visualizer:
1. Navigate to the "Visualizador de Vectores" tab
2. Select language filter (optional) to focus on specific languages
3. Choose sample size (100-10000 words) based on performance
4. Click "Actualizar" to generate the 2D visualization
5. Interact with the plot: hover over points to see words, toggle languages, zoom/pan

#### Visualization Features:
- Points are color-coded by language
- PCA reduces 384 dimensions to 2D for visualization
- Explained variance ratio displayed for quality assessment
- Optimized batch loading for fast rendering

---

## Endpoints
Below is a comprehensive list of the endpoints included in the project:

### Public Endpoints (No Authentication Required)
- **Dictionary Search**
  - `GET /` - Renders the dictionary interface
  - `POST /` - Processes search form and returns semantic results

- **Game Interface**
  - `GET /game` - Renders the word connection game
  - `POST /game` - Processes game answers and new round requests

- **Vector Visualizer**
  - `GET /visualize` - Renders the 2D vector visualization interface
  - `POST /visualize` - Processes visualization parameters and returns Plotly data

### Data Loading Endpoints (Development Only)
- **Initialize Database**
  - Command: `python main.py`
  - Description: Cleans existing Valkey data, generates embeddings for 105,000 words, creates HNSW vector index, verifies storage and search functionality

---

## Configuration and Notes
- **Model Configuration:**
    - Uses paraphrase-multilingual-MiniLM-L12-v2 (384-dimensional vectors)
    - Can be changed in app.py and main.py for different models
    - Model size: ~90MB download on first run
- **Valkey Index Configuration:**
    - HNSW algorithm with cosine distance
    - Prefix: word:vector:*
    - Vector dimension: 384 (FLOAT32)
    - Optimized for hybrid vector + text search
- **Visualization Configuration:**
    - PCA with 2 components for real-time 2D visualization
    - Random sampling for performance (configurable: 100-10000 words)
    - Language-based color coding with 7 predefined colors
    - Plotly.js for interactive rendering
- **Performance:**
    - First load: ~5-10 minutes for 105k words
    - Search latency: <100ms per query
    - Game generation: <500ms per round 
    - Visualization: <2s for 200 words, <5s for 1000 words 
    - Memory usage: ~150MB for full dataset
- **Word Corpus:**
    - 15,000 most frequent words per language from wordfreq
    - Total database size: ~150MB with vectors 
    - Covers 7 languages: es, en, fr, ru, it, de, pt

---

## Database Schema
The project uses Valkey HASH keys with the following structure:

**Key Pattern:**

`word:vector:{lang_code}:{word_hash}`

**Fields**

| **Field** | **Type**        | **Description**           |
|-----------|-----------------|---------------------------|
| world     | String (UFT-8)  | The actual word           |
| language  | String (UFT-8)  | ISO 639-1 language code   |
| vector    | BYTES (FLOAT32) | 384-dimensional embedding |

**Index**
- **Name:** `words_idx`
- **Type:** HASH with HNSW vector index
- **Fields Indexed:** `vector` (for similarity search)

---
