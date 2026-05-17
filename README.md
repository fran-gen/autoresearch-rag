
<div align="center">
  <img src="./static/logo.png" alt="Monotonic Labs Logo" width="100%">
</div>

# AutoRAG Research Lab

An automated research and experimentation platform for enterprise Retrieval-Augmented Generation (RAG) systems. Built as a showcase project for a hackathon, this platform enables developers and researchers to systematically benchmark, evaluate, and optimize RAG pipelines.

## 🚀 Features Showcase

Designed with experimentation in mind, the AutoRAG Research Lab provides a comprehensive suite of tools to take your RAG systems from proof-of-concept to production-ready:

- **🤖 Autonomous AI Agents**: Orchestrates specialized agents (Planner, Worker, Researcher, Evaluator) to automatically formulate hypotheses, build retrieval pipelines, and evaluate their performance.
- **📊 Comprehensive Benchmarking**: Built-in support for benchmark datasets (like the Karpathy Sandbox) to rigorously test retrieval accuracy, context relevance, and generation quality.
- **🔍 Pluggable Retrieval Pipelines**: Easily configure and swap out different retrieval strategies, including Dense (vector-based), Sparse (BM25), and Hybrid retrieval methods.
- **📈 Interactive Dashboard**: A sleek, real-time web interface to track ongoing experiments, compare leaderboard results, visualize metrics, and manage research hypotheses.
- **🧠 Advanced RAG Techniques**: Native support for state-of-the-art techniques like reranking and custom embedding models to squeeze the maximum performance out of your knowledge base.

## 🛠 Tech Stack Deep Dive

The project is built on a modern, scalable, and modular architecture, separating the heavy AI workloads from the user-facing interfaces.

### Core AI & Backend

- **Python (>=3.11)**: The backbone of the entire application.
- **LangChain & LangGraph**: Used for complex agent orchestration, state management, and seamless integrations with Google GenAI and other LLM providers.
- **FastAPI & Uvicorn**: Provides a blazing-fast, asynchronous REST API to handle agent requests and system communications.
- **Qdrant**: High-performance vector database used for storing and querying dense embeddings.
- **Sentence Transformers**: Generates state-of-the-art text embeddings for semantic search.
- **rank-bm25**: Powers the sparse retrieval components for keyword-based search.
- **SQLite (aiosqlite)**: Lightweight, asynchronous database for storing experiment results, hypotheses, and system state.
- **Docker**: Ensures consistent and reproducible environments across different setups.

### Frontend

- **Flask & Werkzeug**: Serves the interactive web dashboard and manages session state.
- **Jinja2**: Powers the dynamic HTML templates (`src/templates`).
- **Bulma CSS**: A modern CSS framework used for clean, responsive, and maintainable styling without writing complex custom CSS.
- **Chart.js**: Renders interactive, real-time charts and visual metrics for experiment leaderboards.

## Contributors

<div align="center">
  <a href="https://github.com/fran-gen">
    <img src="https://github.com/fran-gen.png?size=80" width="80" alt="fran-gen" />
  </a>
  <a href="https://github.com/Bonhollow">
    <img src="https://github.com/Bonhollow.png?size=80" width="80" alt="Bonhollow" />
  </a>
  <a href="https://github.com/DeanHnter">
    <img src="https://github.com/DeanHnter.png?size=80" width="80" alt="DeanHnter" />
  </a>
</div>

