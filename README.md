<p align="center">
  <img src="docs/logo.png" alt="FINALOGY" width="196" />
</p>

# FINALOGY: A Vision-Language System for Candlestick Pattern Analogy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> FINALOGY shifts financial analysis from **rigid pattern classification** to **data-driven visual analogy**. Upload a candlestick chart or provide OHLC data, retrieve similar historical windows via learned embeddings, and generate LLM-based analysis reports.

---

## 📜 Introduction

Traditional financial analysis relies on a fixed taxonomy of candlestick patterns (e.g., "Hammer", "Engulfing"). Yet these predefined rules fail to cover the vast diversity of real-world K-line sequences. Analysts need to ask: *"When did the market show a similar shape before? What happened next?"*—but rigid classification cannot answer this.

**FINALOGY** addresses this gap with a vision-language pipeline:

1. **Visual Encoder**: A specialized model (VICReg) pre-trained on large-scale candlestick images for visual analogy.
2. **Retrieval Module**: Identifies historically analogous market segments for any query image or OHLC sequence.
3. **RAG Pipeline**: An LLM synthesizes retrieved evidence into professional, interpretable reports.

| Input | Output |
|-------|--------|
| Candlestick chart image **or** OHLC data | Similar historical windows + LLM analysis report |

---

## ✨ Features

- **Dual input**: Upload a K-line chart image, or paste OHLC (Open, High, Low, Close) data.
- **Similarity retrieval**: Embedding-based retrieval finds analogous historical patterns.
- **Report generation**: LLM produces structured analysis with references to similar cases.
- **Interactive UI**: Next.js frontend with chat-style interface and cloud sync (Supabase).

---

## 🚀 Getting Started

### Requirements

- **Node.js** 18+
- **pnpm** or **npm**
- **Python** 3.11+ (backend API)
- **Conda** (recommended) or pip

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/nice-zzy/FinAlogy.git
cd FinAlogy
```

**2. Install Node dependencies**

```bash
pnpm install
# or
npm install
```

**3. Set up the Python environment**

**Option A: Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate kline-env
```

**Option B: pip**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables (optional)**

Copy the example env file and fill in your settings:

```bash
cp apps/web/.env.example apps/web/.env.local
```

Edit `apps/web/.env.local` and provide:

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_SUPABASE_URL` | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anon key |
| `NEXT_PUBLIC_API_URL` | Backend API base URL (default: `http://localhost:8000`) |

> If you don't need sign-in features, you can leave these empty; some features may be limited.

**5. Run the project**

```bash
pnpm dev
# or
npm run dev
```

- **Frontend**: http://localhost:3000  
- **Backend API**: http://localhost:8000  

---

## 📦 Project Structure

```
FinAlogy/
├── apps/web           # Next.js frontend
├── services/api       # FastAPI backend (inference, retrieval)
├── services/training  # VICReg training and evaluation
├── main.py            # Training pipeline entrypoint
├── environment.yml    # Conda environment
└── requirements.txt   # pip requirements
```

---

## 🔧 Scripts

| Command | Description |
|---------|-------------|
| `pnpm dev` | Start frontend and backend together |
| `pnpm dev:web` | Start frontend only |
| `pnpm dev:api` | Start backend API only |
| `pnpm build` | Build frontend for production |
| `pnpm run package` | Build deployable `dist` bundle |

---

## 🏋️ Training Pipeline (Optional)

To train the visual encoder from scratch or regenerate the retrieval index:

```bash
conda activate kline-env
python main.py --steps all
# or run specific steps: --steps 1,2,3,3.5,4,5
```

See comments in `main.py` for details.

---

## 📜 License

This project is licensed under the [MIT License](./LICENSE).
