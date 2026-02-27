# K-line Similarity Analysis System

A K-line (candlestick) similarity retrieval and analysis system.  
Users can upload a candlestick chart or provide OHLC data, retrieve similar historical windows, and generate an LLM-based analysis report.

## Requirements

- **Node.js** 18+
- **pnpm** or **npm**
- **Python** 3.11+ (backend API)
- **Conda** (recommended) or pip

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/kline.git
cd kline
```

### 2. Install Node dependencies

```bash
pnpm install
# or
npm install
```

### 3. Set up the Python environment

**Option A: Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate kline-env
```

**Option B: pip**

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables (optional)

Copy the example env file and fill in your settings:

```bash
cp apps/web/.env.example apps/web/.env.local
```

Edit `apps/web/.env.local` and provide Supabase configuration (used for user authentication):

- `NEXT_PUBLIC_SUPABASE_URL` - Supabase project URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Supabase anon key
- `NEXT_PUBLIC_API_URL` - Backend API base URL, defaults to `http://localhost:8000` if empty

> If you don't need sign-in features, you can leave these empty; some features may be limited.

### 5. Run the project

```bash
pnpm dev
# or
npm run dev
```

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000>

## Project Structure

```text
kline/
├── apps/web          # Next.js frontend
├── services/api      # FastAPI backend
├── services/training # Training and inference logic
├── main.py           # Training pipeline entrypoint
├── environment.yml   # Conda environment
└── requirements.txt  # pip requirements
```

## Scripts

| Command            | Description                                  |
|--------------------|----------------------------------------------|
| `pnpm dev`         | Start frontend and backend together          |
| `pnpm dev:web`     | Start frontend only                          |
| `pnpm dev:api`     | Start backend API only                       |
| `pnpm build`       | Build frontend for production                |
| `pnpm run package` | Build deployable `dist` bundle               |

## Training pipeline (optional)

If you want to train the model from scratch or regenerate the retrieval index:

```bash
conda activate kline-env
python main.py --steps all
# or run specific steps: --steps 1,2,3,3.5,4,5
```

See comments in `main.py` for details.

## License

MIT
