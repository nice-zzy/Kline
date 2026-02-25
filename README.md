# K-Line 相似度分析系统

K 线图相似度检索与分析系统，支持上传 K 线图进行相似历史案例检索与形态分析。

## 环境要求

- **Node.js** 18+
- **pnpm** 或 **npm**
- **Python** 3.11+（后端 API）
- **Conda**（推荐）或 pip

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/kline.git
cd kline
```

### 2. 安装 Node 依赖

```bash
pnpm install
# 或
npm install
```

### 3. 配置 Python 环境

**方式 A：使用 Conda（推荐）**

```bash
conda env create -f environment.yml
conda activate kline-env
```

**方式 B：使用 pip**

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量（可选）

复制环境变量模板并填写：

```bash
cp apps/web/.env.example apps/web/.env.local
```

编辑 `apps/web/.env.local`，填入 Supabase 配置（用于用户登录）：

- `NEXT_PUBLIC_SUPABASE_URL` - Supabase 项目 URL
- `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Supabase 匿名密钥
- `NEXT_PUBLIC_API_URL` - 后端 API 地址，留空则默认 `http://localhost:8000`

> 若无需登录功能，可暂时留空；部分功能可能受限。

### 5. 运行项目

```bash
pnpm dev
# 或
npm run dev
```

- 前端：<http://localhost:3000>
- 后端 API：<http://localhost:8000>

**Windows 用户** 也可使用一键启动：

```bash
start_with_conda.bat
```

（需先执行 `conda activate kline-env` 或确保 conda 环境已激活）

## 项目结构

```
kline/
├── apps/web          # Next.js 前端
├── services/api      # FastAPI 后端
├── services/training # 训练与推理逻辑
├── main.py           # 训练流水线入口
├── environment.yml   # Conda 环境
└── requirements.txt  # pip 依赖
```

## 可用命令

| 命令 | 说明 |
|------|------|
| `pnpm dev` | 同时启动前端与后端 |
| `pnpm dev:web` | 仅启动前端 |
| `pnpm dev:api` | 仅启动后端 |
| `pnpm build` | 构建前端生产版本 |
| `pnpm run package` | 打包为可部署的 dist 目录 |

## 训练流程（可选）

若需从零训练模型或更新检索索引：

```bash
conda activate kline-env
python main.py --steps all
# 或分步执行：--steps 1,2,3,3.5,4,5
```

详见 `main.py` 内注释。

## License

MIT
