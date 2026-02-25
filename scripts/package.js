/**
 * 打包脚本 - 仅打包以下所需文件：
 * 1) python main.py --steps 4 所需：main.py、训练代码、output/.../dataset_splits 下训练输入
 * 2) 前后端：apps/web、services/api
 * 使用方法: npm run package 或 pnpm run package
 * 输出目录: dist/
 */

const fs = require('fs');
const path = require('path');

const projectRoot = path.resolve(__dirname, '..');
const distDir = path.join(projectRoot, 'dist');

console.log('🚀 开始打包（仅 step4 + 前后端）...\n');

if (fs.existsSync(distDir)) {
  console.log('📦 清理旧的打包目录...');
  try {
    fs.rmSync(distDir, { recursive: true, force: true });
  } catch (e) {
    console.log('⚠️  无法完全删除 dist（可能被占用），将覆盖写入');
  }
}
fs.mkdirSync(distDir, { recursive: true });

function copyFile(src, dest, logName) {
  if (!fs.existsSync(src)) return false;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
  if (logName) console.log('✅ ' + logName);
  return true;
}

// ---------- 根目录 ----------
console.log('📋 根目录...\n');
const rootFiles = [
  'package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml', '.npmrc',
  'environment.yml', 'requirements.txt', 'main.py', 'start_server.py', 'start_with_conda.bat',
  'README.md', '.gitignore',
];
rootFiles.forEach(f => {
  copyFile(path.join(projectRoot, f), path.join(distDir, f), '复制: ' + f);
});

// ---------- 前端 apps/web ----------
console.log('\n📁 apps/web/');
const appsWeb = path.join(projectRoot, 'apps', 'web');
const distWeb = path.join(distDir, 'apps', 'web');
if (fs.existsSync(appsWeb)) {
  copyDirFiltered(appsWeb, distWeb, 'apps/web', (rel) => {
    if (rel.includes('node_modules') || rel.includes('.next') || rel.includes('.git')) return true;
    if (rel.includes('.env.local') || rel.includes('next-env.d.ts')) return true;
    return false;
  });
}

// ---------- 后端 services/api ----------
console.log('\n📁 services/api/');
const apiSrc = path.join(projectRoot, 'services', 'api');
const apiDest = path.join(distDir, 'services', 'api');
if (fs.existsSync(apiSrc)) {
  copyDirFiltered(apiSrc, apiDest, 'services/api', (rel) => {
    return rel.includes('__pycache__') || rel.includes('.pyc');
  });
}

// ---------- 训练 step4 所需：仅部分 services/training ----------
// main.py --steps 4 需要：clip_contrastive_trainer.py, inference_encoder.py, scripts/train_with_pairs.py, scripts/train_simsiam.py
// 以及 output/<name>/dataset_splits 下的 train_anchor_images.npy, train_positive_images.npy, train_pairs_metadata.json, split_info.json
console.log('\n📁 services/training/（仅 step4 所需）');
const trainingSrc = path.join(projectRoot, 'services', 'training');
const trainingDest = path.join(distDir, 'services', 'training');
if (fs.existsSync(trainingSrc)) {
  fs.mkdirSync(trainingDest, { recursive: true });
  ['clip_contrastive_trainer.py', 'inference_encoder.py'].forEach(f => {
    copyFile(path.join(trainingSrc, f), path.join(trainingDest, f), '复制: services/training/' + f);
  });
  const scriptsDest = path.join(trainingDest, 'scripts');
  fs.mkdirSync(scriptsDest, { recursive: true });
  ['train_with_pairs.py', 'train_simsiam.py'].forEach(f => {
    copyFile(path.join(trainingSrc, 'scripts', f), path.join(scriptsDest, f), '复制: services/training/scripts/' + f);
  });
}

// ---------- output 下仅 dataset_splits（step4 输入） ----------
function copyStep4OutputInputs() {
  const outputSrc = path.join(projectRoot, 'services', 'training', 'output');
  const outputDest = path.join(distDir, 'services', 'training', 'output');
  if (!fs.existsSync(outputSrc)) return;

  const step4Files = [
    'train_anchor_images.npy',
    'train_positive_images.npy',
    'train_pairs_metadata.json',
    'split_info.json',
  ];

  const subdirs = fs.readdirSync(outputSrc, { withFileTypes: true }).filter(d => d.isDirectory());
  for (const d of subdirs) {
    const name = d.name;
    const dsPath = path.join(outputSrc, name, 'dataset_splits');
    if (!fs.existsSync(dsPath)) continue;
    const destSub = path.join(outputDest, name, 'dataset_splits');
    let copied = 0;
    for (const f of step4Files) {
      const srcFile = path.join(dsPath, f);
      if (fs.existsSync(srcFile)) {
        fs.mkdirSync(destSub, { recursive: true });
        fs.copyFileSync(srcFile, path.join(destSub, f));
        console.log('✅ 复制: services/training/output/' + name + '/dataset_splits/' + f);
        copied++;
      }
    }
    if (copied > 0) {
      const miss = [];
      if (!fs.existsSync(path.join(dsPath, 'train_anchor_images.npy'))) miss.push('train_anchor_images.npy');
      if (!fs.existsSync(path.join(dsPath, 'train_positive_images.npy'))) miss.push('train_positive_images.npy');
      if (miss.length) console.log('   ⚠️  缺少 ' + miss.join('、') + ' 时，服务器上 python main.py --steps 4 将报错，请先在本地跑完 Step3.5 再打包。\n');
    }
  }
}
console.log('\n📁 services/training/output/.../dataset_splits（step4 输入）');
copyStep4OutputInputs();

// ---------- 打包脚本（便于在 dist 中再次打包） ----------
console.log('\n📁 scripts/');
fs.mkdirSync(path.join(distDir, 'scripts'), { recursive: true });
copyFile(
  path.join(projectRoot, 'scripts', 'package.js'),
  path.join(distDir, 'scripts', 'package.js'),
  '复制: scripts/package.js'
);

// ---------- 安装说明 ----------
const installReadme = `# 部署说明（仅 step4 + 前后端）

本包包含：
- **前后端**：apps/web、services/api
- **python main.py --steps 4** 所需：main.py、services/training 下训练代码、output/.../dataset_splits 下 train_*.npy 与 train_pairs_metadata.json

## 1. Node 依赖

\`\`\`bash
pnpm install
# 或 npm install
\`\`\`

## 2. Python 环境

\`\`\`bash
# 方式 A：Conda（推荐）
conda env create -f environment.yml
conda activate kline-env

# 方式 B：pip
pip install -r requirements.txt
\`\`\`

## 3. 运行前后端

\`\`\`bash
pnpm dev
# 或先 pnpm build 再分别启动
\`\`\`

## 4. 仅运行训练（step4）

\`\`\`bash
conda activate kline-env
python main.py --steps 4
\`\`\`

默认使用 \`services/training/output/dow30_2010_2021/dataset_splits/\` 下的 train_anchor_images.npy、train_positive_images.npy、train_pairs_metadata.json。若打包时这些 .npy 不存在，请先在本地跑完 Step3、Step3.5 再打包。
`;

fs.writeFileSync(path.join(distDir, 'INSTALL.md'), installReadme);

console.log('\n✨ 打包完成！');
console.log('📦 输出目录: ' + distDir);
console.log('\n📋 本包仅含：前后端 + python main.py --steps 4 所需文件');
console.log('   部署后：pnpm install → conda env create -f environment.yml → pnpm dev / python main.py --steps 4\n');
console.log('📋 部署步骤:');
console.log('   1. 将 dist/ 内容拷到服务器');
console.log('   2. pnpm install（或 npm install）');
console.log('   3. conda env create -f environment.yml && conda activate kline-env');
console.log('   4. 启动: pnpm dev；训练: python main.py --steps 4\n');

function copyDirFiltered(src, dest, baseRel, shouldSkip) {
  if (!fs.existsSync(src)) return;
  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const e of entries) {
    const rel = (baseRel + '/' + e.name).replace(/\\/g, '/');
    if (shouldSkip(rel)) continue;
    const srcPath = path.join(src, e.name);
    const destPath = path.join(dest, e.name);
    if (e.isDirectory()) {
      fs.mkdirSync(destPath, { recursive: true });
      copyDirFiltered(srcPath, destPath, rel, shouldSkip);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}
