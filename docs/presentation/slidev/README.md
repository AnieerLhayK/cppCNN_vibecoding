# cppCNN 课堂汇报幻灯片

基于 [Slidev](https://sli.dev/) 的 HTML 幻灯片，主题：纯 C++ CNN 交通标志识别系统。

## 快速启动

```powershell
# 进入幻灯片目录
cd docs/presentation/slidev

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

浏览器访问终端输出的地址（默认 `http://localhost:3030`），按空格键翻页。

## 构建静态文件

```powershell
npm run build
```

构建产物在 `dist/` 目录，可部署到任意静态服务器。

## 导出 PDF（可选）

```powershell
npm run export
```

需要安装 [Playwright](https://playwright.dev) 浏览器内核：
`npx playwright install chromium`

## 文件说明

| 文件 | 说明 |
|---|---|
| `slides.md` | 幻灯片主内容，Slidev Markdown 格式 |
| `package.json` | 项目依赖与脚本 |
| `.markdownlint.json` | Markdown 格式检查配置 |
| `README.md` | 本文件 |

## 技术栈

- [Slidev](https://sli.dev/) — 基于 Markdown 的演示框架
- `@slidev/theme-seriph` — Slidev 内置 Seriph 主题
- Mermaid — 流程图渲染
