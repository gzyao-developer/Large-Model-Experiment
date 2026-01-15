# RAG 模型实验环境

这是一个用于实验 RAG（检索增强生成）模型的完整环境。

## 什么是 RAG？

RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的AI技术：
1. **检索**：从知识库中检索相关文档
2. **增强**：将检索到的内容作为上下文
3. **生成**：使用LLM基于上下文生成答案

## 安装步骤

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量（创建 `.env` 文件）：
```
OPENAI_API_KEY=your_openai_api_key_here
```

如果你想使用本地模型，可以不配置 OpenAI API。

## 使用方法

### 1. 简单示例（使用内置文档）
```bash
python rag_simple.py
```

### 2. 使用自己的文档
```bash
python rag_with_docs.py
```
将你的文档（PDF、TXT等）放入 `documents/` 文件夹，然后运行上述命令。

### 3. 交互式问答
```bash
python rag_interactive.py
```

## 项目结构

```
rag_test/
├── requirements.txt      # 依赖包
├── .env                  # 环境变量（需自行创建）
├── rag_simple.py         # 简单RAG示例
├── rag_with_docs.py      # 使用文档的RAG
├── rag_interactive.py    # 交互式RAG问答
├── documents/            # 存放你的文档
└── data/                 # 向量数据库存储目录
```

## 技术栈

- **LangChain**: RAG框架
- **ChromaDB**: 向量数据库
- **Sentence Transformers**: 文本向量化（支持本地运行）
- **OpenAI**: LLM（可选，也可以用其他模型）

## 常见问题

**Q: 如何使用本地模型而不是 OpenAI？**
A: 可以使用 Ollama、HuggingFace 模型等，参考代码中的注释进行修改。

**Q: 向量数据库存在哪里？**
A: 默认存储在 `./data/chroma_db` 目录。

**Q: 支持哪些文档格式？**
A: 目前支持 PDF、TXT、Markdown 等常见格式。
