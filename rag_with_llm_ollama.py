"""
完整的 RAG 系统 - 使用 Ollama 本地 LLM
支持智能问答、多轮对话
后续会优化多轮对话记忆功能
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from modelscope import snapshot_download

# 加载环境变量
load_dotenv()
logging.getLogger("modelscope").setLevel(logging.ERROR)

# 推荐的模型配置
EMBEDDING_MODELS = {
    '1': {
        'name': 'BAAI/bge-small-zh-v1.5',
        'modelscope_id': 'AI-ModelScope/bge-small-zh-v1.5',
        'description': '小型中文模型（推荐）- 约100MB，速度快',
    },
    '2': {
        'name': 'BAAI/bge-base-zh-v1.5',
        'modelscope_id': 'AI-ModelScope/bge-base-zh-v1.5',
        'description': '中型中文模型 - 约400MB，效果更好',
    }
}

LLM_MODELS = {
    '1': {
        'name': 'qwen2.5:7b',
        'description': '通义千问 7B（推荐）- 中文最好',
    },
    '2': {
        'name': 'qwen2.5:3b',
        'description': '通义千问 3B - 速度更快',
    },
    '3': {
        'name': 'glm4:9b',
        'description': '智谱 GLM4 - 推理能力强',
    }
}

def download_embedding_model(modelscope_id, cache_dir="./models"):
    """从 ModelScope 下载嵌入模型"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # 先尝试本地加载，避免重复下载
    try:
        model_dir = snapshot_download(
            modelscope_id,
            cache_dir=str(cache_path),
            revision='master',
            local_files_only=True,
        )
        print(f"   [OK] 使用本地模型缓存")
        return model_dir
    except Exception:
        pass

    # 本地没有再下载
    try:
        print(f"   下载嵌入模型...")
        model_dir = snapshot_download(
            modelscope_id,
            cache_dir=str(cache_path),
            revision='master'
        )
        print(f"   [OK] 模型下载成功")
        return model_dir
    except Exception as e:
        print(f"   [ERROR] 下载失败: {e}")
        return None

def select_embedding_model():
    """选择嵌入模型"""
    print("\n选择嵌入模型（用于文档向量化）：")
    print("=" * 70)
    for key, model in EMBEDDING_MODELS.items():
        print(f"{key}. {model['name']}")
        print(f"   {model['description']}")
    print("=" * 70)
    
    choice = input("请选择 (1-2) [默认: 1]: ").strip() or '1'
    if choice not in EMBEDDING_MODELS:
        choice = '1'
    
    selected = EMBEDDING_MODELS[choice]
    print(f"\n[OK] 已选择: {selected['name']}")
    return selected

def select_llm_model():
    """选择 LLM 模型"""
    print("\n选择 LLM 模型（用于生成答案）：")
    print("=" * 70)
    for key, model in LLM_MODELS.items():
        print(f"{key}. {model['name']}")
        print(f"   {model['description']}")
    print("=" * 70)
    print("\n提示：如果模型未下载，请先运行：")
    print("  ollama pull qwen2.5:7b")
    print()
    
    choice = input("请选择 (1-3) [默认: 1]: ").strip() or '1'
    if choice not in LLM_MODELS:
        choice = '1'
    
    selected = LLM_MODELS[choice]
    print(f"\n[OK] 已选择: {selected['name']}")
    return selected

def load_documents(directory="./documents"):
    """从目录加载文档"""
    documents = []
    doc_path = Path(directory)
    
    if not doc_path.exists():
        print(f"创建文档目录: {directory}")
        doc_path.mkdir(parents=True, exist_ok=True)
        return documents
    
    # 加载 TXT 文件
    for txt_file in doc_path.glob("*.txt"):
        print(f"  [*] 加载 TXT: {txt_file.name}")
        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"      [!] 错误: {e}")
    
    # 加载 PDF 文件
    for pdf_file in doc_path.glob("*.pdf"):
        print(f"  [*] 加载 PDF: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"      [!] 无法加载: {e}")
    
    # 加载 Markdown 文件
    for md_file in doc_path.glob("*.md"):
        print(f"  [*] 加载 Markdown: {md_file.name}")
        try:
            loader = TextLoader(str(md_file), encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"      [!] 错误: {e}")
    
    return documents

def main():
    print("=" * 70)
    print("完整 RAG 系统 - 带 Ollama LLM（智能问答）")
    print("=" * 70)
    
    # 选择模型
    embedding_info = select_embedding_model()
    llm_info = select_llm_model()
    
    # 1. 加载文档
    print("\n[1/6] 从 documents/ 目录加载文档...")
    documents = load_documents("./documents")
    
    if not documents:
        print("\n[!] 没有找到任何文档！")
        print("请将文档放入 documents/ 目录后重新运行")
        return
    
    print(f"\n   [OK] 总共加载了 {len(documents)} 个文档")
    
    # 2. 文档分块
    print("\n[2/6] 将文档分割成小块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    print(f"   [OK] 分成了 {len(splits)} 个块")
    
    # 3. 下载嵌入模型
    print(f"\n[3/6] 准备嵌入模型...")
    model_dir = download_embedding_model(
        embedding_info['modelscope_id'],
        cache_dir="./models"
    )
    
    if not model_dir:
        print("\n[ERROR] 嵌入模型准备失败")
        return
    
    # 4. 创建向量数据库
    print(f"\n[4/6] 创建向量数据库...")
    print(f"   使用模型: {embedding_info['name']}")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_dir,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 使用模型特定的数据库目录
        model_short_name = embedding_info['name'].split('/')[-1]
        db_dir = f"./data/chroma_db_{model_short_name}"
        
        db_path = Path(db_dir)
        db_path.mkdir(parents=True, exist_ok=True)

        # 如果数据库已存在，直接复用
        has_db = (db_path / "chroma.sqlite3").exists() or any(db_path.glob("*.parquet"))
        if has_db:
            vectorstore = Chroma(
                persist_directory=db_dir,
                embedding_function=embeddings
            )
            print(f"   [OK] 使用已有向量数据库")
        else:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=db_dir
            )
            print(f"   [OK] 向量数据库创建完成！")
        print(f"   数据库路径: {db_dir}")
        
    except Exception as e:
        print(f"\n   [ERROR] 创建向量数据库失败: {e}")
        return
    
    # 5. 初始化 LLM
    print(f"\n[5/6] 初始化 Ollama LLM...")
    print(f"   模型: {llm_info['name']}")
    print(f"   正在连接...")
    
    try:
        llm = OllamaLLM(
            model=llm_info['name'],
            temperature=0.3,  # 降低温度，更准确
            num_predict=512,   # 限制最大生成长度
        )
        
        # 测试 LLM
        test_response = llm.invoke("你好")
        print(f"   [OK] LLM 就绪！")
        
    except Exception as e:
        print(f"\n   [ERROR] LLM 初始化失败: {e}")
        print("\n可能的原因：")
        print(f"  1. 模型未下载，请运行：ollama pull {llm_info['name']}")
        print("  2. Ollama 服务未启动")
        print("  3. 模型名称错误")
        return
    
    # 6. 创建问答链
    print(f"\n[6/6] 创建问答链...")
    
    # 自定义中文提示词
    template = """你是一个专业的文档助手。请根据以下提供的上下文信息来回答问题。

【重要规则】
1. 只根据提供的上下文信息回答
2. 如果上下文中没有相关信息，请明确说"根据提供的文档无法回答这个问题"
3. 回答要简洁、准确、条理清晰
4. 可以用列表或步骤的形式回答
5. 回答控制在200字以内

【上下文信息】
{context}

【问题】
{question}

【回答】
"""
    
    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 检索前3个最相关的文档
        ),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    
    print(f"   [OK] 问答链创建完成！")
    
    # 7. 交互式问答
    print("\n" + "=" * 70)
    print("🤖 智能问答系统已就绪！")
    print("=" * 70)
    print("\n提示：")
    print("  - 输入问题，AI 会基于你的文档回答")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 首次回答可能较慢（加载模型）")
    print("  - CPU 环境下每次回答约 30-60 秒")
    print("\n" + "=" * 70)
    
    while True:
        question = input("\n💬 你的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', '退出', 'q']:
            break
        
        if not question:
            continue
        
        print(f"\n🔍 正在思考...")
        print("   [1/3] 检索相关文档...")
        
        try:
            # 调用问答链
            result = qa_chain.invoke({"query": question})
            
            print("   [2/3] 生成答案中...")
            print("   [3/3] 完成！\n")
            
            # 显示答案
            print("=" * 70)
            print("🤖 AI 回答：\n")
            print(result['result'])
            print("\n" + "=" * 70)
            
            # 显示引用来源
            if result.get('source_documents'):
                print("\n📚 参考来源：")
                sources = set()
                for i, doc in enumerate(result['source_documents'][:3], 1):
                    source = doc.metadata.get('source', 'unknown')
                    if source not in sources:
                        sources.add(source)
                        print(f"  [{len(sources)}] {source}")
                        # 显示部分原文
                        content_preview = doc.page_content[:100].replace('\n', ' ')
                        print(f"      {content_preview}...")
            
        except Exception as e:
            print(f"\n[ERROR] 生成答案失败: {e}")
            print("\n请检查：")
            print("  1. Ollama 服务是否正常运行")
            print("  2. 模型是否已下载")
            print("  3. 问题是否过于复杂")
    
    print("\n" + "=" * 70)
    print("👋 感谢使用！")
    print("=" * 70)
    print("\n提示：")
    print(f"  - 嵌入模型已缓存到: ./models/")
    print(f"  - 向量数据库已保存到: {db_dir}")
    print(f"  - LLM 模型: {llm_info['name']}")
    print("  - 下次运行会更快（无需重新下载）")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] 程序已中断")
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
