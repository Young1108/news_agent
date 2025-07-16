import json
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class NewsSearcherRAG:
    def __init__(self, model_name='qwen2.5', embedding_model_name='znbang/bge:large-zh-v1.5-q8_0', base_url='http://localhost:11434', store_file="document_store.json"):
        self.model = Ollama(base_url=base_url, model=model_name, temperature=0)
        self.embedding_model = OllamaEmbeddings(base_url=base_url, model=embedding_model_name)
        self.store_file = store_file  # 用于存储文档的文件
        self.document_store = []  # 初始化文档存储
        self.load_documents()  # 加载已有文档

    def extract_content_from_url(self, url):
        """从URL中提取内容."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch URL: {url}, Status Code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(separator='\n', strip=True)

        lines = content.split('\n')
        extracted_lines = []
        # print(lines)
        for i, line in enumerate(lines):
            # print(i, line)
            if "最近 100 条记录" in line:
                extracted_lines = lines[i + 1:i + 50]  # 获取前50行记录
                print("extracted_lines", extracted_lines)
                break

        filtered_lines = []
        for idx, line in enumerate(extracted_lines):
            if ((idx + 1) % 3 == 1 or (idx + 1) % 3 == 2):
                if((idx+1)%3==1):
                    # 转换日期格式
                    date = datetime.strptime(line.strip(), '%Y-%m-%d').strftime('%Y年%m月%d日')
                if((idx+1)%3==2):
                    content = line
                    filtered_lines.append({"日期": date, "内容": content})

        return filtered_lines

    def add_document(self, content):
        """把文档添加到文档存储中"""
        self.document_store.extend(content)  # 直接将每条记录添加到文档存储中
        self.save_documents()  # 每次添加文档后保存到磁盘

    def save_documents(self):
        """保存文档存储到文件中"""
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(self.document_store, f, ensure_ascii=False, indent=4)

    def load_documents(self):
        """从文件中加载文档存储"""
        try:
            with open(self.store_file, "r", encoding="utf-8") as f:
                self.document_store = json.load(f)
        except FileNotFoundError:
            self.document_store = []  # 如果文件不存在，则初始化为空列表

    def retrieve_documents(self, query, top_k=4):
        """检索相关文档"""
        # 计算查询的嵌入
        query_embedding = self.embedding_model.embed_query(query)

        # 计算文档内容的嵌入并与查询进行比较
        doc_embeddings = []
        for doc in self.document_store:
            doc_embedding = self.embedding_model.embed_query(doc["内容"])  # 计算每个文档的嵌入
            doc_embeddings.append((doc, doc_embedding))

        # 计算查询与每个文档的余弦相似度
        ranked_docs = sorted(
            [(doc, cosine_similarity([query_embedding], [doc_embedding])[0][0]) for doc, doc_embedding in doc_embeddings],
            key=lambda x: x[1],
            reverse=True
        )

        return [doc[0]["内容"] for doc in ranked_docs[:top_k]]

    def generate_response(self, query, top_k=5):
        """生成RAG响应"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        relevant_docs = self.retrieve_documents(query, top_k=top_k)
        context = "\n".join(relevant_docs)

        prompt = f"今天是{current_date}。你是一个新闻播报助手,我提供给你的这些新闻都是真实的新闻信息,都是官方且权威的，不需要你来质疑，当用户询问你今天有什么新闻时，你只能回答今天发生的新闻。当用户问你最近发生了什么新闻时，回答请以“以下是近期的一些重要时政新闻”作为开头来进行回答：\n\n{context}\n\n问题：{query}\n\n回答："
        response = self.model(prompt)
        return response

    def rag_response(self, query):
        """Extract, store, and generate response."""
        url = "https://news.google.com/home?hl=zh-CN&gl=CN&ceid=CN:zh-Hans"  # 示例URL
        # 提取内容
        content = self.extract_content_from_url(url)
        # 存储提取的内容到文档
        self.add_document(content)
        # 生成RAG响应
        response = self.generate_response(query)
        return response

# 示例调用
if __name__ == "__main__":
    rag_service = NewsSearcherRAG()
    query = "今天有哪些重要的时政新闻？"
    response = rag_service.rag_response(query)
    print(response)
