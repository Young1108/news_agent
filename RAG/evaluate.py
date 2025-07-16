import json
from rag_llm import NewsSearcherRAG
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rouge_chinese import Rouge
import jieba

class RAGEvaluator:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.rouge = Rouge()
        self.similarity_threshold = 0.3
        
    def load_test_data(self):
        """从document_store.json加载测试数据"""
        with open(self.rag_system.store_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calculate_retrieval_precision(self, query, retrieved_docs, top_k=4):
        """计算检索精确度"""
        # 使用嵌入模型计算相似度
        query_embedding = self.rag_system.embedding_model.embed_query(query)
        
        relevant_count = 0
        similarities = []
        for doc in retrieved_docs:
            doc_embedding = self.rag_system.embedding_model.embed_query(doc)
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append(similarity)
            if similarity > self.similarity_threshold:
                relevant_count += 1
                
        precision = relevant_count / len(retrieved_docs) if retrieved_docs else 0
        avg_similarity = np.mean(similarities) if similarities else 0
        return precision, avg_similarity

    def calculate_rouge_scores(self, reference, hypothesis):
        """
        计算参考文本和生成文本之间的ROUGE评分
        
        Args:
            reference: str, 参考文本（标准答案）
            hypothesis: str, 待评估的生成文本
            
        Returns:
            dict: 包含rouge-1、rouge-2和rouge-l的F1分数
        """
        # 将长文本按句号分割成句子列表，并去除空白句子
        reference_sents = [sent for sent in reference.split('。') if sent.strip()]
        hypothesis_sents = [sent for sent in hypothesis.split('。') if sent.strip()]
        
        # 存储所有句子对的ROUGE分数
        rouge_scores = []
        
        # 对每一对参考句子和生成句子计算ROUGE分数
        for ref_sent in reference_sents:
            for hyp_sent in hypothesis_sents:
                # 使用结巴分词，将句子转换为以空格分隔的词序列
                ref_tokens = ' '.join(jieba.cut(ref_sent))
                hyp_tokens = ' '.join(jieba.cut(hyp_sent))
                
                try:
                    # 计算当前句子对的ROUGE分数
                    # get_scores返回一个列表，取第一个元素（包含rouge-1、rouge-2、rouge-l的分数）
                    score = self.rouge.get_scores(hyp_tokens, ref_tokens)[0]
                    rouge_scores.append(score)
                except:
                    # 如果计算失败，跳过当前句子对
                    continue
        
        if rouge_scores:
            # 从所有句子对中选择每种ROUGE度量的最高F1分数
            max_score = {
                'rouge-1': {'f': max(score['rouge-1']['f'] for score in rouge_scores)},  
                'rouge-2': {'f': max(score['rouge-2']['f'] for score in rouge_scores)},  
                'rouge-l': {'f': max(score['rouge-l']['f'] for score in rouge_scores)}
            }
            return max_score
    
        # 如果没有成功计算出任何分数，返回全0分数
        return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}

    def evaluate_rag_system(self, test_queries):
        """评估RAG系统的整体表现"""
        results = {
            'retrieval_precision': [],
            'avg_similarity': [],
            'rouge_scores': []
        }
        
        for query in test_queries:
            # 获取检索文档
            retrieved_docs = self.rag_system.retrieve_documents(query)
            print(f"检索到的文档: {retrieved_docs}")
            
            # 计算检索精确度和平均相似度
            precision, avg_similarity = self.calculate_retrieval_precision(query, retrieved_docs)
            results['retrieval_precision'].append(precision)
            results['avg_similarity'].append(avg_similarity)
            
            # 生成回答
            generated_response = self.rag_system.generate_response(query)
            print(f"生成的回答: {generated_response}")
            
            # 使用检索文档作为参考
            if retrieved_docs:
                combined_reference = ''.join(retrieved_docs) # 将检索到的文档合并为一个长文本
                print(f"参考文本: {combined_reference}")
                rouge_scores = self.calculate_rouge_scores(combined_reference, generated_response)
                results['rouge_scores'].append(rouge_scores)
        
        # 计算平均分数
        avg_results = {
            # 'avg_precision': np.mean(results['retrieval_precision']),
            # 'avg_similarity': np.mean(results['avg_similarity']),
            'avg_rouge_1': np.mean([score['rouge-1']['f'] for score in results['rouge_scores']]),
            'avg_rouge_2': np.mean([score['rouge-2']['f'] for score in results['rouge_scores']]),
            'avg_rouge_l': np.mean([score['rouge-l']['f'] for score in results['rouge_scores']])
        }
        
        return avg_results

def main():
    # 初始化RAG系统
    rag_system = NewsSearcherRAG()
    evaluator = RAGEvaluator(rag_system)
    
    # 测试查询
    test_queries = [
        "今天有什么重要新闻？",
        "最近发生了什么大事？",
        "现在有什么热点新闻？",
        "最近的重要会议有哪些？"
    ]
    
    # 运行评估
    evaluation_results = evaluator.evaluate_rag_system(test_queries)
    
    # 打印评估结果
    print("\n=== RAG系统评估结果 ===")
    # print(f"平均检索精确度: {evaluation_results['avg_precision']:.4f}")
    # print(f"平均文档相似度: {evaluation_results['avg_similarity']:.4f}")
    print(f"平均ROUGE-1分数: {evaluation_results['avg_rouge_1']:.4f}")
    print(f"平均ROUGE-2分数: {evaluation_results['avg_rouge_2']:.4f}")
    print(f"平均ROUGE-L分数: {evaluation_results['avg_rouge_l']:.4f}")

if __name__ == "__main__":
    main()
