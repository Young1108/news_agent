import requests
import json

class LLMService:
    def __init__(self):
        self.conversation_history = []  # 用于存储对话历史

    def ask_llama(self, prompt):
        """
        利用 Llama API 模型生成回复。
        参数:
            prompt (str): 提示信息，作为模型的输入。
        返回:
            str: Llama 模型的回复内容。
        """
        url = "http://localhost:11434/api/chat"

        # 将用户输入添加到对话历史中
        self.conversation_history.append({"role": "user", "content": prompt})

        data = {
            "model": "llama3:instruct",
            "messages": [
                {"role": "system",
                 "content": ""}
            ] + self.conversation_history,  # 将对话历史加入到消息中
            "temperature": 0.2,
            "max_tokens": 40,
            "top_p": 0.8,
            "stop": "bye",
            "stream": False
        }
        response = requests.post(url, json=data)  # 发送 POST 请求, 并将响应存储在 response 变量中
        # print("response:", response)
        # print("response.status_code:", response.status_code)
        if response.status_code == 200:  # 检查响应状态码是否为 200 (成功)
            try:
                response_json = response.json()  # 将响应 JSON 解码为 Python 字典
                llama_reply = response_json['message']['content']

                # 将 Llama 的回复添加到对话历史中
                self.conversation_history.append({"role": "assistant", "content": llama_reply})

                return llama_reply  # 返回 Llama 的回复内容
            except json.JSONDecodeError as e:
                print("Error decoding JSON response:", e)
                print("Response content:", response.content)
            except KeyError as e:
                print("KeyError:", e)
                print("Response JSON structure:", response_json)
        else:
            print("Error:", response.text)
            return None
