import os
import time
from openai import OpenAI
import httpx
import json
import datetime
import numpy as np

class LLMPrompter():
    def __init__(self, gpt_version, api_key, base_url=None) -> None:
        """
        初始化 LLMPrompter
        
        Args:
            gpt_version: GPT 模型版本，如 "gpt-3.5-turbo", "gpt-4"
            api_key: API 密钥
            base_url: 可选的 API base URL，用于第三方服务（如 poloapi）
                     如果不提供，使用默认的 OpenAI API
        """
        self.gpt_version = gpt_version
        if api_key is None:
            raise ValueError("OpenAI API key is not provided.")
        else:
            # 使用新的 OpenAI 1.0.0+ API，配置代理
            proxy_client = httpx.Client(
                proxy="http://127.0.0.1:7890",
                timeout=60.0
            )
            # 如果提供了 base_url，使用第三方 API 服务
            client_kwargs = {
                "api_key": api_key,
                "http_client": proxy_client,
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str) -> str:
        while True:
            try:
                # 统一使用 ChatCompletion API（新版本不再支持 Completion API）
                if isinstance(prompt, dict):
                    # prompt 是字典格式（包含 system 和 user）
                    messages = [
                        {"role": "system", "content": prompt['system']},
                        {"role": "user", "content": prompt['user']},
                    ]
                else:
                    # prompt 是字符串，转换为 messages 格式
                    messages = [{"role": "user", "content": prompt}]
                
                response = self.client.chat.completions.create(
                    model=self.gpt_version,
                    messages=messages,
                    **sampling_params
                )
                
                # 获取响应内容
                response_content = response.choices[0].message.content
                
            except Exception as e:
                print("Request failed, sleep 2 secs and try again...", e)
                time.sleep(2)
                continue
            break

        if save:
            key = self.make_key()
            output = {}
            os.system('mkdir -p {}'.format(save_dir))
            if os.path.exists(os.path.join(save_dir, 'response.json')):
                with open(os.path.join(save_dir, 'response.json'), 'r') as f:
                    prev_response = json.load(f)
                    output = prev_response

            with open(os.path.join(save_dir, 'response.json'), 'w') as f:
                output[key] = {
                    'prompt': prompt,
                    'sampling_params': sampling_params,
                    'response': response_content.strip()
                }
                json.dump(output, f, indent=4)
            
        return response_content.strip(), None

    def make_key(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")