import ast
import pickle
import re
from collections import Counter

from ModelShare_with_DSR_final import Rule
from ModelShare_with_DSR_final import Predicate
import numpy as np
import openai
import pandas as pd
from openai import OpenAI
from ModelShare_with_DSR_final import Model
openai.api_key = 'sk-Sz5VQcsOmLGRz0Ne837cEc158d9f477292B856335cEfD361'

try:
    client = OpenAI(api_key=openai.api_key, base_url="https://api.gpt.ge/v1/")
    response = client.chat.completions.create(
        model="deepseek-r1-250528",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=16000,
        stop=None,
        temperature=0.7
    )

    # 获取并解析生成的文本
    generated_text = response.choices[0].message.content
    print(generated_text)
except Exception as e:
    print(f"Error: {e}")