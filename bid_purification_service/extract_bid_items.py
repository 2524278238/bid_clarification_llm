import pandas as pd
import os
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

def call_llm(prompt, api_key, base_url, model_id):
    if os.environ.get("MOCK_LLM") == "true":
        return json.dumps({
            "技术清标项-总结类": ["技术方案", "产品设计"],
            "技术清标项-字段类": ["规格型号"],
            "商务清标项-总结类": ["资质证明"],
            "商务清标项-字段类": ["投标报价", "工期"]
        })

    if not api_key:
        raise Exception("Missing API Key")
    
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in analyzing bid documents."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "enable_thinking": False,
        "stream": False,
        "n": 1,
        "response_format": {"type": "json_object"}
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    req = Request(endpoint, data=data, headers=headers, method="POST")

    try:
        with urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(body)
            content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
    except Exception as e:
        print(f"LLM Call Error: {e}")
        return None

def extract_excel_headers(file_path):
    if os.environ.get("MOCK_TEMPLATE_PARSE") == "true":
        return {"Sheet1": ["技术方案", "投标报价", "工期"]}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    xls = pd.ExcelFile(file_path)
    all_headers = {}
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Filter out unnamed columns and clean up strings (remove newlines)
        headers = [str(col).replace('\n', '').strip() for col in df.columns.tolist() if "Unnamed" not in str(col)]
        
        # Also, for "Project Basic Info", the structure is different (Name, Content), 
        # so we should read the values in the first column as "headers" or items.
        if sheet_name == '项目基本信息':
             if '名称' in df.columns:
                 # Clean up values in the '名称' column
                 headers = df['名称'].dropna().astype(str).apply(lambda x: x.replace('\n', '').strip()).tolist()
        
        all_headers[sheet_name] = headers

    return all_headers

def extract_bid_purification_items(template_path, model_url, model_key, model_id):
    try:
        headers_by_sheet = extract_excel_headers(template_path)
    except Exception as e:
        return {"error": str(e)}

    # Construct Prompt
    items_str = json.dumps(headers_by_sheet, ensure_ascii=False, indent=2)
    
    prompt = f"""
    你需要从以下Excel表头/内容中提取“清标项”，并根据性质分类。
    
    Excel数据结构 (Sheet名 -> 列名/项目名):
    {items_str}
    
    分类规则：
    1. **清标项根据性质不同分为两类**：
       - **总结类**：为某一块内容的总结、概括性描述或复杂的方案说明。例如：技术能力情况、产品设计、评审意见、技术方案、质量保证、供货计划、择优要素等。
       - **字段类**：为具体的数据点、数值、日期或简短的文本信息。例如：投标报价、服务期限、工期、注册资金、成立时间、注册地址、税率、项目名称等。
    
    2. **你需要将提取的项分为四大类**：
       - 技术清标项-总结类
       - 技术清标项-字段类
       - 商务清标项-总结类
       - 商务清标项-字段类

    注意：
    - "项目基本信息"、"商务基本信息"、"业绩"、"履约评价"、"投标报价" 下的大部分具体指标通常属于 **商务清标项-字段类**，但如果是复杂的描述则为总结类。
    - "技术" Sheet 下的内容通常属于 **技术清标项**，如果是“技术方案”、“质量保证”这种大块内容，属于**总结类**。
    - 明确示例：
      - "服务期限/工期/交货期" -> 商务清标项-字段类
      - "投标报价（含税）" -> 商务清标项-字段类
      - "技术方案" -> 技术清标项-总结类
    - 请忽略明显的占位符（如 "评审项1", "评审项2", "清标项1", "要素1", "……" 等），只提取有实质意义的名称。
    - 如果某个项在多个Sheet中出现，只需要保留一份。
    - 避免遗漏任何实质的清标项，尤其是在“项目基本信息”、“商务基本信息”、“业绩”、“履约评价”、“投标报价”等Sheet中。
    
    请输出纯 JSON 格式，不要包含 Markdown 格式标记（如 ```json ... ```）：
    {{
        "技术清标项-总结类": [],
        "技术清标项-字段类": [],
        "商务清标项-总结类": [],
        "商务清标项-字段类": []
    }}
    """

    # Call LLM
    result_json_str = call_llm(prompt, model_key, model_url, model_id)
    
    if result_json_str:
        # Clean up potential markdown formatting if LLM ignores instruction
        result_json_str = result_json_str.strip()
        if result_json_str.startswith("```json"):
            result_json_str = result_json_str[7:]
        if result_json_str.endswith("```"):
            result_json_str = result_json_str[:-3]
        
        try:
            return json.loads(result_json_str)
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON.")
            print("Response:", result_json_str)
            return {"error": "LLM response parse error"}
    else:
        return {"error": "LLM call failed"}
