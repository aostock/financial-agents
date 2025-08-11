import json
from types import SimpleNamespace
import re
import base64
from langchain_core.runnables import RunnableConfig


def dict_to_obj(dictionary):
    if isinstance(dictionary, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in dictionary.items()})
    elif isinstance(dictionary, list):
        return [dict_to_obj(item) for item in dictionary]
    else:
        return dictionary

def get_dict_json(s: str) -> dict:
    if s == "" or s is None or s == "{}":
        return {}
    try:
        start = s.find("{")
        s = s[start:]
        end = s.rfind("}") + 1
        if end == 0:
            r = s + "}"
        else:
            r = s[0:end]
        # remove \ and \n from string
        r = r.replace('\\', '').replace('\n', '')
        return json.loads(r)
    except Exception as e:
        print('get_json error:', s, e)
        return {}

def get_at_items(content: str):
    """
    从字符串content中解析出所有的 @[item_name] 的内容， item_name 是动态的字符内容，返回一个列表
    """
    items = re.findall(r'@(\w+)', content)
    return items

def get_array_json(s: str) -> list:
    if s == "" or s is None or s == "[]":
        return []
    try:
        start = s.rfind("[")
        s = s[start:]
        end = s.find("]") + 1
        if end == 0:
            r = s + "]"
        else:
            r = s[0:end]
        # remove \ and \n from string
        r = r.replace('\\', '').replace('\n', '')
        return json.loads(r)
    except Exception as e:
        print('get_json error:', s, e)
        return []



def get_latest_message_content(state):
    if state.get("messages") is None or len(state["messages"]) == 0:
        return ""
    last_message = state["messages"][-1]
    # Ensure content is a string even if it's stored as a list
    content = last_message.content
    if isinstance(content, list) and len(content) > 0:
        lastMessage = content[-1]
        if isinstance(lastMessage, dict):
            content = lastMessage.get('content', lastMessage.get('text', ''))
        elif hasattr(lastMessage, 'content'):
            content = lastMessage.content
        elif hasattr(lastMessage, 'text'):
            content = lastMessage.text
    return content

