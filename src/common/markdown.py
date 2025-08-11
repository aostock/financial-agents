import json
import uuid


def ticker_select(data:dict):
    if data.get('_id_') is None:
        data['_id_'] = str(uuid.uuid4())
    return f"""```TickerSelect
{json.dumps(data)}
```"""

def analysis_data(data:dict):
    if data.get('_id_') is None:
        data['_id_'] = str(uuid.uuid4())
    return f"""```AnalysisData
{json.dumps(data)}
```"""

def from_dict(data:dict):
    """
    Convert a dict to a markdown string.
    """
    markdown = ''
    for key, value in data.items():
        if key == '_id_':
            continue
        markdown += f'## {key}\n\n'
        markdown += f'{value}\n\n'
    return markdown



def dict_to_table(data:dict, keys:list | None = None):
    """
    Convert a dict to a markdown table.
    """
    markdown = ''
    # add key and value header
    markdown += '| Key | Value |\n'
    markdown += '| --- | --- |\n'
    for key, value in data.items():
        if key == '_id_':
            continue
        if keys is not None and key not in keys:
            continue
        markdown += f'| {key} | {value} |\n'
    return markdown

def list_dict_to_table(data:list, keys:list | None = None):
    """
    Convert a list of dict to a markdown table, header is the keys of the dict.
    """
    markdown = ''

    # 第一行是header
    if keys is None:
        keys = data[0].keys()
    markdown += '| ' + ' | '.join(keys) + ' |\n'
    # 第二行是分隔线
    markdown += '| ' + ' | '.join(['---' for _ in keys]) + ' |\n'
    # 第三行开始是数据
    for item in data:
        markdown += '| ' + ' | '.join([str(item.get(key, '')) for key in keys]) + ' |\n'
    return markdown


def list_str_to_sequence(data:list):
    """
    Convert a list of string to a markdown sequence.
    """
    markdown = ''
    for item in data:
        markdown += f'1. {item}\n'
    return markdown


def to_h1(data:str):
    """
    Convert a string to a markdown header.
    """
    return f'# {data}\n\n'

def to_h2(data:str):
    """
    Convert a string to a markdown header.
    """
    return f'## {data}\n\n'

def to_h3(data:str):
    """
    Convert a string to a markdown header.
    """
    return f'### {data}\n\n'
