import re
from typing import List

def split_text(text: str, mode: str = 'paragraph') -> List[str]:
    """
    按段落、句子或章节切分文本，保证完整性。
    :param text: 原始文本
    :param mode: 'paragraph'、'sentence' 或 'section'
    :return: 切分后的文本单元列表
    """
    if mode == 'paragraph':
        # 按空行或换行切分段落
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        return paragraphs
    elif mode == 'sentence':
        # 按句号、问号、感叹号切分句子，保留标点
        sentences = re.split(r'(?<=[。！？!?.])', text)
        return [s.strip() for s in sentences if s.strip()]
    elif mode == 'section':
        # 按常见章节标题切分，如“第X章”、“第X节”、“Chapter X”、“Section X”
        pattern = r'((?:第[一二三四五六七八九十百千0-9]+[章节])|(?:Chapter\s+\d+)|(?:Section\s+\d+))'
        splits = re.split(pattern, text)
        sections = []
        i = 1
        while i < len(splits):
            title = splits[i].strip()
            content = splits[i+1].strip() if (i+1)<len(splits) else ''
            sections.append(f'{title}\n{content}')
            i += 2
        # 若开头有前言等无章节标题内容
        if splits[0].strip():
            sections = [splits[0].strip()] + sections
        return [s for s in sections if s.strip()]
    else:
        raise ValueError('mode 仅支持 paragraph、sentence 或 section') 