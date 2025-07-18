def read_text_file(file_path: str) -> str:
    """
    读取文本文件内容。
    :param file_path: 文件路径
    :return: 文件内容字符串
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read() 