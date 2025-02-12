import os
import glob
from collections import defaultdict
import re

def compress_content(content):
    """压缩代码内容，移除不必要的空白但保持基本可读性"""
    # 移除连续的空行，保留单个空行
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # 移除行尾空白
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # 压缩多个空格为单个空格（除非在字符串中）
    lines = []
    in_string = False
    string_char = None
    
    for line in content.split('\n'):
        new_line = ''
        i = 0
        while i < len(line):
            char = line[i]
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                new_line += char
            elif not in_string and char.isspace():
                # 压缩连续空格
                while i + 1 < len(line) and line[i + 1].isspace():
                    i += 1
                new_line += ' '
            else:
                new_line += char
            i += 1
        lines.append(new_line.rstrip())
    
    return '\n'.join(lines)

def remove_comments(code, file_type):
    """
    移除代码中的注释，支持不同类型的文件格式
    """
    result = []
    i = 0
    code_len = len(code)
    
    in_string = False
    string_char = None
    in_multiline = False
    multiline_quote = None
    
    # JavaScript/CSS 风格的多行注释
    if file_type in ['.js', '.css']:
        while i < code_len:
            if i + 1 < code_len and code[i:i+2] == '/*':
                # 跳过多行注释
                i += 2
                while i + 1 < code_len and code[i:i+2] != '*/':
                    i += 1
                i += 2
                continue
            elif i + 1 < code_len and code[i:i+2] == '//':
                # 跳过单行注释
                while i < code_len and code[i] != '\n':
                    i += 1
                continue
            else:
                result.append(code[i])
                i += 1
    # HTML 注释
    elif file_type == '.html':
        while i < code_len:
            if i + 3 < code_len and code[i:i+4] == '<!--':
                # 跳过HTML注释
                i += 4
                while i + 2 < code_len and code[i:i+3] != '-->':
                    i += 1
                i += 3
                continue
            else:
                result.append(code[i])
                i += 1
    # Python 注释
    else:
        while i < code_len:
            char = code[i]
            
            # 处理转义字符
            if char == '\\' and i + 1 < code_len:
                if in_string:
                    result.extend([char, code[i + 1]])
                i += 2
                continue
            
            # 处理三引号
            if i + 2 < code_len and (code[i:i+3] == '"""' or code[i:i+3] == "'''"):
                quote = code[i:i+3]
                if in_string:
                    result.extend(quote)
                    i += 3
                    continue
                
                # 判断是否为文档字符串（赋值语句）
                j = i - 1
                while j >= 0 and code[j].isspace():
                    j -= 1
                is_assignment = False
                if j >= 0 and (code[j] == '=' or code[j:j+6] == 'return'):
                    is_assignment = True
                
                if not in_multiline and not is_assignment:
                    in_multiline = True
                    multiline_quote = quote
                    i += 3
                    continue
                elif in_multiline and quote == multiline_quote:
                    in_multiline = False
                    multiline_quote = None
                    i += 3
                    continue
            
            # 处理单引号和双引号
            elif char in '"\'':
                if not in_multiline:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                    result.append(char)
            
            # 处理单行注释
            elif char == '#' and not in_string and not in_multiline:
                while i < code_len and code[i] != '\n':
                    i += 1
                continue
            
            # 处理正常字符
            else:
                if not in_multiline:
                    result.append(char)
            
            i += 1
    
    # 清理结果，移除多余空行
    cleaned = ''.join(result)
    lines = cleaned.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def clear_file(file_path):
    """清空文件内容"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('')

def get_file_tree(start_path='.', exclude_dirs=['my-app', '__pycache__', '.git'], exclude_files=['generate_project_doc.py']):
    """生成文件树结构"""
    tree = []
    
    for root, dirs, files in os.walk(start_path):
        # 排除指定目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        level = root.replace(start_path, '').count(os.sep)
        indent = '│   ' * level
        
        # 添加目录
        if level > 0:
            tree.append(f'{indent[:-4]}├── {os.path.basename(root)}/')
        
        # 添加文件，支持更多文件类型
        for file in files:
            if file not in exclude_files and any(file.endswith(ext) for ext in ['.py', '.js', '.html', '.css', '.c', '.cpp', '.h', '.hpp', 'Dockerfile']):
                tree.append(f'{indent}├── {file}')
    
    return tree

def get_file_contents(start_path='.', exclude_dirs=['my-app', '__pycache__', '.git'], exclude_files=['generate_project_doc.py']):
    """获取所有支持的文件内容，并移除注释"""
    contents = []
    
    # 支持的文件类型及其语言标识
    file_types = {
        '.py': 'python',
        '.js': 'javascript',
        '.html': 'html',
        '.css': 'css',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        'Dockerfile': 'dockerfile'
    }
    
    # 文件类型分组 - 根据重构设计调整分组
    type_groups = {
        'server': ['.py'],  # 服务器相关代码
        'web': ['.js', '.html', '.css'],  # Web前端代码
        'services': ['.py'],  # 服务层代码
        'config': ['.yml', '.yaml', '.ini', 'Dockerfile'],  # 配置文件
        'docs': [],  # 文档文件不再记录
        'other': ['.c', '.cpp', '.h', '.hpp']  # 其他代码文件
    }
    
    # 优先级排序 - 确保文件按照重要性排序
    priority_paths = [
        'main.py',
        'server/app.py',
        'server/websocket.py',
        'services/agent_manager.py',
        'services/user_service.py',
        'web/templates/chat.html',
        'web/static/js/chat.js',
        'web/static/css/chat.css'
    ]
    
    # 首先处理优先级文件
    for priority_path in priority_paths:
        if os.path.exists(priority_path):
            file_ext = os.path.splitext(priority_path)[1].lower()
            if file_ext in file_types:
                try:
                    with open(priority_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if file_ext in ['.py', '.js', '.css', '.html']:
                            content = remove_comments(content, file_ext)
                        content = compress_content(content)
                        
                        # 确定文件分组
                        if 'server/' in priority_path:
                            group = 'server'
                        elif 'services/' in priority_path:
                            group = 'services'
                        elif 'web/' in priority_path:
                            group = 'web'
                        else:
                            group = next(
                                (group for group, exts in type_groups.items() 
                                 if file_ext in exts),
                                'other'
                            )
                        
                        contents.append((
                            priority_path,
                            content,
                            file_types.get(file_ext, 'text'),
                            group
                        ))
                except Exception as e:
                    print(f"处理优先级文件 {priority_path} 时出错: {str(e)}")
    
    # 然后处理其他文件
    for root, dirs, files in os.walk(start_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            # 跳过已处理的优先级文件
            if file_path in priority_paths:
                continue
                
            file_ext = os.path.splitext(file)[1].lower()
            if file == 'Dockerfile':
                file_ext = 'Dockerfile'
                
            if file not in exclude_files and (file_ext in file_types or file == 'Dockerfile'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if file_ext in ['.py', '.js', '.css', '.html']:
                            content = remove_comments(content, file_ext)
                        content = compress_content(content)
                        
                        # 确定文件分组
                        if 'server/' in file_path:
                            group = 'server'
                        elif 'services/' in file_path:
                            group = 'services'
                        elif 'web/' in file_path:
                            group = 'web'
                        else:
                            group = next(
                                (group for group, exts in type_groups.items() 
                                 if file_ext in exts),
                                'other'
                            )
                        
                        contents.append((
                            file_path,
                            content,
                            file_types.get(file_ext, 'text'),
                            group
                        ))
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
    
    return contents

def format_file_path(file_path):
    """格式化文件路径，使其更简洁"""
    parts = file_path.split(os.sep)
    if len(parts) <= 2:
        return file_path
    return os.path.join('...', *parts[-2:])

def write_compressed_markdown(f, content):
    """以压缩格式写入Markdown内容"""
    # 移除连续的空行
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    # 移除行尾空白
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    f.write(content)

def main():
    # 清空leicode.md文件
    clear_file('doc/leicode.md')
    
    # 生成文件树
    tree = get_file_tree()
    
    # 获取文件内容
    file_contents = get_file_contents()
    
    # 按分组整理文件
    grouped_contents = defaultdict(list)
    for file_path, content, language, group in file_contents:
        grouped_contents[group].append((file_path, content, language))
    
    # 写入文档
    with open('doc/leicode.md', 'w', encoding='utf-8') as f:
        # 写入文件树
        write_compressed_markdown(f, '# 项目目录结构\n```\n.')
        for line in tree:
            write_compressed_markdown(f, f'\n{line}')
        write_compressed_markdown(f, '\n```\n')
        
        # 按分组写入文件内容，调整分组顺序和标题
        group_titles = {
            'server': 'Web服务器层',
            'services': '服务层',
            'web': '前端界面',
            'config': '配置文件',
            'docs': '文档文件',
            'other': '其他文件'
        }
        
        # 指定分组顺序
        group_order = ['server', 'services', 'web', 'config', 'docs', 'other']
        
        for group in group_order:
            if group in grouped_contents:
                write_compressed_markdown(f, f'\n# {group_titles[group]}\n')
                for file_path, content, language in grouped_contents[group]:
                    short_path = format_file_path(file_path)
                    write_compressed_markdown(f, f'\n## {short_path}\n```{language}\n{content}```\n')
                    write_compressed_markdown(f, '_' * 30 + '\n')

if __name__ == '__main__':
    main() 