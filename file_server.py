"""
简单的文件上传/下载服务器
监听 0.0.0.0:8888，文件存储在 ./data 目录
"""

import os
from flask import Flask, request, send_from_directory, jsonify

app = Flask(__name__)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 确保 data 目录存在
os.makedirs(DATA_DIR, exist_ok=True)


@app.route('/')
def index():
    """列出所有文件"""
    files = os.listdir(DATA_DIR)
    links = ''.join(f'<li><a href="/download/{f}">{f}</a> <button onclick="del(\'{f}\')">删除</button></li>' for f in files)
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>文件服务器</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
        #dropzone {{ border: 3px dashed #ccc; padding: 50px; text-align: center; margin: 20px 0; cursor: pointer; }}
        #dropzone.dragover {{ border-color: #4CAF50; background: #e8f5e9; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
        a {{ color: #1976D2; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        #status {{ margin-top: 10px; color: #666; }}
        .progress {{ width: 100%; height: 20px; background: #eee; border-radius: 10px; margin: 10px 0; display: none; }}
        .progress-bar {{ height: 100%; background: #4CAF50; border-radius: 10px; transition: width 0.2s; }}
        .progress-text {{ text-align: center; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>文件列表</h1>
    <div id="dropzone">拖拽文件到此处上传，或点击选择文件</div>
    <input type="file" id="fileInput" style="display:none" multiple>
    <div class="progress" id="progress"><div class="progress-bar" id="progressBar"></div></div>
    <div id="status"></div>
    <ul>{links}</ul>
    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const status = document.getElementById('status');
        const progress = document.getElementById('progress');
        const progressBar = document.getElementById('progressBar');

        dropzone.onclick = () => fileInput.click();
        dropzone.ondragover = e => {{ e.preventDefault(); dropzone.classList.add('dragover'); }};
        dropzone.ondragleave = () => dropzone.classList.remove('dragover');
        dropzone.ondrop = e => {{
            e.preventDefault();
            dropzone.classList.remove('dragover');
            uploadFiles(e.dataTransfer.files);
        }};
        fileInput.onchange = () => uploadFiles(fileInput.files);

        function del(name) {{
            if (confirm('确定删除 ' + name + '?')) {{
                fetch('/delete/' + name, {{ method: 'DELETE' }})
                    .then(() => location.reload());
            }}
        }}

        function uploadFiles(files) {{
            for (const file of files) {{
                const formData = new FormData();
                formData.append('file', file);

                const xhr = new XMLHttpRequest();
                progress.style.display = 'block';
                progressBar.style.width = '0%';

                xhr.upload.onprogress = e => {{
                    if (e.lengthComputable) {{
                        const pct = Math.round((e.loaded / e.total) * 100);
                        progressBar.style.width = pct + '%';
                        status.textContent = file.name + ' - ' + pct + '%';
                    }}
                }};
                xhr.onload = () => {{
                    if (xhr.status === 200) {{
                        status.textContent = '上传成功: ' + file.name;
                        setTimeout(() => location.reload(), 500);
                    }} else {{
                        status.textContent = '上传失败';
                    }}
                }};
                xhr.onerror = () => status.textContent = '上传失败';
                xhr.open('POST', '/upload');
                xhr.send(formData);
            }}
        }}
    </script>
</body>
</html>'''


@app.route('/upload', methods=['POST'])
def upload():
    """上传文件: curl -F "file=@xxx.zip" http://host:8888/upload"""
    if 'file' not in request.files:
        return jsonify(error='No file provided'), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify(error='Empty filename'), 400

    filepath = os.path.join(DATA_DIR, f.filename)
    f.save(filepath)
    return jsonify(message='OK', filename=f.filename)


@app.route('/download/<filename>')
def download(filename):
    """下载文件: curl -O http://host:8888/download/xxx.zip"""
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


@app.route('/delete/<filename>', methods=['DELETE'])
def delete(filename):
    """删除文件: curl -X DELETE http://host:8888/delete/xxx.zip"""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify(message='Deleted')
    return jsonify(error='Not found'), 404


if __name__ == '__main__':
    print(f"文件存储目录: {DATA_DIR}")
    print("上传: curl -F \"file=@xxx.zip\" http://host:8888/upload")
    print("下载: curl -O http://host:8888/download/xxx.zip")
    print("列表: curl http://host:8888/")
    app.run(host='0.0.0.0', port=8888)
