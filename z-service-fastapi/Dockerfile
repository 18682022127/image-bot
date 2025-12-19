# 使用轻量级 Python 镜像
FROM docker.m.daocloud.io/python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
# 使用清华源进行安装，速度快且稳
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 复制你的代码
COPY main.py .
COPY workflow_txt2img.json .
COPY workflow_img2img.json .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]