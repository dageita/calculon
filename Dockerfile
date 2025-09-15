# 使用多阶段构建
FROM ubuntu:22.04 as builder

# 安装必要的依赖
RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    apt-get install -y python3 python3-pip && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    apt-get install -y nlohmann-json3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制C++项目代码（需要先复制到构建上下文）
COPY LLMFlowSimulator/ ./LLMFlowSimulator/
WORKDIR /app/LLMFlowSimulator

# 编译C++项目生成so文件
RUN make so

# 复制so文件到calculon目录
WORKDIR /app
COPY . ./calculon/
RUN cp ./LLMFlowSimulator/libpycallclass.so ./calculon/

# 第二阶段：运行环境
FROM ubuntu:22.04

# 安装运行依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 从构建阶段复制整个calculon项目（包含so文件）
COPY --from=builder /app/calculon/ ./calculon/

# 复制frontend/dist到/app/dist
COPY frontend/dist /app/dist

# 安装backend的额外依赖
WORKDIR /app/calculon/backend
RUN pip3 install -r requirements.txt

# 安装calculon项目
WORKDIR /app/calculon
RUN pip3 install .
# 配置nginx
COPY frontend/default.conf /etc/nginx/nginx.conf
RUN mkdir -p /var/log/nginx && \
    nginx -t

# 暴露端口（前端nginx端口80，后端Python端口8000）
EXPOSE 80 8000

# 创建启动脚本
RUN echo '#!/bin/bash\n\
set -e\n\
echo "启动服务..."\n\
\n\
# 检查nginx配置\n\
echo "检查nginx配置..."\n\
nginx -t\n\
\n\
# 启动nginx\n\
echo "启动nginx..."\n\
nginx\n\
\n\
# 检查nginx是否启动成功\n\
sleep 2\n\
if ! pgrep nginx > /dev/null; then\n\
    echo "nginx启动失败！"\n\
    exit 1\n\
fi\n\
echo "nginx启动成功"\n\
\n\
# 启动Python后端服务\n\
echo "启动Python后端服务..."\n\
cd /app/calculon && python3 backend/main.py &\n\
\n\
# 等待所有后台进程\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# 启动服务
CMD ["/app/start.sh"]