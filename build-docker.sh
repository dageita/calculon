#!/bin/bash

# Docker构建脚本
# 用于构建包含C++、Python后端和前端三个项目的Docker镜像

echo "开始构建Docker镜像..."

# 检查C++项目是否存在
if [ ! -d "../LLMFlowSimulator" ]; then
    echo "错误：找不到C++项目目录 ../LLMFlowSimulator"
    echo "请确保C++项目位于当前项目的上一级并行目录"
    exit 1
fi

# 检查前端dist目录是否存在
if [ ! -d "frontend/dist" ]; then
    echo "警告：找不到frontend/dist目录"
    echo "请先构建前端项目："
    echo "cd frontend && npm install && npm run build"
    echo "或者确保frontend/dist目录存在"
    exit 1
fi

# 创建临时目录来复制C++项目
echo "复制C++项目到构建上下文..."
cp -r ../LLMFlowSimulator ./LLMFlowSimulator

# 构建Docker镜像
echo "构建Docker镜像..."
if docker build -f Dockerfile -t simulator .; then
    echo "✅ 优化版本构建成功！"
else
    echo "❌ 所有构建都失败了！"
    exit 1
fi

# 清理临时文件
echo "清理临时文件..."
rm -rf ./LLMFlowSimulator

echo "Docker镜像构建完成！"
echo ""
echo "运行方式1（自动启动服务）："
echo "docker run -d -p 3000:80 -p 8001:8000 --name sim simulator"
echo ""
echo "访问方式："
echo "- 前端：http://localhost:3000"
echo "- 后端API：http://localhost:3000/llm_training_calculator"
echo "- 直接后端：http://localhost:8001"
echo ""