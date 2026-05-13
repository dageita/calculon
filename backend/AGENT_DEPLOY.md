# LangChain / LangGraph Agent 部署说明

后端在启用 `OPENAI_API_KEY` 后提供 `/llm_training_calculator/agent/*` 接口；前端通过 Hash 路由 `#/agent` 进入聊天页。

## 环境变量

| 变量 | 含义 |
|------|------|
| `OPENAI_API_KEY` | 必填（启用对话）。未设置时 `/agent/chat` 返回 503。 |
| `OPENAI_BASE_URL` | 可选。兼容 OpenAI 协议的第三方网关（如自建 vLLM、Azure 等）。 |
| `AGENT_MODEL` | 默认 `gpt-4o-mini`。 |
| `AGENT_MAX_ITERATIONS` | 作为推理步数上界的参考；图中使用 `recursion_limit = max(25, AGENT_MAX_ITERATIONS * 4)`，防止工具循环过长。 |
| `AGENT_CHECKPOINT_SQLITE` | 可选。设为 SQLite 文件路径（如 `/var/lib/sim/agent_ckpt.sqlite`）时，会话状态通过 LangGraph checkpoint 持久化；多副本部署请使用共享存储上的同一文件或后续替换为 Postgres 等后端。未设置时使用进程内 `MemorySaver`（重启丢失）。 |
| `AGENT_LLM_TIMEOUT_SEC` | 单次 LLM 调用超时（秒），默认 `120`。 |
| `AGENT_LLM_MAX_RETRIES` | LLM 客户端重试次数，默认 `2`。 |
| `AGENT_MAX_CONCURRENT_INVOCATIONS` | 预留运维说明：当前未做全局并发闸门；建议在网关或反向代理层限制并发。 |

## HTTP 接口摘要

- `POST .../agent/sessions` — 创建 `thread_id`
- `POST .../agent/sessions/{thread_id}/reset` — 清空该线程 checkpoint（对话记忆）
- `GET .../agent/tools` — 工具清单（名称、描述、JSON Schema）
- `POST .../agent/chat` — 同步对话
- `POST .../agent/chat/stream` — SSE 流式

后端默认使用 `ReasoningAwareChatOpenAI`：对 DeepSeek 等「思考模式」接口，会把响应里的 `reasoning_content` 写入消息并在下一轮请求中原样回传，避免 400（`reasoning_content` 必须回传）错误。

## Docker 镜像

构建时需单独构建前端 `frontend/dist`，镜像内由 nginx 反代 API。为 Agent 传入上述环境变量请在编排文件（如 `docker run -e` / Compose）中声明；SQLite 路径需挂载可写卷。

## 前端本地代理

开发时设置 `API_PROXY_TARGET`（默认 `http://127.0.0.1:8000`），与 [`frontend/.umirc.ts`](../frontend/.umirc.ts) 中 dev proxy 一致。
