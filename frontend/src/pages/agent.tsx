import React, { FC, useCallback, useEffect, useMemo, useState } from 'react';
import {
  Button,
  Collapse,
  Typography,
  Alert,
  Select,
  InputNumber,
  Checkbox,
  Row,
  Col,
  message,
} from 'antd';
import { useHistory } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { service_base_url } from '@/utils/constant';
import { getGpuList, getModelList, getNetWork } from '@/services';
import { AI_Prompt } from '@/components/ui/animated-ai-input';
import TLEventChart from '@/components/timelines/timeline-events';
const { Paragraph, Text } = Typography;

interface IntermediateStep {
  tool?: string | null;
  tool_input?: unknown;
  observation?: string;
  /** 后端对 run_calculate / run_optimal 全量 JSON 解析后附加，与 Guide 模式 result.timeline_events 同源 */
  timeline_events?: unknown[];
  simulator_summary?: Record<string, unknown>;
}

interface LogMessage {
  role: string;
  content: string;
  steps?: IntermediateStep[];
}

/** 从单次回复的工具步骤中取 timeline_events（优先后端附加字段，否则尝试解析 observation JSON）。 */
function extractTimelineEventsFromSteps(steps?: IntermediateStep[]): unknown[] {
  if (!steps?.length) return [];
  for (let j = steps.length - 1; j >= 0; j -= 1) {
    const s = steps[j] as IntermediateStep & { timeline_events?: unknown[] };
    const direct = s.timeline_events;
    if (Array.isArray(direct) && direct.length) return direct;
    const tool = (s.tool || '').toString();
    if ((tool === 'run_calculate' || tool === 'run_optimal') && typeof s.observation === 'string') {
      const raw = s.observation.trim();
      const cut = raw.indexOf('\n... [truncated');
      const jsonPart = cut >= 0 ? raw.slice(0, cut) : raw;
      try {
        const obj = JSON.parse(jsonPart) as { status?: string; timeline_events?: unknown[] };
        if (obj?.status === 'error') continue;
        const te = obj.timeline_events;
        if (Array.isArray(te) && te.length) return te;
      } catch {
        /* 截断或非 JSON */
      }
    }
  }
  return [];
}

function pickLatestTimelineEventsFromLog(messages: LogMessage[]): unknown[] {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const m = messages[i];
    if (m.role !== 'assistant') continue;
    const ev = extractTimelineEventsFromSteps(m.steps);
    if (ev.length) return ev;
  }
  return [];
}

interface ChatResponse {
  reply: string;
  intermediate_steps?: IntermediateStep[];
  error?: string;
}

interface GpuRow {
  name?: string;
}

interface ModelRow {
  name?: string;
}

function parseSseBlocks(buffer: string): { events: unknown[]; rest: string } {
  const events: unknown[] = [];
  const parts = buffer.split('\n\n');
  const rest = parts.pop() ?? '';
  for (const block of parts) {
    const line = block.trim();
    if (!line.startsWith('data:')) continue;
    const payload = line.slice(5).trim();
    try {
      events.push(JSON.parse(payload));
    } catch {
      /* ignore malformed chunk */
    }
  }
  return { events, rest };
}

const AgentPage: FC = () => {
  const { t } = useTranslation();
  const history = useHistory();
  const [threadId, setThreadId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamMode, setStreamMode] = useState(false);
  const [log, setLog] = useState<LogMessage[]>([]);
  const [sessionError, setSessionError] = useState<string | null>(null);

  const [gpuOptions, setGpuOptions] = useState<{ label: string; value: string }[]>([]);
  const [modelOptions, setModelOptions] = useState<{ label: string; value: string }[]>([]);
  const [topologyOptions, setTopologyOptions] = useState<string[]>([]);

  const [formGpu, setFormGpu] = useState<string | undefined>();
  const [formModel, setFormModel] = useState<string | undefined>();
  const [formBatch, setFormBatch] = useState<number>(32);
  const [formMicro, setFormMicro] = useState<number>(8);
  const [formTp, setFormTp] = useState<number>(1);
  const [formPp, setFormPp] = useState<number>(1);
  const [formDp, setFormDp] = useState<number>(1);
  const [formNumGpus, setFormNumGpus] = useState<number>(1);
  const [formBw, setFormBw] = useState<number>(0);
  const [formTopo, setFormTopo] = useState<string>('Single machine');
  const [formDatatype, setFormDatatype] = useState<string>('float16');

  const latestTimelineEvents = useMemo(() => pickLatestTimelineEventsFromLog(log), [log]);

  const agentTrainingFormError = useMemo(() => {
    if (formTp < 1 || formPp < 1 || formDp < 1 || formNumGpus < 1 || formMicro < 1 || formBatch < 1) {
      return 'TP、PP、DP、GPU 数量、batch、microbatch 须均为至少 1 的正整数。';
    }
    const parProd = formTp * formPp * formDp;
    if (parProd !== formNumGpus) {
      return `并行度乘积 TP×PP×DP（${parProd}）必须等于 GPU 数量（${formNumGpus}）。`;
    }
    const dpMicro = formDp * formMicro;
    if (formBatch % dpMicro !== 0) {
      return `batch_size（${formBatch}）必须是 DP×microbatch_size（${formDp}×${formMicro}=${dpMicro}）的整数倍。`;
    }
    return null;
  }, [formTp, formPp, formDp, formNumGpus, formMicro, formBatch]);

  const base = `${service_base_url}/llm_training_calculator/agent`;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [gpus, models, net] = await Promise.all([getGpuList(), getModelList(), getNetWork()]);
        if (cancelled) return;
        const glist = Array.isArray(gpus) ? (gpus as GpuRow[]) : [];
        const mlist = Array.isArray(models) ? (models as ModelRow[]) : [];
        setGpuOptions(glist.filter((g) => g?.name).map((g) => ({ label: g.name!, value: g.name! })));
        setModelOptions(mlist.filter((m) => m?.name).map((m) => ({ label: m.name!, value: m.name! })));
        const tops = (net as { network_topology?: string[] })?.network_topology;
        if (tops?.length) {
          setTopologyOptions(tops);
          setFormTopo((t) => (tops.includes(t) ? t : tops[0]));
        }
      } catch {
        /* catalog optional for chat */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const ensureSession = useCallback(async () => {
    setSessionError(null);
    const res = await fetch(`${base}/sessions`, { method: 'POST' });
    if (!res.ok) {
      throw new Error(`sessions HTTP ${res.status}`);
    }
    const data = await res.json();
    setThreadId(data.thread_id);
    return data.thread_id as string;
  }, [base]);

  useEffect(() => {
    ensureSession().catch((e) => setSessionError(String(e)));
  }, [ensureSession]);

  const startNewSession = async () => {
    setSessionError(null);
    setLog([]);
    try {
      await ensureSession();
    } catch (e) {
      setSessionError(String(e));
    }
  };

  const resetSessionMemory = async () => {
    if (!threadId) return;
    setSessionError(null);
    try {
      const res = await fetch(`${base}/sessions/${encodeURIComponent(threadId)}/reset`, {
        method: 'POST',
      });
      if (!res.ok) {
        throw new Error(`reset HTTP ${res.status}`);
      }
      setLog([]);
    } catch (e) {
      setSessionError(String(e));
    }
  };

  const fillPromptFromForm = () => {
    if (agentTrainingFormError) {
      message.error(agentTrainingFormError);
      return;
    }
    const tp = formTp;
    const pp = formPp;
    const dp = formDp;
    const parProd = tp * pp * dp;
    const gpu = formGpu || '（请在表单选择 GPU）';
    const model = formModel || '（请在表单选择模型）';
    const text = [
      `请用模拟器估算下面配置下一次迭代（batch）的训练耗时：`,
      `- GPU: ${gpu}`,
      `- 模型: ${model}`,
      `- num_procs（GPU 总数）=${formNumGpus}，须与 TP×PP×DP 一致`,
      `- 单机/网络拓扑: ${formTopo}，机间带宽 ${formBw} Gbps`,
      `- 并行: tensor_par=${tp}, pipeline_par=${pp}, data_par=${dp}（TP×PP×DP=${parProd}）`,
      `- batch_size=${formBatch}, microbatch_size=${formMicro}（须满足 batch_size 可被 DP×microbatch_size 整除）`,
      `- datatype=${formDatatype}（Calculon 使用 float16 / float32 / bfloat16）`,
      `请先必要时调用 list_simulator_catalog 校验名称，再调用 run_calculate；参数中传入 num_procs=${formNumGpus}、datatype=${formDatatype} 及上述并行与 batch 字段。`,
    ].join('\n');
    setInput(text);
  };

  const sendSync = async (tid: string, msg: string) => {
    const res = await fetch(`${base}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: tid, message: msg }),
    });
    const data: ChatResponse = await res.json();
    if (!res.ok) {
      setSessionError(data.error || `chat HTTP ${res.status}`);
      setLog((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.error || `请求失败 (${res.status})`,
          steps: data.intermediate_steps,
        },
      ]);
      return;
    }
    const replyText = data.error
      ? `错误: ${data.error}\n${data.reply || ''}`.trim()
      : data.reply;
    setLog((prev) => [...prev, { role: 'assistant', content: replyText, steps: data.intermediate_steps }]);
  };

  const sendStream = async (tid: string, msg: string) => {
    const res = await fetch(`${base}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: tid, message: msg }),
    });
    if (!res.ok) {
      const errText = await res.text();
      setSessionError(`stream HTTP ${res.status}: ${errText}`);
      setLog((prev) => [...prev, { role: 'assistant', content: `流式请求失败 (${res.status})` }]);
      return;
    }
    const reader = res.body?.getReader();
    if (!reader) {
      setSessionError('无法读取响应流');
      return;
    }

    let assistantContent = '';
    let steps: IntermediateStep[] = [];
    let buf = '';
    const dec = new TextDecoder();

    setLog((prev) => [...prev, { role: 'assistant', content: '', steps: [] }]);

    const flushEvents = (chunk: string) => {
      buf += chunk;
      const { events, rest } = parseSseBlocks(buf);
      buf = rest;
      for (const ev of events) {
        if (!ev || typeof ev !== 'object') continue;
        const o = ev as Record<string, unknown>;
        if (o.type === 'token' && typeof o.text === 'string') {
          assistantContent += o.text;
        }
        if (o.type === 'result') {
          if (typeof o.reply === 'string') assistantContent = o.reply;
          if (Array.isArray(o.intermediate_steps)) steps = o.intermediate_steps as IntermediateStep[];
        }
        if (o.type === 'error' && typeof o.error === 'string') {
          assistantContent += `\n[错误] ${o.error}`;
        }
      }
      setLog((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === 'assistant') {
          next[next.length - 1] = { ...last, content: assistantContent, steps };
        }
        return next;
      });
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        flushEvents(dec.decode(value, { stream: true }));
      }
      flushEvents(dec.decode());
    } catch (e) {
      setSessionError(String(e));
    }
  };

  const send = async () => {
    const msg = input.trim();
    if (!msg) return;
    if (agentTrainingFormError) {
      message.error(agentTrainingFormError);
      return;
    }
    let tid = threadId;
    if (!tid) {
      try {
        tid = await ensureSession();
      } catch (e) {
        setSessionError(String(e));
        return;
      }
    }
    setLoading(true);
    setInput('');
    setLog((prev) => [...prev, { role: 'user', content: msg }]);
    try {
      if (streamMode) {
        await sendStream(tid, msg);
      } else {
        await sendSync(tid, msg);
      }
    } catch (e) {
      setSessionError(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="agent-sim-root">
      <div className="agent-sim-bg" aria-hidden />
      <div className="agent-sim-grid" aria-hidden />
      <div className="agent-sim-content">
        <header className="agent-sim-header">
          <div>
            <p className="agent-sim-eyebrow">LangGraph · Simulator</p>
            <h1 className="agent-sim-title">
              训练模拟 <span>Agent</span>
            </h1>
            <p className="agent-sim-lede">
              用自然语言驱动模拟器：先核对目录，再估算迭代耗时。会话与工具轨迹均保留在下方时间线。
            </p>
          </div>
          <nav className="agent-sim-nav">
            <Button type="link" onClick={() => history.push('/guide')}>
              {t('guide mode')}
            </Button>
            <Button type="link" onClick={() => history.push('/optimal')}>
              {t('optimal mode')}
            </Button>
            <span className="agent-sim-nav-active" style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>
              {t('agent mode')}
            </span>
          </nav>
        </header>

        <div className="agent-sim-meta">
          <span className="agent-sim-chip">
            <strong>SESSION</strong> {threadId || '…'}
          </span>
          <span className="agent-sim-chip">
            需配置 <strong>OPENAI_API_KEY</strong>
          </span>
        </div>

        {sessionError && <Alert type="error" message={sessionError} showIcon style={{ marginBottom: 16 }} />}

        <div className="agent-sim-panel">
          <Collapse defaultActiveKey={['params']}>
            <Collapse.Panel header="模拟参数（填入提问）" key="params">
              <Row gutter={[12, 12]}>
                <Col xs={24} sm={12}>
                  <Text type="secondary">GPU</Text>
                  <Select
                    style={{ width: '100%', marginTop: 4 }}
                    placeholder="选择 GPU"
                    options={gpuOptions}
                    value={formGpu}
                    onChange={setFormGpu}
                    allowClear
                    showSearch
                    optionFilterProp="label"
                  />
                </Col>
                <Col xs={24} sm={12}>
                  <Text type="secondary">模型</Text>
                  <Select
                    style={{ width: '100%', marginTop: 4 }}
                    placeholder="选择模型"
                    options={modelOptions}
                    value={formModel}
                    onChange={setFormModel}
                    allowClear
                    showSearch
                    optionFilterProp="label"
                  />
                </Col>
                <Col xs={8} sm={6}>
                  <Text type="secondary">batch</Text>
                  <InputNumber style={{ width: '100%', marginTop: 4 }} min={1} value={formBatch} onChange={(v) => setFormBatch(v ?? 1)} />
                </Col>
                <Col xs={8} sm={6}>
                  <Text type="secondary">microbatch</Text>
                  <InputNumber style={{ width: '100%', marginTop: 4 }} min={1} value={formMicro} onChange={(v) => setFormMicro(v ?? 1)} />
                </Col>
                <Col xs={8} sm={4}>
                  <Text type="secondary">TP</Text>
                  <InputNumber style={{ width: '100%', marginTop: 4 }} min={1} value={formTp} onChange={(v) => setFormTp(v ?? 1)} />
                </Col>
                <Col xs={8} sm={4}>
                  <Text type="secondary">PP</Text>
                  <InputNumber style={{ width: '100%', marginTop: 4 }} min={1} value={formPp} onChange={(v) => setFormPp(v ?? 1)} />
                </Col>
                <Col xs={8} sm={4}>
                  <Text type="secondary">DP</Text>
                  <InputNumber style={{ width: '100%', marginTop: 4 }} min={1} value={formDp} onChange={(v) => setFormDp(v ?? 1)} />
                </Col>
                <Col xs={24} sm={8}>
                  <Text type="secondary">GPU 数量（num_procs）</Text>
                  <InputNumber style={{ width: '100%', marginTop: 4 }} min={1} value={formNumGpus} onChange={(v) => setFormNumGpus(v ?? 1)} />
                  <Text type="secondary" style={{ display: 'block', marginTop: 4, fontSize: 12 }}>
                    须满足 TP×PP×DP = GPU 数量；batch 须为 DP×microbatch 的整数倍。
                  </Text>
                </Col>
                <Col xs={24} sm={12}>
                  <Text type="secondary">机间带宽 (Gbps)</Text>
                  <InputNumber style={{ width: '100%', marginTop: 4 }} min={0} value={formBw} onChange={(v) => setFormBw(v ?? 0)} />
                </Col>
                <Col xs={24} sm={12}>
                  <Text type="secondary">网络拓扑</Text>
                  <Select
                    style={{ width: '100%', marginTop: 4 }}
                    options={topologyOptions.map((top) => ({ label: top, value: top }))}
                    value={formTopo}
                    onChange={(v) => setFormTopo(v)}
                  />
                </Col>
                <Col xs={24} sm={12}>
                  <Text type="secondary">datatype</Text>
                  <Select
                    style={{ width: '100%', marginTop: 4 }}
                    value={formDatatype}
                    onChange={setFormDatatype}
                    options={[
                      { label: 'float16（FP16）', value: 'float16' },
                      { label: 'float32（FP32）', value: 'float32' },
                      { label: 'bfloat16（BF16）', value: 'bfloat16' },
                    ]}
                  />
                </Col>
              </Row>
              <Button style={{ marginTop: 12 }} onClick={fillPromptFromForm}>
                填入提问
              </Button>
            </Collapse.Panel>
          </Collapse>
        </div>

        {log.map((m, i) => (
          <div
            key={i}
            className={`agent-sim-msg ${m.role === 'user' ? 'agent-sim-msg-user' : 'agent-sim-msg-assistant'}`}
            style={{ animationDelay: `${Math.min(i, 12) * 0.045}s` }}
          >
            <div className="agent-sim-msg-label">{m.role === 'user' ? 'Operator' : 'Assistant'}</div>
            <Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 8, fontFamily: "'Literata', Georgia, serif" }}>
              {m.content}
            </Paragraph>
            {m.steps && m.steps.length > 0 && (
              <Collapse ghost>
                <Collapse.Panel header="工具调用" key="steps">
                  {m.steps.map((s, j) => (
                    <div key={j} style={{ marginBottom: 8 }}>
                      <Text code style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                        {s.tool}
                      </Text>
                      <pre style={{ fontSize: 12, maxHeight: 200, overflow: 'auto' }}>
                        {JSON.stringify(s.tool_input, null, 2)}
                      </pre>
                      <pre style={{ fontSize: 12, maxHeight: 240, overflow: 'auto' }}>{s.observation}</pre>
                    </div>
                  ))}
                </Collapse.Panel>
              </Collapse>
            )}
          </div>
        ))}

        {latestTimelineEvents.length > 0 && (
          <div className="agent-sim-panel" style={{ marginBottom: 16 }}>
            <Collapse defaultActiveKey={['timeline']}>
              <Collapse.Panel header="Timeline" key="timeline">
                <Text type="secondary" style={{ display: 'block', marginBottom: 8, fontSize: 12 }}>
                  与 Guide 模式相同组件：展示当前会话中最近一次模拟返回的 rank / 事件时间线。
                </Text>
                <div
                  style={{
                    borderRadius: 8,
                    overflow: 'hidden',
                    background: '#fafafa',
                    border: '1px solid rgba(255,255,255,0.12)',
                  }}
                >
                  <TLEventChart result={latestTimelineEvents} embedded />
                </div>
              </Collapse.Panel>
            </Collapse>
          </div>
        )}

        <div className="agent-sim-panel agent-sim-input-mount">
          <AI_Prompt
            value={input}
            onValueChange={setInput}
            onSubmit={() => {
              void send();
            }}
            disabled={loading}
            placeholder="例如：用 H100_80G_SXM 跑 GPT-3 Small，单机，num_procs=2，tp=2 pp=1 dp=1，batch 32 microbatch 8，datatype=float16，算一次迭代时间"
          />
        </div>

        <div className="agent-sim-toolbar">
          <Checkbox checked={streamMode} onChange={(e) => setStreamMode(e.target.checked)}>
            流式输出（SSE）
          </Checkbox>
          <Button onClick={startNewSession} disabled={loading}>
            新会话
          </Button>
          <Button onClick={resetSessionMemory} disabled={loading || !threadId}>
            清空记忆
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AgentPage;
