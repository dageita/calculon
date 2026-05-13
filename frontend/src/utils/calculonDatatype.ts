/** 与后端 Calculon / run_calculate 一致的训练 dtype 字符串。 */
export type CalculonDatatype = 'float16' | 'float32' | 'bfloat16';

const ALIASES: Record<string, CalculonDatatype> = {
  float16: 'float16',
  fp16: 'float16',
  half: 'float16',
  float32: 'float32',
  fp32: 'float32',
  bfloat16: 'bfloat16',
  bf16: 'bfloat16',
  bp16: 'bfloat16',
};

/**
 * 将 API 或旧版 UI 中的 dtype 名称规范为 float16 | float32 | bfloat16。
 * 无法识别时返回 null。
 */
export function normalizeCalculonDatatype(raw: string | undefined | null): CalculonDatatype | null {
  if (raw == null || String(raw).trim() === '') return null;
  const key = String(raw).trim().toLowerCase();
  return ALIASES[key] ?? null;
}

/** 将 GPU profile 返回的 keys 规范为后端可接受的 value（无法识别的项丢弃）。 */
export function normalizeDatatypeList(raw: string[]): { label: string; value: CalculonDatatype }[] {
  const seen = new Set<CalculonDatatype>();
  const out: { label: string; value: CalculonDatatype }[] = [];
  for (const item of raw) {
    const v = normalizeCalculonDatatype(item);
    if (!v || seen.has(v)) continue;
    seen.add(v);
    const short =
      v === 'float16' ? 'FP16' : v === 'float32' ? 'FP32' : 'BF16';
    out.push({ label: `${v}（${short}）`, value: v });
  }
  return out;
}
