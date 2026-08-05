import { request } from 'umi';
import { message } from 'antd';

function extractErrorMessage(error: any): string {
  const detail =
    error?.response?.data?.detail ??
    error?.data?.detail ??
    error?.response?.data?.message ??
    error?.message;
  if (Array.isArray(detail)) {
    // FastAPI / Pydantic validation errors
    return detail
      .map((item: any) => {
        const loc = Array.isArray(item?.loc) ? item.loc.join('.') : '';
        return loc ? `${loc}: ${item?.msg || item}` : (item?.msg || String(item));
      })
      .join('; ');
  }
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (detail && typeof detail === 'object') {
    return JSON.stringify(detail);
  }
  return 'Request failed';
}

function normalizeResult(res: any): any {
  if (!res || typeof res !== 'object') {
    return res;
  }
  // Backend business error: { status: "error", error: "..." }
  if (res.status === 'error' && res.error) {
    const errMsg = typeof res.error === 'string' ? res.error : String(res.error);
    message.error(errMsg);
    return { status: 'error', error: errMsg };
  }
  // FastAPI HTTPException body sometimes returned as 200/proxy body
  if (typeof res.detail === 'string' && res.detail && !res.memory_usage) {
    message.error(res.detail);
    return { status: 'error', error: res.detail };
  }
  // Soft warnings (e.g. mem over capacity) — keep full result for the UI
  if (res.warning && typeof res.warning === 'string') {
    message.warning(res.warning);
  }
  return res;
}

async function MyRequest<T>(url: string, options: any): Promise<T> {
  try {
    const res: any = await request(url, {
      ...options,
      skipErrorHandler: true,
      getResponse: false,
    });
    const { code, result } = res;
    if (!code) {
      return normalizeResult(res);
    }
    if (code === 200) {
      return normalizeResult(result);
    } else {
      const errMsg = res.message || res.detail || 'Request failed';
      message.error(errMsg);
      return { status: 'error', error: errMsg } as any;
    }
  } catch (error: any) {
    const errMsg = extractErrorMessage(error);
    message.error(errMsg);
    return { error: errMsg, status: 'error' } as any;
  }
}

export default MyRequest;

export interface Response<T> {
  code: number;
  message: string;
  result: T;
  success: boolean;
  timestamp: number;
}
