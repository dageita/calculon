import { request } from 'umi';
import { message } from 'antd';
import { reject } from 'lodash';

async function MyRequest<T>(url: string, options: any): Promise<T> {
  try {
    const res: any = await request(url, {
      ...options,
      skipErrorHandler: true,
      getResponse: false,
    });
    const { code, result } = res;
    if (!code) {
      return res;
    }
    if (code === 200) {
      return result;
    } else {
      // 错误处理
      message.error(res.message);
      return result;
    }
  } catch (error: any) {
    return new Promise((resolve, reject) => {
      resolve({ error } as any);

    })
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
