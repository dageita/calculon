import { defineConfig } from 'umi';

export default defineConfig({
  title: 'LLM Training Calculator',
  nodeModulesTransform: {
    type: 'none',
  },
  history: {
    type: 'hash'
  },
  base: '/',
  publicPath: '/',
  hash: true,
  mock:false,
  routes: [
    { path: '/help', component: '@/pages/help/index' },
    { path: '/', component: '@/pages/index' },
  ],
  fastRefresh: {},
  proxy: {
    '/llm_training_calculator': {
      target: 'http://10.121.129.10:8000',
      changeOrigin: true,
    },
  }
});
