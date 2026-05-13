import { defineConfig } from 'umi';

const apiProxyTarget = process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000';

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
    { path: '/agent', component: '@/pages/agent' },
    { path: '/optimal', component: '@/pages/index' },
    { path: '/guide', component: '@/pages/index' },
    { path: '/',  redirect: '/guide'},
  ],
  fastRefresh: {},
  define: {
    'process.env.API_PROXY_TARGET': apiProxyTarget,
  },
  proxy: {
    '/llm_training_calculator': {
      target: apiProxyTarget,
      changeOrigin: true,
    },
  }
});
