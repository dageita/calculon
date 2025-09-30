import { defineConfig, Redirect } from 'umi';

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
    { path: '/optimal', component: '@/pages/index' },
    { path: '/guide', component: '@/pages/index' },
    { path: '/',  redirect: '/guide'},
  ],
  fastRefresh: {},
  proxy: {
    '/llm_training_calculator': {
      target: 'http://10.121.129.10:8000',
      changeOrigin: true,
    },
  }
});
