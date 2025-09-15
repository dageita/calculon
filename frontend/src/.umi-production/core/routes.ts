// @ts-nocheck
import React from 'react';
import { ApplyPluginsType } from 'D:/works/LLM/trainer/llm-training-calculator/frontend/node_modules/@umijs/runtime';
import * as umiExports from './umiExports';
import { plugin } from './plugin';

export function getRoutes() {
  const routes = [
  {
    "path": "/help",
    "component": require('@/pages/help/index').default,
    "exact": true
  },
  {
    "path": "/",
    "component": require('@/pages/index').default,
    "exact": true
  }
];

  // allow user to extend routes
  plugin.applyPlugins({
    key: 'patchRoutes',
    type: ApplyPluginsType.event,
    args: { routes },
  });

  return routes;
}
