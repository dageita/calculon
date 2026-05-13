// @ts-nocheck
import React from 'react';
import { ApplyPluginsType } from '/src/simulator/frontend/node_modules/@umijs/runtime';
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
    "path": "/agent",
    "component": require('@/pages/agent').default,
    "exact": true
  },
  {
    "path": "/optimal",
    "component": require('@/pages/index').default,
    "exact": true
  },
  {
    "path": "/guide",
    "component": require('@/pages/index').default,
    "exact": true
  },
  {
    "path": "/",
    "redirect": "/guide",
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
