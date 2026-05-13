// @ts-nocheck
import { Plugin } from '/src/simulator/frontend/node_modules/@umijs/runtime';

const plugin = new Plugin({
  validKeys: ['modifyClientRenderOpts','patchRoutes','rootContainer','render','onRouteChange','__mfsu','getInitialState','initialStateConfig','request',],
});

export { plugin };
