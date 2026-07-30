"use strict";

exports.__esModule = true;
exports.default = top;
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _utils = require("../utils");
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
/**
 * @param {CustomBorders} customBordersPlugin The plugin instance.
 * @returns {object}
 */
function top(customBordersPlugin) {
  return {
    key: 'borders:top',
    name() {
      let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_BORDERS_TOP);
      const hasBorder = (0, _utils.checkSelectionBorders)(this, 'top');
      if (hasBorder) {
        label = (0, _utils.markSelected)(label);
      }
      return label;
    },
    callback(key, selected) {
      const hasBorder = (0, _utils.checkSelectionBorders)(this, 'top');
      customBordersPlugin.prepareBorder(selected, 'top', hasBorder);
    }
  };
}