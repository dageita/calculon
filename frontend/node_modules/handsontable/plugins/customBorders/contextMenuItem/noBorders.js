"use strict";

exports.__esModule = true;
exports.default = noBorders;
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _utils = require("../utils");
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
/**
 * @param {CustomBorders} customBordersPlugin The plugin instance.
 * @returns {object}
 */
function noBorders(customBordersPlugin) {
  return {
    key: 'borders:no_borders',
    name() {
      return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_REMOVE_BORDERS);
    },
    callback(key, selected) {
      customBordersPlugin.prepareBorder(selected, 'noBorders');
    },
    disabled() {
      return !(0, _utils.checkSelectionBorders)(this);
    }
  };
}