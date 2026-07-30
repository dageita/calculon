"use strict";

exports.__esModule = true;
exports.default = readOnlyItem;
var _utils = require("../utils");
var _array = require("../../../helpers/array");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const KEY = 'make_read_only';

/**
 * @returns {object}
 */
exports.KEY = KEY;
function readOnlyItem() {
  return {
    key: KEY,
    name() {
      let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_READ_ONLY);
      const atLeastOneReadOnly = (0, _utils.checkSelectionConsistency)(this.getSelectedRange(), (row, col) => this.getCellMeta(row, col).readOnly);
      if (atLeastOneReadOnly) {
        label = (0, _utils.markLabelAsSelected)(label);
      }
      return label;
    },
    callback() {
      const ranges = this.getSelectedRange();
      const atLeastOneReadOnly = (0, _utils.checkSelectionConsistency)(ranges, (row, col) => this.getCellMeta(row, col).readOnly);
      (0, _array.arrayEach)(ranges, range => {
        range.forAll((row, col) => {
          if (row >= 0 && col >= 0) {
            this.setCellMeta(row, col, 'readOnly', !atLeastOneReadOnly);
          }
        });
      });
      this.render();
    },
    disabled() {
      if (this.selection.isSelectedByCorner()) {
        return true;
      }
      if (this.countRows() === 0 || this.countCols() === 0) {
        return true;
      }
      if (!this.getSelectedRange() || this.getSelectedRange().length === 0) {
        return true;
      }
      return false;
    }
  };
}