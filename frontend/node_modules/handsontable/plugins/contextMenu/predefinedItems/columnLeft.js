"use strict";

exports.__esModule = true;
exports.default = columnLeftItem;
var _utils = require("../utils");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const KEY = 'col_left';

/**
 * @returns {object}
 */
exports.KEY = KEY;
function columnLeftItem() {
  return {
    key: KEY,
    name() {
      return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_INSERT_LEFT);
    },
    callback() {
      const latestSelection = this.getSelectedRangeLast().getTopLeftCorner();
      const alterAction = this.isRtl() ? 'insert_col_end' : 'insert_col_start';
      this.alter(alterAction, latestSelection.col, 1, 'ContextMenu.columnLeft');
    },
    disabled() {
      if (!this.isColumnModificationAllowed()) {
        return true;
      }
      const selected = (0, _utils.getValidSelection)(this);
      if (!selected) {
        return true;
      }
      if (this.selection.isSelectedByCorner()) {
        const totalColumns = this.countCols();

        // Enable "Insert column left" only when there is at least one column.
        return totalColumns === 0;
      }
      return this.selection.isSelectedByRowHeader() || this.countCols() >= this.getSettings().maxCols;
    },
    hidden() {
      return !this.getSettings().allowInsertColumn;
    }
  };
}