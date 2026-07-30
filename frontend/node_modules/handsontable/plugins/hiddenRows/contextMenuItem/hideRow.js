"use strict";

exports.__esModule = true;
exports.default = hideRowItem;
require("core-js/modules/es.array.push.js");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
/**
 * @param {HiddenRows} hiddenRowsPlugin The plugin instance.
 * @returns {object}
 */
function hideRowItem(hiddenRowsPlugin) {
  return {
    key: 'hidden_rows_hide',
    name() {
      const selection = this.getSelectedLast();
      let pluralForm = 0;
      if (Array.isArray(selection)) {
        const [fromRow,, toRow] = selection;
        if (fromRow - toRow !== 0) {
          pluralForm = 1;
        }
      }
      return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_HIDE_ROW, pluralForm);
    },
    callback() {
      const {
        from,
        to
      } = this.getSelectedRangeLast();
      const start = Math.max(Math.min(from.row, to.row), 0);
      const end = Math.max(from.row, to.row);
      const rowsToHide = [];
      for (let visualRow = start; visualRow <= end; visualRow += 1) {
        rowsToHide.push(visualRow);
      }
      hiddenRowsPlugin.hideRows(rowsToHide);
      const lastHiddenRow = rowsToHide[rowsToHide.length - 1];
      const rowToSelect = this.rowIndexMapper.getNearestNotHiddenIndex(lastHiddenRow, 1, true);
      if (Number.isInteger(rowToSelect) && rowToSelect >= 0) {
        this.selectRows(rowToSelect);
      } else {
        this.deselectCell();
      }
      this.render();
      this.view.adjustElementsSize(true);
    },
    disabled: false,
    hidden() {
      return !(this.selection.isSelectedByRowHeader() || this.selection.isSelectedByCorner());
    }
  };
}