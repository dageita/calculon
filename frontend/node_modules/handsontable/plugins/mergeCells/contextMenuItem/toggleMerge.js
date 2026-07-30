"use strict";

exports.__esModule = true;
exports.default = toggleMergeItem;
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _cellCoords = _interopRequireDefault(require("../cellCoords"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
/**
 * @param {*} plugin The plugin instance.
 * @returns {object}
 */
function toggleMergeItem(plugin) {
  return {
    key: 'mergeCells',
    name() {
      const sel = this.getSelectedLast();
      if (sel) {
        const info = plugin.mergedCellsCollection.get(sel[0], sel[1]);
        if (info.row === sel[0] && info.col === sel[1] && info.row + info.rowspan - 1 === sel[2] && info.col + info.colspan - 1 === sel[3]) {
          return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_UNMERGE_CELLS);
        }
      }
      return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_MERGE_CELLS);
    },
    callback() {
      plugin.toggleMergeOnSelection();
    },
    disabled() {
      const sel = this.getSelectedLast();
      if (!sel) {
        return true;
      }
      const isSingleCell = _cellCoords.default.isSingleCell({
        row: sel[0],
        col: sel[1],
        rowspan: sel[2] - sel[0] + 1,
        colspan: sel[3] - sel[1] + 1
      });
      return isSingleCell || this.selection.isSelectedByCorner();
    },
    hidden: false
  };
}