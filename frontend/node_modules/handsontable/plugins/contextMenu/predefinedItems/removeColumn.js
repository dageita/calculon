"use strict";

exports.__esModule = true;
exports.default = removeColumnItem;
var _utils = require("../utils");
var _utils2 = require("../../../selection/utils");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const KEY = 'remove_col';

/**
 * @returns {object}
 */
exports.KEY = KEY;
function removeColumnItem() {
  return {
    key: KEY,
    name() {
      const selection = this.getSelected();
      let pluralForm = 0;
      if (selection) {
        if (selection.length > 1) {
          pluralForm = 1;
        } else {
          const [, fromColumn,, toColumn] = selection[0];
          if (fromColumn - toColumn !== 0) {
            pluralForm = 1;
          }
        }
      }
      return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_REMOVE_COLUMN, pluralForm);
    },
    callback() {
      this.alter('remove_col', (0, _utils2.transformSelectionToColumnDistance)(this.getSelected()), null, 'ContextMenu.removeColumn');
    },
    disabled() {
      if (!this.isColumnModificationAllowed()) {
        return true;
      }
      const selected = (0, _utils.getValidSelection)(this);
      if (!selected) {
        return true;
      }
      const totalColumns = this.countCols();
      if (this.selection.isSelectedByCorner()) {
        // Enable "Remove column" only when there is at least one column.
        return totalColumns === 0;
      }
      return this.selection.isSelectedByRowHeader() || totalColumns === 0;
    },
    hidden() {
      return !this.getSettings().allowRemoveColumn;
    }
  };
}