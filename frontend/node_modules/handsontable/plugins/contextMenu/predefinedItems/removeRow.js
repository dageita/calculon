"use strict";

exports.__esModule = true;
exports.default = removeRowItem;
var _utils = require("../utils");
var _utils2 = require("../../../selection/utils");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const KEY = 'remove_row';

/**
 * @returns {object}
 */
exports.KEY = KEY;
function removeRowItem() {
  return {
    key: KEY,
    name() {
      const selection = this.getSelected();
      let pluralForm = 0;
      if (selection) {
        if (selection.length > 1) {
          pluralForm = 1;
        } else {
          const [fromRow,, toRow] = selection[0];
          if (fromRow - toRow !== 0) {
            pluralForm = 1;
          }
        }
      }
      return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_REMOVE_ROW, pluralForm);
    },
    callback() {
      // TODO: Please keep in mind that below `1` may be improper. The table's way of work, before change `f1747b3912ea3b21fe423fd102ca94c87db81379` was restored.
      // There is still problem when removing more than one row.
      this.alter('remove_row', (0, _utils2.transformSelectionToRowDistance)(this.getSelected()), 1, 'ContextMenu.removeRow');
    },
    disabled() {
      const selected = (0, _utils.getValidSelection)(this);
      if (!selected) {
        return true;
      }
      const totalRows = this.countRows();
      if (this.selection.isSelectedByCorner()) {
        // Enable "Remove row" only when there is at least one row.
        return totalRows === 0;
      }
      return this.selection.isSelectedByColumnHeader() || totalRows === 0;
    },
    hidden() {
      return !this.getSettings().allowRemoveRow;
    }
  };
}