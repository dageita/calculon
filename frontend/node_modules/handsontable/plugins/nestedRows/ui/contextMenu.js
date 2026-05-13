"use strict";

exports.__esModule = true;
var _number = require("../../../helpers/number");
var _array = require("../../../helpers/array");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _base = _interopRequireDefault(require("./_base"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const privatePool = new WeakMap();

/**
 * Class responsible for the Context Menu entries for the Nested Rows plugin.
 *
 * @private
 * @class ContextMenuUI
 * @augments BaseUI
 */
class ContextMenuUI extends _base.default {
  constructor(nestedRowsPlugin, hotInstance) {
    super(nestedRowsPlugin, hotInstance);
    privatePool.set(this, {
      row_above: (key, selection) => {
        const lastSelection = selection[selection.length - 1];
        this.dataManager.addSibling(lastSelection.start.row, 'above');
      },
      row_below: (key, selection) => {
        const lastSelection = selection[selection.length - 1];
        this.dataManager.addSibling(lastSelection.start.row, 'below');
      }
    });
    /**
     * Reference to the DataManager instance connected with the Nested Rows plugin.
     *
     * @type {DataManager}
     */
    this.dataManager = this.plugin.dataManager;
  }
  /**
   * Append options to the context menu. (Propagated from the `afterContextMenuDefaultOptions` hook callback)
   * f.
   *
   * @private
   * @param {object} defaultOptions Default context menu options.
   * @returns {*}
   */
  appendOptions(defaultOptions) {
    const newEntries = [{
      key: 'add_child',
      name() {
        return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_NESTED_ROWS_INSERT_CHILD);
      },
      callback: () => {
        const translatedRowIndex = this.dataManager.translateTrimmedRow(this.hot.getSelectedLast()[0]);
        const parent = this.dataManager.getDataObject(translatedRowIndex);
        this.dataManager.addChild(parent);
      },
      disabled: () => {
        const selected = this.hot.getSelectedLast();
        return !selected || selected[0] < 0 || this.hot.selection.isSelectedByColumnHeader() || this.hot.countRows() >= this.hot.getSettings().maxRows;
      }
    }, {
      key: 'detach_from_parent',
      name() {
        return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_NESTED_ROWS_DETACH_CHILD);
      },
      callback: () => {
        this.dataManager.detachFromParent(this.hot.getSelectedLast());
      },
      disabled: () => {
        const selected = this.hot.getSelectedLast();
        const translatedRowIndex = this.dataManager.translateTrimmedRow(selected[0]);
        const parent = this.dataManager.getRowParent(translatedRowIndex);
        return !parent || !selected || selected[0] < 0 || this.hot.selection.isSelectedByColumnHeader() || this.hot.countRows() >= this.hot.getSettings().maxRows;
      }
    }, {
      name: '---------'
    }];
    (0, _number.rangeEach)(0, defaultOptions.items.length - 1, i => {
      if (i === 0) {
        (0, _array.arrayEach)(newEntries, (val, j) => {
          defaultOptions.items.splice(i + j, 0, val);
        });
        return false;
      }
    });
    return this.modifyRowInsertingOptions(defaultOptions);
  }

  /**
   * Modify how the row inserting options work.
   *
   * @private
   * @param {object} defaultOptions Default context menu items.
   * @returns {*}
   */
  modifyRowInsertingOptions(defaultOptions) {
    const priv = privatePool.get(this);
    (0, _number.rangeEach)(0, defaultOptions.items.length - 1, i => {
      const option = priv[defaultOptions.items[i].key];
      if (option !== null && option !== void 0) {
        defaultOptions.items[i].callback = option;
      }
    });
    return defaultOptions;
  }
}
var _default = ContextMenuUI;
exports.default = _default;