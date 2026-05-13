"use strict";

exports.__esModule = true;
require("core-js/modules/es.array.push.js");
require("core-js/modules/es.error.cause.js");
var _base = require("../base");
var _element = require("../../helpers/dom/element");
var _number = require("../../helpers/number");
var _array = require("../../helpers/array");
var _object = require("../../helpers/object");
var _mixed = require("../../helpers/mixed");
var _predefinedItems = require("../contextMenu/predefinedItems");
var _pluginHooks = _interopRequireDefault(require("../../pluginHooks"));
var _hideRow = _interopRequireDefault(require("./contextMenuItem/hideRow"));
var _showRow = _interopRequireDefault(require("./contextMenuItem/showRow"));
var _translations = require("../../translations");
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _classPrivateFieldInitSpec(obj, privateMap, value) { _checkPrivateRedeclaration(obj, privateMap); privateMap.set(obj, value); }
function _checkPrivateRedeclaration(obj, privateCollection) { if (privateCollection.has(obj)) { throw new TypeError("Cannot initialize the same private elements twice on an object"); } }
function _classPrivateFieldGet(receiver, privateMap) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "get"); return _classApplyDescriptorGet(receiver, descriptor); }
function _classApplyDescriptorGet(receiver, descriptor) { if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }
function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "set"); _classApplyDescriptorSet(receiver, descriptor, value); return value; }
function _classExtractFieldDescriptor(receiver, privateMap, action) { if (!privateMap.has(receiver)) { throw new TypeError("attempted to " + action + " private field on non-instance"); } return privateMap.get(receiver); }
function _classApplyDescriptorSet(receiver, descriptor, value) { if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } }
_pluginHooks.default.getSingleton().register('beforeHideRows');
_pluginHooks.default.getSingleton().register('afterHideRows');
_pluginHooks.default.getSingleton().register('beforeUnhideRows');
_pluginHooks.default.getSingleton().register('afterUnhideRows');
const PLUGIN_KEY = 'hiddenRows';
exports.PLUGIN_KEY = PLUGIN_KEY;
const PLUGIN_PRIORITY = 320;

/* eslint-disable jsdoc/require-description-complete-sentence */

/**
 * @plugin HiddenRows
 * @class HiddenRows
 *
 * @description
 * The `HiddenRows` plugin lets you [hide specified rows](@/guides/rows/row-hiding.md).
 *
 * "Hiding a row" means that the hidden row doesn't get rendered as a DOM element.
 *
 * The `HiddenRows` plugin doesn't modify the source data,
 * and doesn't participate in data transformation
 * (the shape of the data returned by the [`getData*()` methods](@/api/core.md#getdata) stays intact).
 *
 * You can set the following configuration options:
 *
 * | Option | Required | Type | Default | Description |
 * |---|---|---|---|---|
 * | `rows` | No | Array | - | [Hides specified rows by default](@/guides/rows/row-hiding.md#step-1-specify-rows-hidden-by-default) |
 * | `indicators` | No | Boolean | `false` | [Shows UI indicators](@/guides/rows/row-hiding.md#step-2-show-ui-indicators) |
 * | `copyPasteEnabled` | No | Boolean | `true` | [Sets up copy/paste behavior](@/guides/rows/row-hiding.md#step-4-set-up-copy-and-paste-behavior) |
 *
 * @example
 *
 * ::: only-for javascript
 * ```js
 * const container = document.getElementById('example');
 * const hot = new Handsontable(container, {
 *   data: getData(),
 *   hiddenRows: {
 *     copyPasteEnabled: true,
 *     indicators: true,
 *     rows: [1, 2, 5]
 *   }
 * });
 *
 * // access the `HiddenRows` plugin's instance
 * const hiddenRowsPlugin = hot.getPlugin('hiddenRows');
 *
 * // hide a single row
 * hiddenRowsPlugin.hideRow(1);
 *
 * // hide multiple rows
 * hiddenRowsPlugin.hideRow(1, 2, 9);
 *
 * // hide multiple rows as an array
 * hiddenRowsPlugin.hideRows([1, 2, 9]);
 *
 * // unhide a single row
 * hiddenRowsPlugin.showRow(1);
 *
 * // unhide multiple rows
 * hiddenRowsPlugin.showRow(1, 2, 9);
 *
 * // unhide multiple rows as an array
 * hiddenRowsPlugin.showRows([1, 2, 9]);
 *
 * // to see your changes, re-render your Handsontable instance
 * hot.render();
 * ```
 * :::
 *
 * ::: only-for react
 * ```jsx
 * const hotRef = useRef(null);
 *
 * ...
 *
 * <HotTable
 *   ref={hotRef}
 *   data={getData()}
 *   hiddenRows={{
 *     copyPasteEnabled: true,
 *     indicators: true,
 *     rows: [1, 2, 5]
 *   }}
 * />
 *
 * // access the `HiddenRows` plugin's instance
 * const hot = hotRef.current.hotInstance;
 * const hiddenRowsPlugin = hot.getPlugin('hiddenRows');
 *
 * // hide a single row
 * hiddenRowsPlugin.hideRow(1);
 *
 * // hide multiple rows
 * hiddenRowsPlugin.hideRow(1, 2, 9);
 *
 * // hide multiple rows as an array
 * hiddenRowsPlugin.hideRows([1, 2, 9]);
 *
 * // unhide a single row
 * hiddenRowsPlugin.showRow(1);
 *
 * // unhide multiple rows
 * hiddenRowsPlugin.showRow(1, 2, 9);
 *
 * // unhide multiple rows as an array
 * hiddenRowsPlugin.showRows([1, 2, 9]);
 *
 * // to see your changes, re-render your Handsontable instance
 * hot.render();
 * ```
 * :::
 */
exports.PLUGIN_PRIORITY = PLUGIN_PRIORITY;
var _settings = /*#__PURE__*/new WeakMap();
var _hiddenRowsMap = /*#__PURE__*/new WeakMap();
class HiddenRows extends _base.BasePlugin {
  constructor() {
    super(...arguments);
    /**
     * Cached settings from Handsontable settings.
     *
     * @private
     * @type {object}
     */
    _classPrivateFieldInitSpec(this, _settings, {
      writable: true,
      value: {}
    });
    /**
     * Map of hidden rows by the plugin.
     *
     * @private
     * @type {HidingMap|null}
     */
    _classPrivateFieldInitSpec(this, _hiddenRowsMap, {
      writable: true,
      value: null
    });
  }
  static get PLUGIN_KEY() {
    return PLUGIN_KEY;
  }
  static get PLUGIN_PRIORITY() {
    return PLUGIN_PRIORITY;
  }
  /**
   * Checks if the plugin is enabled in the handsontable settings. This method is executed in {@link Hooks#beforeInit}
   * hook and if it returns `true` then the {@link HiddenRows#enablePlugin} method is called.
   *
   * @returns {boolean}
   */
  isEnabled() {
    return !!this.hot.getSettings()[PLUGIN_KEY];
  }

  /**
   * Enables the plugin functionality for this Handsontable instance.
   */
  enablePlugin() {
    var _this = this;
    if (this.enabled) {
      return;
    }
    const pluginSettings = this.hot.getSettings()[PLUGIN_KEY];
    if ((0, _object.isObject)(pluginSettings)) {
      _classPrivateFieldSet(this, _settings, pluginSettings);
      if ((0, _mixed.isUndefined)(pluginSettings.copyPasteEnabled)) {
        pluginSettings.copyPasteEnabled = true;
      }
    }
    _classPrivateFieldSet(this, _hiddenRowsMap, new _translations.HidingMap());
    _classPrivateFieldGet(this, _hiddenRowsMap).addLocalHook('init', () => this.onMapInit());
    this.hot.rowIndexMapper.registerMap(this.pluginName, _classPrivateFieldGet(this, _hiddenRowsMap));
    this.addHook('afterContextMenuDefaultOptions', function () {
      return _this.onAfterContextMenuDefaultOptions(...arguments);
    });
    this.addHook('afterGetCellMeta', (row, col, cellProperties) => this.onAfterGetCellMeta(row, col, cellProperties));
    this.addHook('modifyRowHeight', (height, row) => this.onModifyRowHeight(height, row));
    this.addHook('afterGetRowHeader', function () {
      return _this.onAfterGetRowHeader(...arguments);
    });
    this.addHook('modifyCopyableRange', ranges => this.onModifyCopyableRange(ranges));
    super.enablePlugin();
  }

  /**
   * Updates the plugin's state.
   *
   * This method is executed when [`updateSettings()`](@/api/core.md#updatesettings) is invoked with any of the following configuration options:
   *  - [`hiddenRows`](@/api/options.md#hiddenrows)
   */
  updatePlugin() {
    this.disablePlugin();
    this.enablePlugin();
    super.updatePlugin();
  }

  /**
   * Disables the plugin functionality for this Handsontable instance.
   */
  disablePlugin() {
    this.hot.rowIndexMapper.unregisterMap(this.pluginName);
    _classPrivateFieldSet(this, _settings, {});
    super.disablePlugin();
    this.resetCellsMeta();
  }

  /**
   * Shows the rows provided in the array.
   *
   * @param {number[]} rows Array of visual row indexes.
   */
  showRows(rows) {
    const currentHideConfig = this.getHiddenRows();
    const isValidConfig = this.isValidConfig(rows);
    let destinationHideConfig = currentHideConfig;
    const hidingMapValues = _classPrivateFieldGet(this, _hiddenRowsMap).getValues().slice();
    const isAnyRowShowed = rows.length > 0;
    if (isValidConfig && isAnyRowShowed) {
      const physicalRows = rows.map(visualRow => this.hot.toPhysicalRow(visualRow));

      // Preparing new values for hiding map.
      (0, _array.arrayEach)(physicalRows, physicalRow => {
        hidingMapValues[physicalRow] = false;
      });

      // Preparing new hiding config.
      destinationHideConfig = (0, _array.arrayReduce)(hidingMapValues, (hiddenIndexes, isHidden, physicalIndex) => {
        if (isHidden) {
          hiddenIndexes.push(this.hot.toVisualRow(physicalIndex));
        }
        return hiddenIndexes;
      }, []);
    }
    const continueHiding = this.hot.runHooks('beforeUnhideRows', currentHideConfig, destinationHideConfig, isValidConfig && isAnyRowShowed);
    if (continueHiding === false) {
      return;
    }
    if (isValidConfig && isAnyRowShowed) {
      _classPrivateFieldGet(this, _hiddenRowsMap).setValues(hidingMapValues);
    }
    this.hot.runHooks('afterUnhideRows', currentHideConfig, destinationHideConfig, isValidConfig && isAnyRowShowed, isValidConfig && destinationHideConfig.length < currentHideConfig.length);
  }

  /**
   * Shows the row provided as row index (counting from 0).
   *
   * @param {...number} row Visual row index.
   */
  showRow() {
    for (var _len = arguments.length, row = new Array(_len), _key = 0; _key < _len; _key++) {
      row[_key] = arguments[_key];
    }
    this.showRows(row);
  }

  /**
   * Hides the rows provided in the array.
   *
   * @param {number[]} rows Array of visual row indexes.
   */
  hideRows(rows) {
    const currentHideConfig = this.getHiddenRows();
    const isConfigValid = this.isValidConfig(rows);
    let destinationHideConfig = currentHideConfig;
    if (isConfigValid) {
      destinationHideConfig = Array.from(new Set(currentHideConfig.concat(rows)));
    }
    const continueHiding = this.hot.runHooks('beforeHideRows', currentHideConfig, destinationHideConfig, isConfigValid);
    if (continueHiding === false) {
      return;
    }
    if (isConfigValid) {
      this.hot.batchExecution(() => {
        (0, _array.arrayEach)(rows, visualRow => {
          _classPrivateFieldGet(this, _hiddenRowsMap).setValueAtIndex(this.hot.toPhysicalRow(visualRow), true);
        });
      }, true);
    }
    this.hot.runHooks('afterHideRows', currentHideConfig, destinationHideConfig, isConfigValid, isConfigValid && destinationHideConfig.length > currentHideConfig.length);
  }

  /**
   * Hides the row provided as row index (counting from 0).
   *
   * @param {...number} row Visual row index.
   */
  hideRow() {
    for (var _len2 = arguments.length, row = new Array(_len2), _key2 = 0; _key2 < _len2; _key2++) {
      row[_key2] = arguments[_key2];
    }
    this.hideRows(row);
  }

  /**
   * Returns an array of visual indexes of hidden rows.
   *
   * @returns {number[]}
   */
  getHiddenRows() {
    return (0, _array.arrayMap)(_classPrivateFieldGet(this, _hiddenRowsMap).getHiddenIndexes(), physicalRowIndex => {
      return this.hot.toVisualRow(physicalRowIndex);
    });
  }

  /**
   * Checks if the provided row is hidden.
   *
   * @param {number} row Visual row index.
   * @returns {boolean}
   */
  isHidden(row) {
    return _classPrivateFieldGet(this, _hiddenRowsMap).getValueAtIndex(this.hot.toPhysicalRow(row)) || false;
  }

  /**
   * Checks whether all of the provided row indexes are within the bounds of the table.
   *
   * @param {Array} hiddenRows List of hidden visual row indexes.
   * @returns {boolean}
   */
  isValidConfig(hiddenRows) {
    const nrOfRows = this.hot.countRows();
    if (Array.isArray(hiddenRows) && hiddenRows.length > 0) {
      return hiddenRows.every(visualRow => Number.isInteger(visualRow) && visualRow >= 0 && visualRow < nrOfRows);
    }
    return false;
  }

  /**
   * Resets all rendered cells meta.
   *
   * @private
   */
  resetCellsMeta() {
    (0, _array.arrayEach)(this.hot.getCellsMeta(), meta => {
      if (meta) {
        meta.skipRowOnPaste = false;
      }
    });
  }

  /**
   * Adds the additional row height for the hidden row indicators.
   *
   * @private
   * @param {number|undefined} height Row height.
   * @param {number} row Visual row index.
   * @returns {number}
   */
  onModifyRowHeight(height, row) {
    // Hook is triggered internally only for the visible rows. Conditional will be handled for the API
    // calls of the `getRowHeight` function on not visible indexes.
    if (this.isHidden(row)) {
      return 0;
    }
    return height;
  }

  /**
   * Sets the copy-related cell meta.
   *
   * @private
   * @param {number} row Visual row index.
   * @param {number} column Visual column index.
   * @param {object} cellProperties Object containing the cell properties.
   */
  onAfterGetCellMeta(row, column, cellProperties) {
    if (_classPrivateFieldGet(this, _settings).copyPasteEnabled === false && this.isHidden(row)) {
      // Cell property handled by the `Autofill` and the `CopyPaste` plugins.
      cellProperties.skipRowOnPaste = true;
    }
    if (this.isHidden(row - 1)) {
      cellProperties.className = cellProperties.className || '';
      if (cellProperties.className.indexOf('afterHiddenRow') === -1) {
        cellProperties.className += ' afterHiddenRow';
      }
    } else if (cellProperties.className) {
      const classArr = cellProperties.className.split(' ');
      if (classArr.length > 0) {
        const containAfterHiddenRow = classArr.indexOf('afterHiddenRow');
        if (containAfterHiddenRow > -1) {
          classArr.splice(containAfterHiddenRow, 1);
        }
        cellProperties.className = classArr.join(' ');
      }
    }
  }

  /**
   * Modifies the copyable range, accordingly to the provided config.
   *
   * @private
   * @param {Array} ranges An array of objects defining copyable cells.
   * @returns {Array}
   */
  onModifyCopyableRange(ranges) {
    // Ranges shouldn't be modified when `copyPasteEnabled` option is set to `true` (by default).
    if (_classPrivateFieldGet(this, _settings).copyPasteEnabled) {
      return ranges;
    }
    const newRanges = [];
    const pushRange = (startRow, endRow, startCol, endCol) => {
      newRanges.push({
        startRow,
        endRow,
        startCol,
        endCol
      });
    };
    (0, _array.arrayEach)(ranges, range => {
      let isHidden = true;
      let rangeStart = 0;
      (0, _number.rangeEach)(range.startRow, range.endRow, visualRow => {
        if (this.isHidden(visualRow)) {
          if (!isHidden) {
            pushRange(rangeStart, visualRow - 1, range.startCol, range.endCol);
          }
          isHidden = true;
        } else {
          if (isHidden) {
            rangeStart = visualRow;
          }
          if (visualRow === range.endRow) {
            pushRange(rangeStart, visualRow, range.startCol, range.endCol);
          }
          isHidden = false;
        }
      });
    });
    return newRanges;
  }

  /**
   * Adds the needed classes to the headers.
   *
   * @private
   * @param {number} row Visual row index.
   * @param {HTMLElement} TH Header's TH element.
   */
  onAfterGetRowHeader(row, TH) {
    if (!_classPrivateFieldGet(this, _settings).indicators || row < 0) {
      return;
    }
    const classList = [];
    if (row >= 1 && this.isHidden(row - 1)) {
      classList.push('afterHiddenRow');
    }
    if (row < this.hot.countRows() - 1 && this.isHidden(row + 1)) {
      classList.push('beforeHiddenRow');
    }
    (0, _element.addClass)(TH, classList);
  }

  /**
   * Add Show-hide rows to context menu.
   *
   * @private
   * @param {object} options An array of objects containing information about the pre-defined Context Menu items.
   */
  onAfterContextMenuDefaultOptions(options) {
    options.items.push({
      name: _predefinedItems.SEPARATOR
    }, (0, _hideRow.default)(this), (0, _showRow.default)(this));
  }

  /**
   * On map initialized hook callback.
   *
   * @private
   */
  onMapInit() {
    if (Array.isArray(_classPrivateFieldGet(this, _settings).rows)) {
      this.hideRows(_classPrivateFieldGet(this, _settings).rows);
    }
  }

  /**
   * Destroys the plugin instance.
   */
  destroy() {
    _classPrivateFieldSet(this, _settings, null);
    _classPrivateFieldSet(this, _hiddenRowsMap, null);
    super.destroy();
  }
}
exports.HiddenRows = HiddenRows;