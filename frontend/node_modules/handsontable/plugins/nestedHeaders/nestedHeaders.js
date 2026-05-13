"use strict";

require("core-js/modules/es.error.cause.js");
exports.__esModule = true;
require("core-js/modules/es.array.push.js");
var _element = require("../../helpers/dom/element");
var _number = require("../../helpers/number");
var _event = require("../../helpers/dom/event");
var _templateLiteralTag = require("../../helpers/templateLiteralTag");
var _console = require("../../helpers/console");
var _selection = require("../../selection");
var _base = require("../base");
var _stateManager2 = _interopRequireDefault(require("./stateManager"));
var _ghostTable = _interopRequireDefault(require("./utils/ghostTable"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _defineProperty(obj, key, value) { key = _toPropertyKey(key); if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _toPropertyKey(arg) { var key = _toPrimitive(arg, "string"); return typeof key === "symbol" ? key : String(key); }
function _toPrimitive(input, hint) { if (typeof input !== "object" || input === null) return input; var prim = input[Symbol.toPrimitive]; if (prim !== undefined) { var res = prim.call(input, hint || "default"); if (typeof res !== "object") return res; throw new TypeError("@@toPrimitive must return a primitive value."); } return (hint === "string" ? String : Number)(input); }
function _classPrivateFieldInitSpec(obj, privateMap, value) { _checkPrivateRedeclaration(obj, privateMap); privateMap.set(obj, value); }
function _checkPrivateRedeclaration(obj, privateCollection) { if (privateCollection.has(obj)) { throw new TypeError("Cannot initialize the same private elements twice on an object"); } }
function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "set"); _classApplyDescriptorSet(receiver, descriptor, value); return value; }
function _classApplyDescriptorSet(receiver, descriptor, value) { if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } }
function _classPrivateFieldGet(receiver, privateMap) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "get"); return _classApplyDescriptorGet(receiver, descriptor); }
function _classExtractFieldDescriptor(receiver, privateMap, action) { if (!privateMap.has(receiver)) { throw new TypeError("attempted to " + action + " private field on non-instance"); } return privateMap.get(receiver); }
function _classApplyDescriptorGet(receiver, descriptor) { if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }
const PLUGIN_KEY = 'nestedHeaders';
exports.PLUGIN_KEY = PLUGIN_KEY;
const PLUGIN_PRIORITY = 280;

/* eslint-disable jsdoc/require-description-complete-sentence */

/**
 * @plugin NestedHeaders
 * @class NestedHeaders
 *
 * @description
 * The plugin allows to create a nested header structure, using the HTML's colspan attribute.
 *
 * To make any header wider (covering multiple table columns), it's corresponding configuration array element should be
 * provided as an object with `label` and `colspan` properties. The `label` property defines the header's label,
 * while the `colspan` property defines a number of columns that the header should cover.
 *
 * __Note__ that the plugin supports a *nested* structure, which means, any header cannot be wider than it's "parent". In
 * other words, headers cannot overlap each other.
 * @example
 *
 * ::: only-for javascript
 * ```js
 * const container = document.getElementById('example');
 * const hot = new Handsontable(container, {
 *   data: getData(),
 *   nestedHeaders: [
 *     ['A', {label: 'B', colspan: 8}, 'C'],
 *     ['D', {label: 'E', colspan: 4}, {label: 'F', colspan: 4}, 'G'],
 *     ['H', {label: 'I', colspan: 2}, {label: 'J', colspan: 2}, {label: 'K', colspan: 2}, {label: 'L', colspan: 2}, 'M'],
 *     ['N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
 *  ],
 * ```
 * :::
 *
 * ::: only-for react
 * ```jsx
 * <HotTable
 *   data={getData()}
 *   nestedHeaders={[
 *     ['A', {label: 'B', colspan: 8}, 'C'],
 *     ['D', {label: 'E', colspan: 4}, {label: 'F', colspan: 4}, 'G'],
 *     ['H', {label: 'I', colspan: 2}, {label: 'J', colspan: 2}, {label: 'K', colspan: 2}, {label: 'L', colspan: 2}, 'M'],
 *     ['N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
 *  ]}
 * />
 * ```
 * :::
 */
exports.PLUGIN_PRIORITY = PLUGIN_PRIORITY;
var _stateManager = /*#__PURE__*/new WeakMap();
var _hidingIndexMapObserver = /*#__PURE__*/new WeakMap();
class NestedHeaders extends _base.BasePlugin {
  constructor() {
    super(...arguments);
    /**
     * The state manager for the nested headers.
     *
     * @private
     * @type {StateManager}
     */
    _classPrivateFieldInitSpec(this, _stateManager, {
      writable: true,
      value: new _stateManager2.default()
    });
    /**
     * The instance of the ChangesObservable class that allows track the changes that happens in the
     * column indexes.
     *
     * @private
     * @type {ChangesObservable}
     */
    _classPrivateFieldInitSpec(this, _hidingIndexMapObserver, {
      writable: true,
      value: null
    });
    /**
     * Custom helper for getting widths of the nested headers.
     *
     * @private
     * @type {GhostTable}
     */
    // @TODO This should be changed after refactor handsontable/utils/ghostTable.
    _defineProperty(this, "ghostTable", new _ghostTable.default(this.hot, (row, column) => this.getHeaderSettings(row, column)));
    /**
     * The flag which determines that the nested header settings contains overlapping headers
     * configuration.
     *
     * @type {boolean}
     */
    _defineProperty(this, "detectedOverlappedHeaders", false);
  }
  static get PLUGIN_KEY() {
    return PLUGIN_KEY;
  }
  static get PLUGIN_PRIORITY() {
    return PLUGIN_PRIORITY;
  }
  /**
   * Check if plugin is enabled.
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
    const {
      nestedHeaders
    } = this.hot.getSettings();
    if (!Array.isArray(nestedHeaders) || !Array.isArray(nestedHeaders[0])) {
      (0, _console.warn)((0, _templateLiteralTag.toSingleLine)`Your Nested Headers plugin configuration is invalid. The settings has to be\x20
                        passed as an array of arrays e.q. [['A1', { label: 'A2', colspan: 2 }]]`);
    }
    this.addHook('init', () => this.onInit());
    this.addHook('afterLoadData', function () {
      return _this.onAfterLoadData(...arguments);
    });
    this.addHook('beforeOnCellMouseDown', function () {
      return _this.onBeforeOnCellMouseDown(...arguments);
    });
    this.addHook('afterOnCellMouseDown', function () {
      return _this.onAfterOnCellMouseDown(...arguments);
    });
    this.addHook('beforeOnCellMouseOver', function () {
      return _this.onBeforeOnCellMouseOver(...arguments);
    });
    this.addHook('afterGetColumnHeaderRenderers', array => this.onAfterGetColumnHeaderRenderers(array));
    this.addHook('modifyColWidth', function () {
      return _this.onModifyColWidth(...arguments);
    });
    this.addHook('modifyColumnHeaderValue', function () {
      return _this.onModifyColumnHeaderValue(...arguments);
    });
    this.addHook('beforeHighlightingColumnHeader', function () {
      return _this.onBeforeHighlightingColumnHeader(...arguments);
    });
    this.addHook('beforeCopy', function () {
      return _this.onBeforeCopy(...arguments);
    });
    this.addHook('afterViewportColumnCalculatorOverride', function () {
      return _this.onAfterViewportColumnCalculatorOverride(...arguments);
    });
    super.enablePlugin();
    this.updatePlugin(); // @TODO: Workaround for broken plugin initialization abstraction.
  }

  /**
   * Updates the plugin's state.
   *
   * This method is executed when [`updateSettings()`](@/api/core.md#updatesettings) is invoked with any of the following configuration options:
   *  - [`nestedHeaders`](@/api/options.md#nestedheaders)
   */
  updatePlugin() {
    if (!this.hot.view) {
      // @TODO: Workaround for broken plugin initialization abstraction.
      return;
    }
    const {
      nestedHeaders
    } = this.hot.getSettings();
    _classPrivateFieldGet(this, _stateManager).setColumnsLimit(this.hot.countCols());
    if (Array.isArray(nestedHeaders)) {
      this.detectedOverlappedHeaders = _classPrivateFieldGet(this, _stateManager).setState(nestedHeaders);
    }
    if (this.detectedOverlappedHeaders) {
      (0, _console.warn)((0, _templateLiteralTag.toSingleLine)`Your Nested Headers plugin setup contains overlapping headers. This kind of configuration\x20
                        is currently not supported.`);
    }
    if (this.enabled) {
      // This line covers the case when a developer uses the external hiding maps to manipulate
      // the columns' visibility. The tree state built from the settings - which is always built
      // as if all the columns are visible, needs to be modified to be in sync with a dataset.
      this.hot.columnIndexMapper.hidingMapsCollection.getMergedValues().forEach((isColumnHidden, physicalColumnIndex) => {
        const actionName = isColumnHidden === true ? 'hide-column' : 'show-column';
        _classPrivateFieldGet(this, _stateManager).triggerColumnModification(actionName, physicalColumnIndex);
      });
    }
    if (!_classPrivateFieldGet(this, _hidingIndexMapObserver) && this.enabled) {
      _classPrivateFieldSet(this, _hidingIndexMapObserver, this.hot.columnIndexMapper.createChangesObserver('hiding').subscribe(changes => {
        changes.forEach(_ref => {
          let {
            op,
            index: columnIndex,
            newValue
          } = _ref;
          if (op === 'replace') {
            const actionName = newValue === true ? 'hide-column' : 'show-column';
            _classPrivateFieldGet(this, _stateManager).triggerColumnModification(actionName, columnIndex);
          }
        });
        this.ghostTable.buildWidthsMap();
      }));
    }
    this.ghostTable.setLayersCount(this.getLayersCount()).buildWidthsMap();
    super.updatePlugin();
  }

  /**
   * Disables the plugin functionality for this Handsontable instance.
   */
  disablePlugin() {
    this.clearColspans();
    _classPrivateFieldGet(this, _stateManager).clear();
    _classPrivateFieldGet(this, _hidingIndexMapObserver).unsubscribe();
    _classPrivateFieldSet(this, _hidingIndexMapObserver, null);
    this.ghostTable.clear();
    super.disablePlugin();
  }

  /**
   * Returns an instance of the internal state manager of the plugin.
   *
   * @private
   * @returns {StateManager}
   */
  getStateManager() {
    return _classPrivateFieldGet(this, _stateManager);
  }

  /**
   * Gets a total number of headers levels.
   *
   * @private
   * @returns {number}
   */
  getLayersCount() {
    return _classPrivateFieldGet(this, _stateManager).getLayersCount();
  }

  /**
   * Gets column settings for a specified header. The returned object contains
   * information about the header label, its colspan length, or if it is hidden
   * in the header renderers.
   *
   * @private
   * @param {number} headerLevel Header level (0 = most distant to the table).
   * @param {number} columnIndex A visual column index.
   * @returns {object}
   */
  getHeaderSettings(headerLevel, columnIndex) {
    return _classPrivateFieldGet(this, _stateManager).getHeaderSettings(headerLevel, columnIndex);
  }

  /**
   * Clear the colspans remaining after plugin usage.
   *
   * @private
   */
  clearColspans() {
    if (!this.hot.view) {
      return;
    }
    const {
      _wt: wt
    } = this.hot.view;
    const headerLevels = wt.getSetting('columnHeaders').length;
    const mainHeaders = wt.wtTable.THEAD;
    const topHeaders = wt.wtOverlays.topOverlay.clone.wtTable.THEAD;
    const topLeftCornerHeaders = wt.wtOverlays.topInlineStartCornerOverlay ? wt.wtOverlays.topInlineStartCornerOverlay.clone.wtTable.THEAD : null;
    for (let i = 0; i < headerLevels; i++) {
      const masterLevel = mainHeaders.childNodes[i];
      if (!masterLevel) {
        break;
      }
      const topLevel = topHeaders.childNodes[i];
      const topLeftCornerLevel = topLeftCornerHeaders ? topLeftCornerHeaders.childNodes[i] : null;
      for (let j = 0, masterNodes = masterLevel.childNodes.length; j < masterNodes; j++) {
        masterLevel.childNodes[j].removeAttribute('colspan');
        (0, _element.removeClass)(masterLevel.childNodes[j], 'hiddenHeader');
        if (topLevel && topLevel.childNodes[j]) {
          topLevel.childNodes[j].removeAttribute('colspan');
          (0, _element.removeClass)(topLevel.childNodes[j], 'hiddenHeader');
        }
        if (topLeftCornerHeaders && topLeftCornerLevel && topLeftCornerLevel.childNodes[j]) {
          topLeftCornerLevel.childNodes[j].removeAttribute('colspan');
          (0, _element.removeClass)(topLeftCornerLevel.childNodes[j], 'hiddenHeader');
        }
      }
    }
  }

  /**
   * Generates the appropriate header renderer for a header row.
   *
   * @private
   * @param {number} headerLevel The index of header level counting from the top (positive
   *                             values counting from 0 to N).
   * @returns {Function}
   * @fires Hooks#afterGetColHeader
   */
  headerRendererFactory(headerLevel) {
    var _this2 = this;
    const fixedColumnsStart = this.hot.view._wt.getSetting('fixedColumnsStart');
    return (renderedColumnIndex, TH) => {
      var _classPrivateFieldGet2;
      const {
        columnIndexMapper,
        view
      } = this.hot;
      let visualColumnIndex = columnIndexMapper.getVisualFromRenderableIndex(renderedColumnIndex);
      if (visualColumnIndex === null) {
        visualColumnIndex = renderedColumnIndex;
      }
      TH.removeAttribute('colspan');
      (0, _element.removeClass)(TH, 'hiddenHeader');
      const {
        colspan,
        isHidden,
        isPlaceholder
      } = (_classPrivateFieldGet2 = _classPrivateFieldGet(this, _stateManager).getHeaderSettings(headerLevel, visualColumnIndex)) !== null && _classPrivateFieldGet2 !== void 0 ? _classPrivateFieldGet2 : {
        label: ''
      };
      if (isPlaceholder || isHidden) {
        (0, _element.addClass)(TH, 'hiddenHeader');
      } else if (colspan > 1) {
        var _wtOverlays$topInline, _wtOverlays$inlineSta;
        const {
          wtOverlays
        } = view._wt;
        const isTopInlineStartOverlay = (_wtOverlays$topInline = wtOverlays.topInlineStartCornerOverlay) === null || _wtOverlays$topInline === void 0 ? void 0 : _wtOverlays$topInline.clone.wtTable.THEAD.contains(TH);
        const isInlineStartOverlay = (_wtOverlays$inlineSta = wtOverlays.inlineStartOverlay) === null || _wtOverlays$inlineSta === void 0 ? void 0 : _wtOverlays$inlineSta.clone.wtTable.THEAD.contains(TH);

        // Check if there is a fixed column enabled, if so then reduce colspan to fixed column width.
        const correctedColspan = isTopInlineStartOverlay || isInlineStartOverlay ? Math.min(colspan, fixedColumnsStart - renderedColumnIndex) : colspan;
        if (correctedColspan > 1) {
          TH.setAttribute('colspan', correctedColspan);
        }
      }
      this.hot.view.appendColHeader(visualColumnIndex, TH, function () {
        return _this2.getColumnHeaderValue(...arguments);
      }, headerLevel);
    };
  }

  /**
   * Returns the column header value for specified column and header level index.
   *
   * @private
   * @param {number} visualColumnIndex Visual column index.
   * @param {number} headerLevel The index of header level. The header level accepts positive (0 to N)
   *                             and negative (-1 to -N) values. For positive values, 0 points to the
   *                             top most header, and for negative direction, -1 points to the most bottom
   *                             header (the header closest to the cells).
   * @returns {string} Returns the column header value to update.
   */
  getColumnHeaderValue(visualColumnIndex, headerLevel) {
    var _classPrivateFieldGet3;
    const {
      isHidden,
      isPlaceholder
    } = (_classPrivateFieldGet3 = _classPrivateFieldGet(this, _stateManager).getHeaderSettings(headerLevel, visualColumnIndex)) !== null && _classPrivateFieldGet3 !== void 0 ? _classPrivateFieldGet3 : {};
    if (isPlaceholder || isHidden) {
      return '';
    }
    return this.hot.getColHeader(visualColumnIndex, headerLevel);
  }

  /**
   * Allows to control which header DOM element will be used to highlight.
   *
   * @private
   * @param {number} visualColumn A visual column index of the highlighted row header.
   * @param {number} headerLevel A row header level that is currently highlighted.
   * @param {object} highlightMeta An object with meta data that describes the highlight state.
   * @returns {number}
   */
  onBeforeHighlightingColumnHeader(visualColumn, headerLevel, highlightMeta) {
    const headerNodeData = _classPrivateFieldGet(this, _stateManager).getHeaderTreeNodeData(headerLevel, visualColumn);
    if (!headerNodeData) {
      return visualColumn;
    }
    const {
      classNames,
      columnCursor,
      selectionType,
      selectionWidth
    } = highlightMeta;
    const {
      isRoot,
      colspan
    } = _classPrivateFieldGet(this, _stateManager).getHeaderSettings(headerLevel, visualColumn);
    if (selectionType === _selection.HEADER_TYPE) {
      if (!isRoot) {
        return headerNodeData.columnIndex;
      }
    } else if (selectionType === _selection.ACTIVE_HEADER_TYPE) {
      if (colspan > selectionWidth - columnCursor || !isRoot) {
        // Reset the class names array so the generated TH element won't be modified.
        classNames.length = 0;
      }
    }
    return visualColumn;
  }

  /**
   * Listens the `beforeCopy` hook that allows processing the copied column headers so that the
   * merged column headers do not propagate the value for each column but only once at the beginning
   * of the column.
   *
   * @private
   * @param {Array[]} data An array of arrays which contains data to copied.
   * @param {object[]} copyableRanges An array of objects with ranges of the visual indexes (`startRow`, `startCol`, `endRow`, `endCol`)
   *                                  which will copied.
   * @param {{ columnHeadersCount: number }} copiedHeadersCount An object with keys that holds information with
   *                                                            the number of copied headers.
   */
  onBeforeCopy(data, copyableRanges, _ref2) {
    let {
      columnHeadersCount
    } = _ref2;
    if (columnHeadersCount === 0) {
      return;
    }
    for (let rangeIndex = 0; rangeIndex < copyableRanges.length; rangeIndex++) {
      const {
        startRow,
        startCol,
        endRow,
        endCol
      } = copyableRanges[rangeIndex];
      const rowsCount = endRow - startRow + 1;
      const columnsCount = startCol - endCol + 1;

      // do not process dataset ranges and column headers where only one column is copied
      if (startRow >= 0 || columnsCount === 1) {
        break;
      }
      for (let column = startCol; column <= endCol; column++) {
        for (let row = startRow; row <= endRow; row++) {
          var _classPrivateFieldGet4;
          const zeroBasedColumnHeaderLevel = rowsCount + row;
          const zeroBasedColumnIndex = column - startCol;
          if (zeroBasedColumnIndex === 0) {
            continue; // eslint-disable-line no-continue
          }

          const isRoot = (_classPrivateFieldGet4 = _classPrivateFieldGet(this, _stateManager).getHeaderTreeNodeData(row, column)) === null || _classPrivateFieldGet4 === void 0 ? void 0 : _classPrivateFieldGet4.isRoot;
          if (isRoot === false) {
            data[zeroBasedColumnHeaderLevel][zeroBasedColumnIndex] = '';
          }
        }
      }
    }
  }

  /**
   * Allows blocking the column selection that is controlled by the core Selection module.
   *
   * @private
   * @param {MouseEvent} event Mouse event.
   * @param {CellCoords} coords Cell coords object containing the visual coordinates of the clicked cell.
   * @param {CellCoords} TD The table cell or header element.
   * @param {object} controller An object with properties `row`, `column` and `cell`. Each property contains
   *                            a boolean value that allows or disallows changing the selection for that particular area.
   */
  onBeforeOnCellMouseDown(event, coords, TD, controller) {
    const headerNodeData = this._getHeaderTreeNodeDataByCoords(coords);
    if (headerNodeData) {
      // Block the Selection module in controlling how the columns are selected. Pass the
      // responsibility of the column selection to this plugin (see "onAfterOnCellMouseDown" hook).
      controller.column = true;
    }
  }

  /**
   * Allows to control how the column selection based on the coordinates and the nested headers is made.
   *
   * @private
   * @param {MouseEvent} event Mouse event.
   * @param {CellCoords} coords Cell coords object containing the visual coordinates of the clicked cell.
   */
  onAfterOnCellMouseDown(event, coords) {
    const headerNodeData = this._getHeaderTreeNodeDataByCoords(coords);
    if (!headerNodeData) {
      return;
    }
    const {
      selection
    } = this.hot;
    const currentSelection = selection.isSelected() ? selection.getSelectedRange().current() : null;
    const columnsToSelect = [];
    const {
      columnIndex,
      origColspan
    } = headerNodeData;

    // The Selection module doesn't allow it to extend its behavior easily. That's why here we need
    // to re-implement the "click" and "shift" behavior. As a workaround, the logic for the nested
    // headers must implement a similar logic as in the original Selection handler
    // (see src/selection/mouseEventHandler.js).
    const allowRightClickSelection = !selection.inInSelection(coords);
    if (event.shiftKey && currentSelection) {
      if (coords.col < currentSelection.from.col) {
        columnsToSelect.push(currentSelection.getTopEndCorner().col, columnIndex, coords.row);
      } else if (coords.col > currentSelection.from.col) {
        columnsToSelect.push(currentSelection.getTopStartCorner().col, columnIndex + origColspan - 1, coords.row);
      } else {
        columnsToSelect.push(columnIndex, columnIndex + origColspan - 1, coords.row);
      }
    } else if ((0, _event.isLeftClick)(event) || (0, _event.isRightClick)(event) && allowRightClickSelection) {
      columnsToSelect.push(columnIndex, columnIndex + origColspan - 1, coords.row);
    }

    // The plugin takes control of how the columns are selected.
    selection.selectColumns(...columnsToSelect);
  }

  /**
   * Makes the header-selection properly select the nested headers.
   *
   * @private
   * @param {MouseEvent} event Mouse event.
   * @param {CellCoords} coords Cell coords object containing the visual coordinates of the clicked cell.
   * @param {HTMLElement} TD The cell element.
   * @param {object} controller An object with properties `row`, `column` and `cell`. Each property contains
   *                            a boolean value that allows or disallows changing the selection for that particular area.
   */
  onBeforeOnCellMouseOver(event, coords, TD, controller) {
    if (!this.hot.view.isMouseDown()) {
      return;
    }
    const headerNodeData = this._getHeaderTreeNodeDataByCoords(coords);
    if (!headerNodeData) {
      return;
    }
    const {
      columnIndex,
      origColspan
    } = headerNodeData;
    const selectedRange = this.hot.getSelectedRangeLast();
    const topStartCoords = selectedRange.getTopStartCorner();
    const bottomEndCoords = selectedRange.getBottomEndCorner();
    const {
      from
    } = selectedRange;

    // Block the Selection module in controlling how the columns and cells are selected.
    // From now on, the plugin is responsible for the selection.
    controller.column = true;
    controller.cell = true;
    const columnsToSelect = [];
    if (coords.col < from.col) {
      columnsToSelect.push(bottomEndCoords.col, columnIndex);
    } else if (coords.col > from.col) {
      columnsToSelect.push(topStartCoords.col, columnIndex + origColspan - 1);
    } else {
      columnsToSelect.push(columnIndex, columnIndex + origColspan - 1);
    }
    this.hot.selectColumns(...columnsToSelect);
  }

  /**
   * `afterGetColumnHeader` hook callback - prepares the header structure.
   *
   * @private
   * @param {Array} renderersArray Array of renderers.
   */
  onAfterGetColumnHeaderRenderers(renderersArray) {
    renderersArray.length = 0;
    for (let headerLayer = 0; headerLayer < _classPrivateFieldGet(this, _stateManager).getLayersCount(); headerLayer++) {
      renderersArray.push(this.headerRendererFactory(headerLayer));
    }
  }

  /**
   * Make the renderer render the first nested column in its entirety.
   *
   * @private
   * @param {object} calc Viewport column calculator.
   */
  onAfterViewportColumnCalculatorOverride(calc) {
    const headerLayersCount = _classPrivateFieldGet(this, _stateManager).getLayersCount();
    let newStartColumn = calc.startColumn;
    let nonRenderable = !!headerLayersCount;
    for (let headerLayer = 0; headerLayer < headerLayersCount; headerLayer++) {
      const startColumn = _classPrivateFieldGet(this, _stateManager).findLeftMostColumnIndex(headerLayer, calc.startColumn);
      const renderedStartColumn = this.hot.columnIndexMapper.getRenderableFromVisualIndex(startColumn);

      // If any of the headers for that column index is rendered, all of them should be rendered properly, see
      // comment below.
      if (startColumn >= 0) {
        nonRenderable = false;
      }

      // `renderedStartColumn` can be `null` if the leftmost columns are hidden. In that case -> ignore that header
      // level, as it should be handled by the "parent" header
      if ((0, _number.isNumeric)(renderedStartColumn) && renderedStartColumn < calc.startColumn) {
        newStartColumn = renderedStartColumn;
        break;
      }
    }

    // If no headers for the provided column index are renderable, start rendering from the beginning of the upmost
    // header for that position.
    calc.startColumn = nonRenderable ? _classPrivateFieldGet(this, _stateManager).getHeaderTreeNodeData(0, newStartColumn).columnIndex : newStartColumn;
  }

  /**
   * `modifyColWidth` hook callback - returns width from cache, when is greater than incoming from hook.
   *
   * @private
   * @param {number} width Width from hook.
   * @param {number} column Visual index of an column.
   * @returns {number}
   */
  onModifyColWidth(width, column) {
    const cachedWidth = this.ghostTable.getWidth(column);
    return width > cachedWidth ? width : cachedWidth;
  }

  /**
   * Listens the `modifyColumnHeaderValue` hook that overwrites the column headers values based on
   * the internal state and settings of the plugin.
   *
   * @private
   * @param {string} value The column header value.
   * @param {number} visualColumnIndex The visual column index.
   * @param {number} headerLevel The index of header level. The header level accepts positive (0 to N)
   *                             and negative (-1 to -N) values. For positive values, 0 points to the
   *                             top most header, and for negative direction, -1 points to the most bottom
   *                             header (the header closest to the cells).
   * @returns {string} Returns the column header value to update.
   */
  onModifyColumnHeaderValue(value, visualColumnIndex, headerLevel) {
    var _classPrivateFieldGet5;
    const {
      label
    } = (_classPrivateFieldGet5 = _classPrivateFieldGet(this, _stateManager).getHeaderTreeNodeData(headerLevel, visualColumnIndex)) !== null && _classPrivateFieldGet5 !== void 0 ? _classPrivateFieldGet5 : {
      label: ''
    };
    return label;
  }

  /**
   * Updates the plugin state after HoT initialization.
   *
   * @private
   */
  onInit() {
    // @TODO: Workaround for broken plugin initialization abstraction.
    this.updatePlugin();
  }

  /**
   * Updates the plugin state after new dataset load.
   *
   * @private
   * @param {Array[]} sourceData Array of arrays or array of objects containing data.
   * @param {boolean} initialLoad Flag that determines whether the data has been loaded
   *                              during the initialization.
   */
  onAfterLoadData(sourceData, initialLoad) {
    if (!initialLoad) {
      this.updatePlugin();
    }
  }

  /**
   * Destroys the plugin instance.
   */
  destroy() {
    _classPrivateFieldSet(this, _stateManager, null);
    if (_classPrivateFieldGet(this, _hidingIndexMapObserver) !== null) {
      _classPrivateFieldGet(this, _hidingIndexMapObserver).unsubscribe();
      _classPrivateFieldSet(this, _hidingIndexMapObserver, null);
    }
    super.destroy();
  }

  /**
   * Gets the tree data that belongs to the column headers pointed by the passed coordinates.
   *
   * @private
   * @param {CellCoords} coords The CellCoords instance.
   * @returns {object|undefined}
   */
  _getHeaderTreeNodeDataByCoords(coords) {
    if (coords.row >= 0 || coords.col < 0) {
      return;
    }
    return _classPrivateFieldGet(this, _stateManager).getHeaderTreeNodeData(coords.row, coords.col);
  }
}
exports.NestedHeaders = NestedHeaders;