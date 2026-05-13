import "core-js/modules/es.error.cause.js";
import "core-js/modules/es.array.push.js";
function _classPrivateFieldInitSpec(obj, privateMap, value) { _checkPrivateRedeclaration(obj, privateMap); privateMap.set(obj, value); }
function _checkPrivateRedeclaration(obj, privateCollection) { if (privateCollection.has(obj)) { throw new TypeError("Cannot initialize the same private elements twice on an object"); } }
function _defineProperty(obj, key, value) { key = _toPropertyKey(key); if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _toPropertyKey(arg) { var key = _toPrimitive(arg, "string"); return typeof key === "symbol" ? key : String(key); }
function _toPrimitive(input, hint) { if (typeof input !== "object" || input === null) return input; var prim = input[Symbol.toPrimitive]; if (prim !== undefined) { var res = prim.call(input, hint || "default"); if (typeof res !== "object") return res; throw new TypeError("@@toPrimitive must return a primitive value."); } return (hint === "string" ? String : Number)(input); }
function _classPrivateFieldGet(receiver, privateMap) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "get"); return _classApplyDescriptorGet(receiver, descriptor); }
function _classApplyDescriptorGet(receiver, descriptor) { if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }
function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "set"); _classApplyDescriptorSet(receiver, descriptor, value); return value; }
function _classExtractFieldDescriptor(receiver, privateMap, action) { if (!privateMap.has(receiver)) { throw new TypeError("attempted to " + action + " private field on non-instance"); } return privateMap.get(receiver); }
function _classApplyDescriptorSet(receiver, descriptor, value) { if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } }
import { BasePlugin } from "../base/index.mjs";
import { arrayEach, arrayFilter, arrayUnique } from "../../helpers/array.mjs";
import { rangeEach } from "../../helpers/number.mjs";
import { warn } from "../../helpers/console.mjs";
import { addClass, hasClass, removeClass, fastInnerText } from "../../helpers/dom/element.mjs";
import EventManager from "../../eventManager.mjs";
import { stopImmediatePropagation } from "../../helpers/dom/event.mjs";
export const PLUGIN_KEY = 'collapsibleColumns';
export const PLUGIN_PRIORITY = 290;
const SETTING_KEYS = ['nestedHeaders'];
const COLLAPSIBLE_ELEMENT_CLASS = 'collapsibleIndicator';
const actionDictionary = new Map([['collapse', {
  hideColumn: true,
  beforeHook: 'beforeColumnCollapse',
  afterHook: 'afterColumnCollapse'
}], ['expand', {
  hideColumn: false,
  beforeHook: 'beforeColumnExpand',
  afterHook: 'afterColumnExpand'
}]]);

/* eslint-disable jsdoc/require-description-complete-sentence */

/**
 * @plugin CollapsibleColumns
 * @class CollapsibleColumns
 *
 * @description
 * The _CollapsibleColumns_ plugin allows collapsing of columns, covered by a header with the `colspan` property defined.
 *
 * Clicking the "collapse/expand" button collapses (or expands) all "child" headers except the first one.
 *
 * Setting the {@link Options#collapsiblecolumns} property to `true` will display a "collapse/expand" button in every header
 * with a defined `colspan` property.
 *
 * To limit this functionality to a smaller group of headers, define the `collapsibleColumns` property as an array
 * of objects, as in the example below.
 *
 * @example
 * ::: only-for javascript
 * ```js
 * const container = document.getElementById('example');
 * const hot = new Handsontable(container, {
 *   data: generateDataObj(),
 *   colHeaders: true,
 *   rowHeaders: true,
 *   nestedHeaders: true,
 *   // enable plugin
 *   collapsibleColumns: true,
 * });
 *
 * // or
 * const hot = new Handsontable(container, {
 *   data: generateDataObj(),
 *   colHeaders: true,
 *   rowHeaders: true,
 *   nestedHeaders: true,
 *   // enable and configure which columns can be collapsed
 *   collapsibleColumns: [
 *     {row: -4, col: 1, collapsible: true},
 *     {row: -3, col: 5, collapsible: true}
 *   ],
 * });
 * ```
 * :::
 *
 * ::: only-for react
 * ```jsx
 * <HotTable
 *   data={generateDataObj()}
 *   colHeaders={true}
 *   rowHeaders={true}
 *   nestedHeaders={true}
 *   // enable plugin
 *   collapsibleColumns={true}
 * />
 *
 * // or
 * <HotTable
 *   data={generateDataObj()}
 *   colHeaders={true}
 *   rowHeaders={true}
 *   nestedHeaders={true}
 *   // enable and configure which columns can be collapsed
 *   collapsibleColumns={[
 *     {row: -4, col: 1, collapsible: true},
 *     {row: -3, col: 5, collapsible: true}
 *   ]}
 * />
 * ```
 * :::
 */
var _collapsedColumnsMap = /*#__PURE__*/new WeakMap();
export class CollapsibleColumns extends BasePlugin {
  constructor() {
    super(...arguments);
    /**
     * Cached reference to the NestedHeaders plugin.
     *
     * @private
     * @type {NestedHeaders}
     */
    _defineProperty(this, "nestedHeadersPlugin", null);
    /**
     * Event manager instance reference.
     *
     * @private
     * @type {EventManager}
     */
    _defineProperty(this, "eventManager", new EventManager(this));
    /**
     * The NestedHeaders plugin StateManager instance.
     *
     * @private
     * @type {StateManager}
     */
    _defineProperty(this, "headerStateManager", null);
    /**
     * Map of collapsed columns by the plugin.
     *
     * @private
     * @type {HidingMap|null}
     */
    _classPrivateFieldInitSpec(this, _collapsedColumnsMap, {
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
  static get PLUGIN_DEPS() {
    return ['plugin:NestedHeaders'];
  }
  static get SETTING_KEYS() {
    return [PLUGIN_KEY, ...SETTING_KEYS];
  }
  /**
   * Checks if the plugin is enabled in the handsontable settings. This method is executed in {@link Hooks#beforeInit}
   * hook and if it returns `true` then the {@link CollapsibleColumns#enablePlugin} method is called.
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
    if (!nestedHeaders) {
      warn('You need to configure the Nested Headers plugin in order to use collapsible headers.');
    }
    _classPrivateFieldSet(this, _collapsedColumnsMap, this.hot.columnIndexMapper.createAndRegisterIndexMap(this.pluginName, 'hiding'));
    this.nestedHeadersPlugin = this.hot.getPlugin('nestedHeaders');
    this.headerStateManager = this.nestedHeadersPlugin.getStateManager();
    this.addHook('init', () => this.onInit());
    this.addHook('afterLoadData', function () {
      return _this.onAfterLoadData(...arguments);
    });
    this.addHook('afterGetColHeader', function () {
      return _this.onAfterGetColHeader(...arguments);
    });
    this.addHook('beforeOnCellMouseDown', (event, coords, TD) => this.onBeforeOnCellMouseDown(event, coords, TD));
    super.enablePlugin();
    // @TODO: Workaround for broken plugin initialization abstraction (#6806).
    this.updatePlugin();
  }

  /**
   * Updates the plugin's state.
   *
   * This method is executed when [`updateSettings()`](@/api/core.md#updatesettings) is invoked with any of the following configuration options:
   *   - [`collapsibleColumns`](@/api/options.md#collapsiblecolumns)
   *   - [`nestedHeaders`](@/api/options.md#nestedheaders)
   */
  updatePlugin() {
    // @TODO: Workaround for broken plugin initialization abstraction (#6806).
    if (!this.hot.view) {
      return;
    }
    if (!this.nestedHeadersPlugin.detectedOverlappedHeaders) {
      const {
        collapsibleColumns
      } = this.hot.getSettings();
      if (typeof collapsibleColumns === 'boolean') {
        // Add `collapsible: true` attribute to all headers with colspan higher than 1.
        this.headerStateManager.mapState(headerSettings => {
          return {
            collapsible: headerSettings.origColspan > 1
          };
        });
      } else if (Array.isArray(collapsibleColumns)) {
        this.headerStateManager.mapState(() => {
          return {
            collapsible: false
          };
        });
        this.headerStateManager.mergeStateWith(collapsibleColumns);
      }
    }
    super.updatePlugin();
  }

  /**
   * Disables the plugin functionality for this Handsontable instance.
   */
  disablePlugin() {
    this.hot.columnIndexMapper.unregisterMap(this.pluginName);
    _classPrivateFieldSet(this, _collapsedColumnsMap, null);
    this.nestedHeadersPlugin = null;
    this.clearButtons();
    super.disablePlugin();
  }

  /**
   * Clears the expand/collapse buttons.
   *
   * @private
   */
  clearButtons() {
    if (!this.hot.view) {
      return;
    }
    const headerLevels = this.hot.view._wt.getSetting('columnHeaders').length;
    const mainHeaders = this.hot.view._wt.wtTable.THEAD;
    const topHeaders = this.hot.view._wt.wtOverlays.topOverlay.clone.wtTable.THEAD;
    const topLeftCornerHeaders = this.hot.view._wt.wtOverlays.topInlineStartCornerOverlay ? this.hot.view._wt.wtOverlays.topInlineStartCornerOverlay.clone.wtTable.THEAD : null;
    const removeButton = function (button) {
      if (button) {
        button.parentNode.removeChild(button);
      }
    };
    rangeEach(0, headerLevels - 1, i => {
      const masterLevel = mainHeaders.childNodes[i];
      const topLevel = topHeaders.childNodes[i];
      const topLeftCornerLevel = topLeftCornerHeaders ? topLeftCornerHeaders.childNodes[i] : null;
      rangeEach(0, masterLevel.childNodes.length - 1, j => {
        let button = masterLevel.childNodes[j].querySelector(`.${COLLAPSIBLE_ELEMENT_CLASS}`);
        removeButton(button);
        if (topLevel && topLevel.childNodes[j]) {
          button = topLevel.childNodes[j].querySelector(`.${COLLAPSIBLE_ELEMENT_CLASS}`);
          removeButton(button);
        }
        if (topLeftCornerHeaders && topLeftCornerLevel && topLeftCornerLevel.childNodes[j]) {
          button = topLeftCornerLevel.childNodes[j].querySelector(`.${COLLAPSIBLE_ELEMENT_CLASS}`);
          removeButton(button);
        }
      });
    }, true);
  }

  /**
   * Expands section at the provided coords.
   *
   * @param {object} coords Contains coordinates information. (`coords.row`, `coords.col`).
   */
  expandSection(coords) {
    this.toggleCollapsibleSection([coords], 'expand');
  }

  /**
   * Collapses section at the provided coords.
   *
   * @param {object} coords Contains coordinates information. (`coords.row`, `coords.col`).
   */
  collapseSection(coords) {
    this.toggleCollapsibleSection([coords], 'collapse');
  }

  /**
   * Collapses or expand all collapsible sections, depending on the action parameter.
   *
   * @param {string} action 'collapse' or 'expand'.
   */
  toggleAllCollapsibleSections(action) {
    const coords = this.headerStateManager.mapNodes(headerSettings => {
      const {
        collapsible,
        origColspan,
        headerLevel,
        columnIndex,
        isCollapsed
      } = headerSettings;
      if (collapsible === true && origColspan > 1 && (isCollapsed && action === 'expand' || !isCollapsed && action === 'collapse')) {
        return {
          row: this.headerStateManager.levelToRowCoords(headerLevel),
          col: columnIndex
        };
      }
    });
    this.toggleCollapsibleSection(coords, action);
  }

  /**
   * Collapses all collapsible sections.
   */
  collapseAll() {
    this.toggleAllCollapsibleSections('collapse');
  }

  /**
   * Expands all collapsible sections.
   */
  expandAll() {
    this.toggleAllCollapsibleSections('expand');
  }

  /**
   * Collapses/Expands a section.
   *
   * @param {Array} coords Array of coords - section coordinates.
   * @param {string} [action] Action definition ('collapse' or 'expand').
   * @fires Hooks#beforeColumnCollapse
   * @fires Hooks#beforeColumnExpand
   * @fires Hooks#afterColumnCollapse
   * @fires Hooks#afterColumnExpand
   */
  toggleCollapsibleSection(coords, action) {
    if (!actionDictionary.has(action)) {
      throw new Error(`Unsupported action is passed (${action}).`);
    }
    if (!Array.isArray(coords)) {
      return;
    }

    // Ignore coordinates which points to the cells range.
    const filteredCoords = arrayFilter(coords, _ref => {
      let {
        row
      } = _ref;
      return row < 0;
    });
    let isActionPossible = filteredCoords.length > 0;
    arrayEach(filteredCoords, _ref2 => {
      var _this$headerStateMana;
      let {
        row,
        col: column
      } = _ref2;
      const {
        collapsible,
        isCollapsed
      } = (_this$headerStateMana = this.headerStateManager.getHeaderSettings(row, column)) !== null && _this$headerStateMana !== void 0 ? _this$headerStateMana : {};
      if (!collapsible || isCollapsed && action === 'collapse' || !isCollapsed && action === 'expand') {
        isActionPossible = false;
        return false;
      }
    });
    const nodeModRollbacks = [];
    const affectedColumnsIndexes = [];
    if (isActionPossible) {
      arrayEach(filteredCoords, _ref3 => {
        let {
          row,
          col: column
        } = _ref3;
        const {
          colspanCompensation,
          affectedColumns,
          rollbackModification
        } = this.headerStateManager.triggerNodeModification(action, row, column);
        if (colspanCompensation > 0) {
          affectedColumnsIndexes.push(...affectedColumns);
          nodeModRollbacks.push(rollbackModification);
        }
      });
    }
    const currentCollapsedColumns = this.getCollapsedColumns();
    let destinationCollapsedColumns = [];
    if (action === 'collapse') {
      destinationCollapsedColumns = arrayUnique([...currentCollapsedColumns, ...affectedColumnsIndexes]);
    } else if (action === 'expand') {
      destinationCollapsedColumns = arrayFilter(currentCollapsedColumns, index => !affectedColumnsIndexes.includes(index));
    }
    const actionTranslator = actionDictionary.get(action);
    const isActionAllowed = this.hot.runHooks(actionTranslator.beforeHook, currentCollapsedColumns, destinationCollapsedColumns, isActionPossible);
    if (isActionAllowed === false) {
      // Rollback all header nodes modification (collapse or expand).
      arrayEach(nodeModRollbacks, nodeModRollback => {
        nodeModRollback();
      });
      return;
    }
    this.hot.batchExecution(() => {
      arrayEach(affectedColumnsIndexes, visualColumn => {
        _classPrivateFieldGet(this, _collapsedColumnsMap).setValueAtIndex(this.hot.toPhysicalColumn(visualColumn), actionTranslator.hideColumn);
      });
    }, true);
    const isActionPerformed = this.getCollapsedColumns().length !== currentCollapsedColumns.length;
    this.hot.runHooks(actionTranslator.afterHook, currentCollapsedColumns, destinationCollapsedColumns, isActionPossible, isActionPerformed);
    this.hot.render();
    this.hot.view.adjustElementsSize(true);
  }

  /**
   * Gets an array of physical indexes of collapsed columns.
   *
   * @private
   * @returns {number[]}
   */
  getCollapsedColumns() {
    return _classPrivateFieldGet(this, _collapsedColumnsMap).getHiddenIndexes();
  }

  /**
   * Adds the indicator to the headers.
   *
   * @private
   * @param {number} column Column index.
   * @param {HTMLElement} TH TH element.
   * @param {number} headerLevel The index of header level counting from the top (positive
   *                             values counting from 0 to N).
   */
  onAfterGetColHeader(column, TH, headerLevel) {
    var _this$headerStateMana2;
    const {
      collapsible,
      origColspan,
      isCollapsed
    } = (_this$headerStateMana2 = this.headerStateManager.getHeaderSettings(headerLevel, column)) !== null && _this$headerStateMana2 !== void 0 ? _this$headerStateMana2 : {};
    const isNodeCollapsible = collapsible && origColspan > 1 && column >= this.hot.getSettings().fixedColumnsStart;
    let collapsibleElement = TH.querySelector(`.${COLLAPSIBLE_ELEMENT_CLASS}`);
    if (isNodeCollapsible) {
      if (!collapsibleElement) {
        collapsibleElement = this.hot.rootDocument.createElement('div');
        addClass(collapsibleElement, COLLAPSIBLE_ELEMENT_CLASS);
        TH.querySelector('div:first-child').appendChild(collapsibleElement);
      }
      removeClass(collapsibleElement, ['collapsed', 'expanded']);
      if (isCollapsed) {
        addClass(collapsibleElement, 'collapsed');
        fastInnerText(collapsibleElement, '+');
      } else {
        addClass(collapsibleElement, 'expanded');
        fastInnerText(collapsibleElement, '-');
      }
    } else {
      var _collapsibleElement;
      (_collapsibleElement = collapsibleElement) === null || _collapsibleElement === void 0 || _collapsibleElement.remove();
    }
  }

  /**
   * Indicator mouse event callback.
   *
   * @private
   * @param {object} event Mouse event.
   * @param {object} coords Event coordinates.
   */
  onBeforeOnCellMouseDown(event, coords) {
    if (hasClass(event.target, COLLAPSIBLE_ELEMENT_CLASS)) {
      if (hasClass(event.target, 'expanded')) {
        this.eventManager.fireEvent(event.target, 'mouseup');
        this.toggleCollapsibleSection([coords], 'collapse');
      } else if (hasClass(event.target, 'collapsed')) {
        this.eventManager.fireEvent(event.target, 'mouseup');
        this.toggleCollapsibleSection([coords], 'expand');
      }
      stopImmediatePropagation(event);
    }
  }

  /**
   * Updates the plugin state after HoT initialization.
   *
   * @private
   */
  onInit() {
    // @TODO: Workaround for broken plugin initialization abstraction (#6806).
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
    _classPrivateFieldSet(this, _collapsedColumnsMap, null);
    super.destroy();
  }
}