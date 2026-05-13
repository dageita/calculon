"use strict";

require("core-js/modules/es.error.cause.js");
exports.__esModule = true;
require("core-js/modules/es.json.stringify.js");
require("core-js/modules/es.array.push.js");
var _base = require("../base");
var _staticRegister = _interopRequireDefault(require("../../utils/staticRegister"));
var _console = require("../../helpers/console");
var _number = require("../../helpers/number");
var _mixed = require("../../helpers/mixed");
var _register = require("./engine/register");
var _utils = require("./utils");
var _settings = require("./engine/settings");
var _data = require("../../helpers/data");
var _string = require("../../helpers/string");
var _pluginHooks = _interopRequireDefault(require("../../pluginHooks"));
var _indexSyncer = _interopRequireDefault(require("./indexSyncer"));
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
const PLUGIN_KEY = 'formulas';
exports.PLUGIN_KEY = PLUGIN_KEY;
const SETTING_KEYS = ['maxRows', 'maxColumns', 'language'];
exports.SETTING_KEYS = SETTING_KEYS;
const PLUGIN_PRIORITY = 260;
exports.PLUGIN_PRIORITY = PLUGIN_PRIORITY;
_pluginHooks.default.getSingleton().register('afterNamedExpressionAdded');
_pluginHooks.default.getSingleton().register('afterNamedExpressionRemoved');
_pluginHooks.default.getSingleton().register('afterSheetAdded');
_pluginHooks.default.getSingleton().register('afterSheetRemoved');
_pluginHooks.default.getSingleton().register('afterSheetRenamed');
_pluginHooks.default.getSingleton().register('afterFormulasValuesUpdate');

// This function will be used for detecting changes coming from the `UndoRedo` plugin. This kind of change won't be
// handled by whole body of listeners and therefore won't change undo/redo stack inside engine provided by HyperFormula.
// HyperFormula's `undo` and `redo` methods will do it instead. Please keep in mind that undo/redo stacks inside
// instances of Handsontable and HyperFormula should be synced (number of actions should be the same).
const isBlockedSource = source => source === 'UndoRedo.undo' || source === 'UndoRedo.redo' || source === 'auto';

/**
 * This plugin allows you to perform Excel-like calculations in your business applications. It does it by an
 * integration with our other product, [HyperFormula](https://github.com/handsontable/hyperformula/), which is a
 * powerful calculation engine with an extensive number of features.
 *
 * To test out HyperFormula, see [this guide](@/guides/formulas/formula-calculation.md#available-functions).
 *
 * @plugin Formulas
 * @class Formulas
 */
var _internalOperationPending = /*#__PURE__*/new WeakMap();
var _hotWasInitializedWithEmptyData = /*#__PURE__*/new WeakMap();
var _engineListeners = /*#__PURE__*/new WeakMap();
class Formulas extends _base.BasePlugin {
  constructor() {
    var _this;
    super(...arguments);
    _this = this;
    /**
     * Flag used to bypass hooks in internal operations.
     *
     * @private
     * @type {boolean}
     */
    _classPrivateFieldInitSpec(this, _internalOperationPending, {
      writable: true,
      value: false
    });
    /**
     * Flag needed to mark if Handsontable was initialized with no data.
     * (Required to work around the fact, that Handsontable auto-generates sample data, when no data is provided).
     *
     * @type {boolean}
     */
    _classPrivateFieldInitSpec(this, _hotWasInitializedWithEmptyData, {
      writable: true,
      value: false
    });
    /**
     * The list of the HyperFormula listeners.
     *
     * @type {Array}
     */
    _classPrivateFieldInitSpec(this, _engineListeners, {
      writable: true,
      value: [['valuesUpdated', function () {
        return _this.onEngineValuesUpdated(...arguments);
      }], ['namedExpressionAdded', function () {
        return _this.onEngineNamedExpressionsAdded(...arguments);
      }], ['namedExpressionRemoved', function () {
        return _this.onEngineNamedExpressionsRemoved(...arguments);
      }], ['sheetAdded', function () {
        return _this.onEngineSheetAdded(...arguments);
      }], ['sheetRenamed', function () {
        return _this.onEngineSheetRenamed(...arguments);
      }], ['sheetRemoved', function () {
        return _this.onEngineSheetRemoved(...arguments);
      }]]
    });
    /**
     * Static register used to set up one global HyperFormula instance.
     * TODO: currently used in tests, might be removed later.
     *
     * @private
     * @type {object}
     */
    _defineProperty(this, "staticRegister", (0, _staticRegister.default)('formulas'));
    /**
     * The engine instance that will be used for this instance of Handsontable.
     *
     * @type {HyperFormula|null}
     */
    _defineProperty(this, "engine", null);
    /**
     * HyperFormula's sheet name.
     *
     * @type {string|null}
     */
    _defineProperty(this, "sheetName", null);
    /**
     * Index synchronizer responsible for manipulating with some general options related to indexes synchronization.
     *
     * @type {IndexSyncer|null}
     */
    _defineProperty(this, "indexSyncer", null);
    /**
     * Index synchronizer responsible for syncing the order of HOT and HF's data for the axis of the rows.
     *
     * @type {AxisSyncer|null}
     */
    _defineProperty(this, "rowAxisSyncer", null);
    /**
     * Index synchronizer responsible for syncing the order of HOT and HF's data for the axis of the columns.
     *
     * @type {AxisSyncer|null}
     */
    _defineProperty(this, "columnAxisSyncer", null);
  }
  static get PLUGIN_KEY() {
    return PLUGIN_KEY;
  }
  static get PLUGIN_PRIORITY() {
    return PLUGIN_PRIORITY;
  }
  static get SETTING_KEYS() {
    return [PLUGIN_KEY, ...SETTING_KEYS];
  }
  /**
   * HyperFormula's sheet id.
   *
   * @type {number|null}
   */
  get sheetId() {
    return this.sheetName === null ? null : this.engine.getSheetId(this.sheetName);
  }

  /**
   * Checks if the plugin is enabled in the handsontable settings. This method is executed in {@link Hooks#beforeInit}
   * hook and if it returns `true` then the {@link Formulas#enablePlugin} method is called.
   *
   * @returns {boolean}
   */
  isEnabled() {
    /* eslint-disable no-unneeded-ternary */
    return this.hot.getSettings()[PLUGIN_KEY] ? true : false;
  }

  /**
   * Enables the plugin functionality for this Handsontable instance.
   */
  enablePlugin() {
    var _setupEngine,
      _this2 = this;
    if (this.enabled) {
      return;
    }
    this.engine = (_setupEngine = (0, _register.setupEngine)(this.hot)) !== null && _setupEngine !== void 0 ? _setupEngine : this.engine;
    if (!this.engine) {
      (0, _console.warn)('Missing the required `engine` key in the Formulas settings. Please fill it with either an' + ' engine class or an engine instance.');
      return;
    }

    // Useful for disabling -> enabling the plugin using `updateSettings` or the API.
    if (this.sheetName !== null && !this.engine.doesSheetExist(this.sheetName)) {
      const newSheetName = this.addSheet(this.sheetName, this.hot.getSourceDataArray());
      if (newSheetName !== false) {
        this.sheetName = newSheetName;
      }
    }
    this.addHook('beforeLoadData', function () {
      return _this2.onBeforeLoadData(...arguments);
    });
    this.addHook('afterLoadData', function () {
      return _this2.onAfterLoadData(...arguments);
    });

    // The `updateData` hooks utilize the same logic as the `loadData` hooks.
    this.addHook('beforeUpdateData', function () {
      return _this2.onBeforeLoadData(...arguments);
    });
    this.addHook('afterUpdateData', function () {
      return _this2.onAfterLoadData(...arguments);
    });
    this.addHook('modifyData', function () {
      return _this2.onModifyData(...arguments);
    });
    this.addHook('modifySourceData', function () {
      return _this2.onModifySourceData(...arguments);
    });
    this.addHook('beforeValidate', function () {
      return _this2.onBeforeValidate(...arguments);
    });
    this.addHook('afterSetSourceDataAtCell', function () {
      return _this2.onAfterSetSourceDataAtCell(...arguments);
    });
    this.addHook('afterSetDataAtCell', function () {
      return _this2.onAfterSetDataAtCell(...arguments);
    });
    this.addHook('afterSetDataAtRowProp', function () {
      return _this2.onAfterSetDataAtCell(...arguments);
    });
    this.addHook('beforeCreateRow', function () {
      return _this2.onBeforeCreateRow(...arguments);
    });
    this.addHook('beforeCreateCol', function () {
      return _this2.onBeforeCreateCol(...arguments);
    });
    this.addHook('afterCreateRow', function () {
      return _this2.onAfterCreateRow(...arguments);
    });
    this.addHook('afterCreateCol', function () {
      return _this2.onAfterCreateCol(...arguments);
    });
    this.addHook('beforeRemoveRow', function () {
      return _this2.onBeforeRemoveRow(...arguments);
    });
    this.addHook('beforeRemoveCol', function () {
      return _this2.onBeforeRemoveCol(...arguments);
    });
    this.addHook('afterRemoveRow', function () {
      return _this2.onAfterRemoveRow(...arguments);
    });
    this.addHook('afterRemoveCol', function () {
      return _this2.onAfterRemoveCol(...arguments);
    });
    this.indexSyncer = new _indexSyncer.default(this.hot.rowIndexMapper, this.hot.columnIndexMapper, postponedAction => {
      this.hot.addHookOnce('init', () => {
        // Engine is initialized after executing callback to `afterLoadData` hook. Thus, some actions on indexes should
        // be postponed.
        postponedAction();
      });
    });
    this.rowAxisSyncer = this.indexSyncer.getForAxis('row');
    this.columnAxisSyncer = this.indexSyncer.getForAxis('column');
    this.hot.addHook('afterRowSequenceChange', this.rowAxisSyncer.getIndexesChangeSyncMethod());
    this.hot.addHook('afterColumnSequenceChange', this.columnAxisSyncer.getIndexesChangeSyncMethod());
    this.hot.addHook('beforeRowMove', (movedRows, finalIndex, _, movePossible) => {
      this.rowAxisSyncer.storeMovesInformation(movedRows, finalIndex, movePossible);
    });
    this.hot.addHook('beforeColumnMove', (movedColumns, finalIndex, _, movePossible) => {
      this.columnAxisSyncer.storeMovesInformation(movedColumns, finalIndex, movePossible);
    });
    this.hot.addHook('afterRowMove', (movedRows, finalIndex, dropIndex, movePossible, orderChanged) => {
      this.rowAxisSyncer.calculateAndSyncMoves(movePossible, orderChanged);
    });
    this.hot.addHook('afterColumnMove', (movedColumns, finalIndex, dropIndex, movePossible, orderChanged) => {
      this.columnAxisSyncer.calculateAndSyncMoves(movePossible, orderChanged);
    });
    this.hot.addHook('beforeColumnFreeze', (column, freezePerformed) => {
      this.columnAxisSyncer.storeMovesInformation([column], this.hot.getSettings().fixedColumnsStart, freezePerformed);
    });
    this.hot.addHook('afterColumnFreeze', (_, freezePerformed) => {
      this.columnAxisSyncer.calculateAndSyncMoves(freezePerformed, freezePerformed);
    });
    this.hot.addHook('beforeColumnUnfreeze', (column, unfreezePerformed) => {
      this.columnAxisSyncer.storeMovesInformation([column], this.hot.getSettings().fixedColumnsStart - 1, unfreezePerformed);
    });
    this.hot.addHook('afterColumnUnfreeze', (_, unfreezePerformed) => {
      this.columnAxisSyncer.calculateAndSyncMoves(unfreezePerformed, unfreezePerformed);
    });

    // TODO: Actions related to overwriting dates from HOT format to HF default format are done as callback to this
    // hook, because some hooks, such as `afterLoadData` doesn't have information about composed cell properties.
    // Another hooks are triggered to late for setting HF's engine data needed for some actions.
    this.addHook('afterCellMetaReset', function () {
      return _this2.onAfterCellMetaReset(...arguments);
    });

    // Handling undo actions on data just using HyperFormula's UndoRedo mechanism
    this.addHook('beforeUndo', () => {
      this.indexSyncer.setPerformUndo(true);
      this.engine.undo();
    });

    // Handling redo actions on data just using HyperFormula's UndoRedo mechanism
    this.addHook('beforeRedo', () => {
      this.indexSyncer.setPerformRedo(true);
      this.engine.redo();
    });
    this.addHook('afterUndo', () => {
      this.indexSyncer.setPerformUndo(false);
    });
    this.addHook('afterUndo', () => {
      this.indexSyncer.setPerformRedo(false);
    });
    this.addHook('afterDetachChild', function () {
      return _this2.onAfterDetachChild(...arguments);
    });
    this.addHook('beforeAutofill', function () {
      return _this2.onBeforeAutofill(...arguments);
    });
    _classPrivateFieldGet(this, _engineListeners).forEach(_ref => {
      let [eventName, listener] = _ref;
      return this.engine.on(eventName, listener);
    });
    super.enablePlugin();
  }

  /**
   * Disables the plugin functionality for this Handsontable instance.
   */
  disablePlugin() {
    _classPrivateFieldGet(this, _engineListeners).forEach(_ref2 => {
      let [eventName, listener] = _ref2;
      return this.engine.off(eventName, listener);
    });
    (0, _register.unregisterEngine)(this.engine, this.hot);
    this.engine = null;
    super.disablePlugin();
  }

  /**
   * Triggered on `updateSettings`.
   *
   * @private
   * @param {object} newSettings New set of settings passed to the `updateSettings` method.
   */
  updatePlugin(newSettings) {
    const newEngineSettings = (0, _settings.getEngineSettingsWithOverrides)(this.hot.getSettings());
    if ((0, _settings.haveEngineSettingsChanged)(this.engine.getConfig(), newEngineSettings)) {
      this.engine.updateConfig(newEngineSettings);
    }
    const pluginSettings = this.hot.getSettings()[PLUGIN_KEY];
    if ((0, _mixed.isDefined)(pluginSettings) && (0, _mixed.isDefined)(pluginSettings.sheetName) && pluginSettings.sheetName !== this.sheetName) {
      this.switchSheet(pluginSettings.sheetName);
    }

    // If no data was passed to the `updateSettings` method and no sheet is connected to the instance -> create a
    // new sheet using the currently used data. Otherwise, it will be handled by the `afterLoadData` call.
    if (!newSettings.data && this.sheetName === null) {
      const sheetName = this.hot.getSettings()[PLUGIN_KEY].sheetName;
      if (sheetName && this.engine.doesSheetExist(sheetName)) {
        this.switchSheet(this.sheetName);
      } else {
        this.sheetName = this.addSheet(sheetName !== null && sheetName !== void 0 ? sheetName : void 0, this.hot.getSourceDataArray());
      }
    }
    super.updatePlugin(newSettings);
  }

  /**
   * Destroys the plugin instance.
   */
  destroy() {
    _classPrivateFieldGet(this, _engineListeners).forEach(_ref3 => {
      var _this$engine;
      let [eventName, listener] = _ref3;
      return (_this$engine = this.engine) === null || _this$engine === void 0 ? void 0 : _this$engine.off(eventName, listener);
    });
    _classPrivateFieldSet(this, _engineListeners, null);
    (0, _register.unregisterEngine)(this.engine, this.hot);
    this.engine = null;
    super.destroy();
  }

  /**
   * Add a sheet to the shared HyperFormula instance.
   *
   * @param {string|null} [sheetName] The new sheet name. If not provided (or a null is passed), will be
   * auto-generated by HyperFormula.
   * @param {Array} [sheetData] Data passed to the shared HyperFormula instance. Has to be declared as an array of
   * arrays - array of objects is not supported in this scenario.
   * @returns {boolean|string} `false` if the data format is unusable or it is impossible to add a new sheet to the
   * engine, the created sheet name otherwise.
   */
  addSheet(sheetName, sheetData) {
    if ((0, _mixed.isDefined)(sheetData) && !(0, _data.isArrayOfArrays)(sheetData)) {
      (0, _console.warn)('The provided data should be an array of arrays.');
      return false;
    }
    if (sheetName !== void 0 && sheetName !== null && this.engine.doesSheetExist(sheetName)) {
      (0, _console.warn)('Sheet with the provided name already exists.');
      return false;
    }
    try {
      const actualSheetName = this.engine.addSheet(sheetName !== null && sheetName !== void 0 ? sheetName : void 0);
      if (sheetData) {
        this.engine.setSheetContent(this.engine.getSheetId(actualSheetName), sheetData);
      }
      return actualSheetName;
    } catch (e) {
      (0, _console.warn)(e.message);
      return false;
    }
  }

  /**
   * Switch the sheet used as data in the Handsontable instance (it loads the data from the shared HyperFormula
   * instance).
   *
   * @param {string} sheetName Sheet name used in the shared HyperFormula instance.
   */
  switchSheet(sheetName) {
    if (!this.engine.doesSheetExist(sheetName)) {
      (0, _console.error)(`The sheet named \`${sheetName}\` does not exist, switch aborted.`);
      return;
    }
    this.sheetName = sheetName;
    const serialized = this.engine.getSheetSerialized(this.sheetId);
    if (serialized.length > 0) {
      this.hot.loadData(serialized, `${(0, _string.toUpperCaseFirst)(PLUGIN_KEY)}.switchSheet`);
    }
  }

  /**
   * Get the cell type under specified visual coordinates.
   *
   * @param {number} row Visual row index.
   * @param {number} column Visual column index.
   * @param {number} [sheet] The target sheet id, defaults to the current sheet.
   * @returns {string} Possible values: 'FORMULA' | 'VALUE' | 'ARRAYFORMULA' | 'EMPTY'.
   */
  getCellType(row, column) {
    let sheet = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : this.sheetId;
    const physicalRow = this.hot.toPhysicalRow(row);
    const physicalColumn = this.hot.toPhysicalColumn(column);
    if (physicalRow !== null && physicalColumn !== null) {
      return this.engine.getCellType({
        sheet,
        row: this.rowAxisSyncer.getHfIndexFromVisualIndex(row),
        col: this.columnAxisSyncer.getHfIndexFromVisualIndex(column)
      });
    } else {
      // Should return `EMPTY` when out of bounds (according to the test cases).
      return 'EMPTY';
    }
  }

  /**
   * Returns `true` if under specified visual coordinates is formula.
   *
   * @param {number} row Visual row index.
   * @param {number} column Visual column index.
   * @param {number} [sheet] The target sheet id, defaults to the current sheet.
   * @returns {boolean}
   */
  isFormulaCellType(row, column) {
    let sheet = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : this.sheetId;
    return this.engine.doesCellHaveFormula({
      sheet,
      row: this.rowAxisSyncer.getHfIndexFromVisualIndex(row),
      col: this.columnAxisSyncer.getHfIndexFromVisualIndex(column)
    });
  }

  /**
   * Renders dependent sheets (handsontable instances) based on the changes - list of the
   * recalculated dependent cells.
   *
   * @private
   * @param {object[]} dependentCells The values and location of applied changes within HF engine.
   * @param {boolean} [renderSelf] `true` if it's supposed to render itself, `false` otherwise.
   */
  renderDependentSheets(dependentCells) {
    let renderSelf = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
    const affectedSheetIds = new Set();
    dependentCells.forEach(change => {
      var _change$address;
      // For the Named expression the address is empty, hence the `sheetId` is undefined.
      const sheetId = change === null || change === void 0 || (_change$address = change.address) === null || _change$address === void 0 ? void 0 : _change$address.sheet;
      if (sheetId !== void 0) {
        if (!affectedSheetIds.has(sheetId)) {
          affectedSheetIds.add(sheetId);
        }
      }
    });
    (0, _register.getRegisteredHotInstances)(this.engine).forEach((relatedHot, sheetId) => {
      if ((renderSelf || sheetId !== this.sheetId) && affectedSheetIds.has(sheetId)) {
        var _relatedHot$view;
        relatedHot.render();
        (_relatedHot$view = relatedHot.view) === null || _relatedHot$view === void 0 || _relatedHot$view.adjustElementsSize();
      }
    });
  }

  /**
   * Validates dependent cells based on the cells that are modified by the change.
   *
   * @private
   * @param {object[]} dependentCells The values and location of applied changes within HF engine.
   * @param {object[]} [changedCells] The values and location of applied changes by developer (through API or UI).
   */
  validateDependentCells(dependentCells) {
    let changedCells = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : [];
    const stringifyAddress = change => {
      var _change$address2;
      const {
        row,
        col,
        sheet
      } = (_change$address2 = change === null || change === void 0 ? void 0 : change.address) !== null && _change$address2 !== void 0 ? _change$address2 : {};
      return (0, _mixed.isDefined)(sheet) ? `${sheet}:${row}x${col}` : '';
    };
    const changedCellsSet = new Set(changedCells.map(change => stringifyAddress(change)));
    dependentCells.forEach(change => {
      var _change$address3, _change$address4;
      const {
        row,
        col
      } = (_change$address3 = change.address) !== null && _change$address3 !== void 0 ? _change$address3 : {};

      // Don't try to validate cells outside of the visual part of the table.
      if ((0, _mixed.isDefined)(row) === false || (0, _mixed.isDefined)(col) === false || row >= this.hot.countRows() || col >= this.hot.countCols()) {
        return;
      }

      // For the Named expression the address is empty, hence the `sheetId` is undefined.
      const sheetId = change === null || change === void 0 || (_change$address4 = change.address) === null || _change$address4 === void 0 ? void 0 : _change$address4.sheet;
      const addressId = stringifyAddress(change);

      // Validate the cells that depend on the calculated formulas. Skip that cells
      // where the user directly changes the values - the Core triggers those validators.
      if (sheetId !== void 0 && !changedCellsSet.has(addressId)) {
        const boundHot = (0, _register.getRegisteredHotInstances)(this.engine).get(sheetId);

        // if `sheetId` is not bound to any Handsontable instance, skip the validation process
        if (!boundHot) {
          return;
        }

        // It will just re-render certain cell when necessary.
        boundHot.validateCell(boundHot.getDataAtCell(row, col), boundHot.getCellMeta(row, col), () => {});
      }
    });
  }

  /**
   * Sync a change from the change-related hooks with the engine.
   *
   * @private
   * @param {number} row Visual row index.
   * @param {number} column Visual column index.
   * @param {Handsontable.CellValue} newValue New value.
   * @returns {Array} Array of changes exported from the engine.
   */
  syncChangeWithEngine(row, column, newValue) {
    const address = {
      row: this.rowAxisSyncer.getHfIndexFromVisualIndex(row),
      col: this.columnAxisSyncer.getHfIndexFromVisualIndex(column),
      sheet: this.sheetId
    };
    if (!this.engine.isItPossibleToSetCellContents(address)) {
      (0, _console.warn)(`Not possible to set cell data at ${JSON.stringify(address)}`);
      return;
    }
    const cellMeta = this.hot.getCellMeta(row, column);
    if ((0, _utils.isDate)(newValue, cellMeta.type)) {
      if ((0, _utils.isDateValid)(newValue, cellMeta.dateFormat)) {
        // Rewriting date in HOT format to HF format.
        newValue = (0, _utils.getDateInHfFormat)(newValue, cellMeta.dateFormat);
      } else if ((0, _utils.isFormula)(newValue) === false) {
        // Escaping value from date parsing using "'" sign (HF feature).
        newValue = `'${newValue}`;
      }
    }
    return this.engine.setCellContents(address, newValue);
  }

  /**
   * The hook allows to translate the formula value to calculated value before it goes to the
   * validator function.
   *
   * @private
   * @param {*} value The cell value to validate.
   * @param {number} visualRow The visual row index.
   * @param {number|string} prop The visual column index or property name of the column.
   * @returns {*} Returns value to validate.
   */
  onBeforeValidate(value, visualRow, prop) {
    const visualColumn = this.hot.propToCol(prop);
    if (this.isFormulaCellType(visualRow, visualColumn)) {
      const address = {
        row: this.rowAxisSyncer.getHfIndexFromVisualIndex(visualRow),
        col: this.columnAxisSyncer.getHfIndexFromVisualIndex(visualColumn),
        sheet: this.sheetId
      };
      const cellMeta = this.hot.getCellMeta(visualRow, visualColumn);
      let cellValue = this.engine.getCellValue(address); // Date as an integer (Excel-like date).

      if (cellMeta.type === 'date' && (0, _number.isNumeric)(cellValue)) {
        cellValue = (0, _utils.getDateFromExcelDate)(cellValue, cellMeta.dateFormat);
      }

      // If `cellValue` is an object it is expected to be an error
      return typeof cellValue === 'object' && cellValue !== null ? cellValue.value : cellValue;
    }
    return value;
  }

  /**
   * `onBeforeAutofill` hook callback.
   *
   * @private
   * @param {Array[]} fillData The data that was used to fill the `targetRange`. If `beforeAutofill` was used
   * and returned `[[]]`, this will be the same object that was returned from `beforeAutofill`.
   * @param {CellRange} sourceRange The range values will be filled from.
   * @param {CellRange} targetRange The range new values will be filled into.
   * @returns {boolean|*}
   */
  onBeforeAutofill(fillData, sourceRange, targetRange) {
    const {
      row: sourceTopStartRow,
      col: sourceTopStartColumn
    } = sourceRange.getTopStartCorner();
    const {
      row: sourceBottomEndRow,
      col: sourceBottomEndColumn
    } = sourceRange.getBottomEndCorner();
    const {
      row: targetTopStartRow,
      col: targetTopStartColumn
    } = targetRange.getTopStartCorner();
    const {
      row: targetBottomEndRow,
      col: targetBottomEndColumn
    } = targetRange.getBottomEndCorner();
    const engineSourceRange = {
      start: {
        row: this.rowAxisSyncer.getHfIndexFromVisualIndex(sourceTopStartRow),
        col: this.columnAxisSyncer.getHfIndexFromVisualIndex(sourceTopStartColumn),
        sheet: this.sheetId
      },
      end: {
        row: this.rowAxisSyncer.getHfIndexFromVisualIndex(sourceBottomEndRow),
        col: this.columnAxisSyncer.getHfIndexFromVisualIndex(sourceBottomEndColumn),
        sheet: this.sheetId
      }
    };
    const engineTargetRange = {
      start: {
        row: this.rowAxisSyncer.getHfIndexFromVisualIndex(targetTopStartRow),
        col: this.columnAxisSyncer.getHfIndexFromVisualIndex(targetTopStartColumn),
        sheet: this.sheetId
      },
      end: {
        row: this.rowAxisSyncer.getHfIndexFromVisualIndex(targetBottomEndRow),
        col: this.columnAxisSyncer.getHfIndexFromVisualIndex(targetBottomEndColumn),
        sheet: this.sheetId
      }
    };

    // Blocks the autofill operation if HyperFormula says that at least one of
    // the underlying cell's contents cannot be set.
    if (this.engine.isItPossibleToSetCellContents(engineTargetRange) === false) {
      return false;
    }
    const fillRangeData = this.engine.getFillRangeData(engineSourceRange, engineTargetRange);
    const {
      row: sourceStartRow,
      col: sourceStartColumn
    } = engineSourceRange.start;
    const {
      row: sourceEndRow,
      col: sourceEndColumn
    } = engineSourceRange.end;
    const populationRowLength = sourceEndRow - sourceStartRow + 1;
    const populationColumnLength = sourceEndColumn - sourceStartColumn + 1;
    for (let populatedRowIndex = 0; populatedRowIndex < fillRangeData.length; populatedRowIndex += 1) {
      for (let populatedColumnIndex = 0; populatedColumnIndex < fillRangeData[populatedRowIndex].length; populatedColumnIndex += 1) {
        const populatedValue = fillRangeData[populatedRowIndex][populatedColumnIndex];
        const sourceRow = populatedRowIndex % populationRowLength;
        const sourceColumn = populatedColumnIndex % populationColumnLength;
        const sourceCellMeta = this.hot.getCellMeta(sourceRow, sourceColumn);
        if ((0, _utils.isDate)(populatedValue, sourceCellMeta.type)) {
          if (populatedValue.startsWith('\'')) {
            // Populating values on HOT side without apostrophe.
            fillRangeData[populatedRowIndex][populatedColumnIndex] = populatedValue.slice(1);
          } else if (this.isFormulaCellType(sourceRow, sourceColumn, this.sheetId) === false) {
            // Populating date in proper format, coming from the source cell.
            fillRangeData[populatedRowIndex][populatedColumnIndex] = (0, _utils.getDateInHotFormat)(populatedValue, sourceCellMeta.dateFormat);
          }
        }
      }
    }
    return fillRangeData;
  }

  /**
   * `beforeLoadData` hook callback.
   *
   * @param {Array} sourceData Array of arrays or array of objects containing data.
   * @param {boolean} initialLoad Flag that determines whether the data has been loaded during the initialization.
   * @param {string} [source] Source of the call.
   * @private
   */
  onBeforeLoadData(sourceData, initialLoad) {
    let source = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : '';
    if (source.includes((0, _string.toUpperCaseFirst)(PLUGIN_KEY))) {
      return;
    }

    // This flag needs to be defined, because not passing data to HOT results in HOT auto-generating a `null`-filled
    // initial dataset.
    _classPrivateFieldSet(this, _hotWasInitializedWithEmptyData, (0, _mixed.isUndefined)(this.hot.getSettings().data));
  }

  /**
   * Callback to `afterCellMetaReset` hook which is triggered after setting cell meta.
   *
   * @private
   */
  onAfterCellMetaReset() {
    const sourceDataArray = this.hot.getSourceDataArray();
    let valueChanged = false;
    sourceDataArray.forEach((rowData, rowIndex) => {
      rowData.forEach((cellValue, columnIndex) => {
        const cellMeta = this.hot.getCellMeta(rowIndex, columnIndex);
        const dateFormat = cellMeta.dateFormat;
        if ((0, _utils.isDate)(cellValue, cellMeta.type)) {
          valueChanged = true;
          if ((0, _utils.isDateValid)(cellValue, dateFormat)) {
            // Rewriting date in HOT format to HF format.
            sourceDataArray[rowIndex][columnIndex] = (0, _utils.getDateInHfFormat)(cellValue, dateFormat);
          } else if (this.isFormulaCellType(rowIndex, columnIndex) === false) {
            // Escaping value from date parsing using "'" sign (HF feature).
            sourceDataArray[rowIndex][columnIndex] = `'${cellValue}`;
          }
        }
      });
    });
    if (valueChanged === true) {
      _classPrivateFieldSet(this, _internalOperationPending, true);
      this.engine.setSheetContent(this.sheetId, sourceDataArray);
      _classPrivateFieldSet(this, _internalOperationPending, false);
    }
  }

  /**
   * `afterLoadData` hook callback.
   *
   * @param {Array} sourceData Array of arrays or array of objects containing data.
   * @param {boolean} initialLoad Flag that determines whether the data has been loaded during the initialization.
   * @param {string} [source] Source of the call.
   * @private
   */
  onAfterLoadData(sourceData, initialLoad) {
    let source = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : '';
    if (source.includes((0, _string.toUpperCaseFirst)(PLUGIN_KEY))) {
      return;
    }
    this.sheetName = (0, _register.setupSheet)(this.engine, this.hot.getSettings()[PLUGIN_KEY].sheetName);
    if (!_classPrivateFieldGet(this, _hotWasInitializedWithEmptyData)) {
      const sourceDataArray = this.hot.getSourceDataArray();
      if (this.engine.isItPossibleToReplaceSheetContent(this.sheetId, sourceDataArray)) {
        _classPrivateFieldSet(this, _internalOperationPending, true);
        const dependentCells = this.engine.setSheetContent(this.sheetId, sourceDataArray);
        this.indexSyncer.setupSyncEndpoint(this.engine, this.sheetId);
        this.renderDependentSheets(dependentCells);
        _classPrivateFieldSet(this, _internalOperationPending, false);
      }
    } else {
      this.switchSheet(this.sheetName);
    }
  }

  /**
   * `modifyData` hook callback.
   *
   * @private
   * @param {number} physicalRow Physical row index.
   * @param {number} visualColumn Visual column index.
   * @param {object} valueHolder Object which contains original value which can be modified by overwriting `.value`
   *   property.
   * @param {string} ioMode String which indicates for what operation hook is fired (`get` or `set`).
   */
  onModifyData(physicalRow, visualColumn, valueHolder, ioMode) {
    if (ioMode !== 'get' || _classPrivateFieldGet(this, _internalOperationPending) || this.sheetName === null || !this.engine.doesSheetExist(this.sheetName)) {
      return;
    }
    const visualRow = this.hot.toVisualRow(physicalRow);
    if (visualRow === null || visualColumn === null) {
      return;
    }

    // `column` is here as visual index because of inconsistencies related to hook execution in `src/dataMap`.
    const isFormulaCellType = this.isFormulaCellType(visualRow, visualColumn);
    if (!isFormulaCellType) {
      const cellType = this.getCellType(visualRow, visualColumn);
      if (cellType !== 'ARRAY') {
        if ((0, _utils.isEscapedFormulaExpression)(valueHolder.value)) {
          valueHolder.value = (0, _utils.unescapeFormulaExpression)(valueHolder.value);
        }
        return;
      }
    }
    const address = {
      row: this.rowAxisSyncer.getHfIndexFromVisualIndex(visualRow),
      col: this.columnAxisSyncer.getHfIndexFromVisualIndex(visualColumn),
      sheet: this.sheetId
    };
    let cellValue = this.engine.getCellValue(address); // Date as an integer (Excel like date).
    const cellMeta = this.hot.getCellMeta(visualRow, visualColumn);
    if (cellMeta.type === 'date' && (0, _number.isNumeric)(cellValue)) {
      cellValue = (0, _utils.getDateFromExcelDate)(cellValue, cellMeta.dateFormat);
    }

    // If `cellValue` is an object it is expected to be an error
    const value = typeof cellValue === 'object' && cellValue !== null ? cellValue.value : cellValue;
    valueHolder.value = value;
  }

  /**
   * `modifySourceData` hook callback.
   *
   * @private
   * @param {number} row Physical row index.
   * @param {number|string} columnOrProp Physical column index or prop.
   * @param {object} valueHolder Object which contains original value which can be modified by overwriting `.value`
   *   property.
   * @param {string} ioMode String which indicates for what operation hook is fired (`get` or `set`).
   */
  onModifySourceData(row, columnOrProp, valueHolder, ioMode) {
    if (ioMode !== 'get' || _classPrivateFieldGet(this, _internalOperationPending) || this.sheetName === null || !this.engine.doesSheetExist(this.sheetName)) {
      return;
    }
    const visualRow = this.hot.toVisualRow(row);
    const visualColumn = this.hot.propToCol(columnOrProp);
    if (visualRow === null || visualColumn === null) {
      return;
    }

    // `column` is here as visual index because of inconsistencies related to hook execution in `src/dataMap`.
    const isFormulaCellType = this.isFormulaCellType(visualRow, visualColumn);
    if (!isFormulaCellType) {
      const cellType = this.getCellType(visualRow, visualColumn);
      if (cellType !== 'ARRAY') {
        return;
      }
    }
    const dimensions = this.engine.getSheetDimensions(this.engine.getSheetId(this.sheetName));

    // Don't actually change the source data if HyperFormula is not
    // initialized yet. This is done to allow the `afterLoadData` hook to
    // load the existing source data with `Handsontable#getSourceDataArray`
    // properly.
    if (dimensions.width === 0 && dimensions.height === 0) {
      return;
    }
    const address = {
      row: this.rowAxisSyncer.getHfIndexFromVisualIndex(visualRow),
      col: this.columnAxisSyncer.getHfIndexFromVisualIndex(visualColumn),
      sheet: this.sheetId
    };
    valueHolder.value = this.engine.getCellSerialized(address);
  }

  /**
   * `onAfterSetDataAtCell` hook callback.
   *
   * @private
   * @param {Array[]} changes An array of changes in format [[row, prop, oldValue, value], ...].
   * @param {string} [source] String that identifies source of hook call
   *                          ([list of all available sources]{@link https://handsontable.com/docs/javascript-data-grid/events-and-hooks/#handsontable-hooks}).
   */
  onAfterSetDataAtCell(changes, source) {
    if (isBlockedSource(source)) {
      return;
    }
    const outOfBoundsChanges = [];
    const changedCells = [];
    const dependentCells = this.engine.batch(() => {
      changes.forEach(_ref4 => {
        let [visualRow, prop,, newValue] = _ref4;
        const visualColumn = this.hot.propToCol(prop);
        const physicalRow = this.hot.toPhysicalRow(visualRow);
        const physicalColumn = this.hot.toPhysicalColumn(visualColumn);
        const address = {
          row: this.rowAxisSyncer.getHfIndexFromVisualIndex(visualRow),
          col: this.columnAxisSyncer.getHfIndexFromVisualIndex(visualColumn),
          sheet: this.sheetId
        };
        if (physicalRow !== null && physicalColumn !== null) {
          this.syncChangeWithEngine(visualRow, visualColumn, newValue);
        } else {
          outOfBoundsChanges.push([visualRow, visualColumn, newValue]);
        }
        changedCells.push({
          address
        });
      });
    });
    if (outOfBoundsChanges.length) {
      // Workaround for rows/columns being created two times (by HOT and the engine).
      // (unfortunately, this requires an extra re-render)
      this.hot.addHookOnce('afterChange', () => {
        const outOfBoundsDependentCells = this.engine.batch(() => {
          outOfBoundsChanges.forEach(_ref5 => {
            let [row, column, newValue] = _ref5;
            this.syncChangeWithEngine(row, column, newValue);
          });
        });
        this.renderDependentSheets(outOfBoundsDependentCells, true);
      });
    }
    this.renderDependentSheets(dependentCells);
    this.validateDependentCells(dependentCells, changedCells);
  }

  /**
   * `onAfterSetSourceDataAtCell` hook callback.
   *
   * @private
   * @param {Array[]} changes An array of changes in format [[row, column, oldValue, value], ...].
   * @param {string} [source] String that identifies source of hook call
   *                          ([list of all available sources]{@link https://handsontable.com/docs/javascript-data-grid/events-and-hooks/#handsontable-hooks}).
   */
  onAfterSetSourceDataAtCell(changes, source) {
    if (isBlockedSource(source)) {
      return;
    }
    const dependentCells = [];
    const changedCells = [];
    changes.forEach(_ref6 => {
      let [visualRow, prop,, newValue] = _ref6;
      const visualColumn = this.hot.propToCol(prop);
      if (!(0, _number.isNumeric)(visualColumn)) {
        return;
      }
      const address = {
        row: this.rowAxisSyncer.getHfIndexFromVisualIndex(visualRow),
        col: this.columnAxisSyncer.getHfIndexFromVisualIndex(visualColumn),
        sheet: this.sheetId
      };
      if (!this.engine.isItPossibleToSetCellContents(address)) {
        (0, _console.warn)(`Not possible to set source cell data at ${JSON.stringify(address)}`);
        return;
      }
      changedCells.push({
        address
      });
      dependentCells.push(...this.engine.setCellContents(address, newValue));
    });
    this.renderDependentSheets(dependentCells);
    this.validateDependentCells(dependentCells, changedCells);
  }

  /**
   * `beforeCreateRow` hook callback.
   *
   * @private
   * @param {number} visualRow Represents the visual index of first newly created row in the data source array.
   * @param {number} amount Number of newly created rows in the data source array.
   * @returns {*|boolean} If false is returned the action is canceled.
   */
  onBeforeCreateRow(visualRow, amount) {
    let hfRowIndex = this.rowAxisSyncer.getHfIndexFromVisualIndex(visualRow);
    if (visualRow >= this.hot.countRows()) {
      hfRowIndex = visualRow; // Row beyond the table boundaries.
    }

    if (this.sheetId === null || !this.engine.doesSheetExist(this.sheetName) || !this.engine.isItPossibleToAddRows(this.sheetId, [hfRowIndex, amount])) {
      return false;
    }
  }

  /**
   * `beforeCreateCol` hook callback.
   *
   * @private
   * @param {number} visualColumn Represents the visual index of first newly created column in the data source.
   * @param {number} amount Number of newly created columns in the data source.
   * @returns {*|boolean} If false is returned the action is canceled.
   */
  onBeforeCreateCol(visualColumn, amount) {
    let hfColumnIndex = this.columnAxisSyncer.getHfIndexFromVisualIndex(visualColumn);
    if (visualColumn >= this.hot.countCols()) {
      hfColumnIndex = visualColumn; // Column beyond the table boundaries.
    }

    if (this.sheetId === null || !this.engine.doesSheetExist(this.sheetName) || !this.engine.isItPossibleToAddColumns(this.sheetId, [hfColumnIndex, amount])) {
      return false;
    }
  }

  /**
   * `beforeRemoveRow` hook callback.
   *
   * @private
   * @param {number} row Visual index of starter row.
   * @param {number} amount Amount of rows to be removed.
   * @param {number[]} physicalRows An array of physical rows removed from the data source.
   * @returns {*|boolean} If false is returned the action is canceled.
   */
  onBeforeRemoveRow(row, amount, physicalRows) {
    const hfRows = this.rowAxisSyncer.setRemovedHfIndexes(physicalRows);
    const possible = hfRows.every(hfRow => {
      return this.engine.isItPossibleToRemoveRows(this.sheetId, [hfRow, 1]);
    });
    return possible === false ? false : void 0;
  }

  /**
   * `beforeRemoveCol` hook callback.
   *
   * @private
   * @param {number} col Visual index of starter column.
   * @param {number} amount Amount of columns to be removed.
   * @param {number[]} physicalColumns An array of physical columns removed from the data source.
   * @returns {*|boolean} If false is returned the action is canceled.
   */
  onBeforeRemoveCol(col, amount, physicalColumns) {
    const hfColumns = this.columnAxisSyncer.setRemovedHfIndexes(physicalColumns);
    const possible = hfColumns.every(hfColumn => {
      return this.engine.isItPossibleToRemoveColumns(this.sheetId, [hfColumn, 1]);
    });
    return possible === false ? false : void 0;
  }

  /**
   * `afterCreateRow` hook callback.
   *
   * @private
   * @param {number} visualRow Represents the visual index of first newly created row in the data source array.
   * @param {number} amount Number of newly created rows in the data source array.
   * @param {string} [source] String that identifies source of hook call
   *                          ([list of all available sources]{@link https://handsontable.com/docs/javascript-data-grid/events-and-hooks/#handsontable-hooks}).
   */
  onAfterCreateRow(visualRow, amount, source) {
    if (isBlockedSource(source)) {
      return;
    }
    const changes = this.engine.addRows(this.sheetId, [this.rowAxisSyncer.getHfIndexFromVisualIndex(visualRow), amount]);
    this.renderDependentSheets(changes);
  }

  /**
   * `afterCreateCol` hook callback.
   *
   * @private
   * @param {number} visualColumn Represents the visual index of first newly created column in the data source.
   * @param {number} amount Number of newly created columns in the data source.
   * @param {string} [source] String that identifies source of hook call
   *                          ([list of all available sources]{@link https://handsontable.com/docs/javascript-data-grid/events-and-hooks/#handsontable-hooks}).
   */
  onAfterCreateCol(visualColumn, amount, source) {
    if (isBlockedSource(source)) {
      return;
    }
    const changes = this.engine.addColumns(this.sheetId, [this.columnAxisSyncer.getHfIndexFromVisualIndex(visualColumn), amount]);
    this.renderDependentSheets(changes);
  }

  /**
   * `afterRemoveRow` hook callback.
   *
   * @private
   * @param {number} row Visual index of starter row.
   * @param {number} amount An amount of removed rows.
   * @param {number[]} physicalRows An array of physical rows removed from the data source.
   * @param {string} [source] String that identifies source of hook call
   *                          ([list of all available sources]{@link https://handsontable.com/docs/javascript-data-grid/events-and-hooks/#handsontable-hooks}).
   */
  onAfterRemoveRow(row, amount, physicalRows, source) {
    if (isBlockedSource(source)) {
      return;
    }
    const descendingHfRows = this.rowAxisSyncer.getRemovedHfIndexes().sort().reverse();
    const changes = this.engine.batch(() => {
      descendingHfRows.forEach(hfRow => {
        this.engine.removeRows(this.sheetId, [hfRow, 1]);
      });
    });
    this.renderDependentSheets(changes);
  }

  /**
   * `afterRemoveCol` hook callback.
   *
   * @private
   * @param {number} col Visual index of starter column.
   * @param {number} amount An amount of removed columns.
   * @param {number[]} physicalColumns An array of physical columns removed from the data source.
   * @param {string} [source] String that identifies source of hook call
   *                          ([list of all available sources]{@link https://handsontable.com/docs/javascript-data-grid/events-and-hooks/#handsontable-hooks}).
   */
  onAfterRemoveCol(col, amount, physicalColumns, source) {
    if (isBlockedSource(source)) {
      return;
    }
    const descendingHfColumns = this.columnAxisSyncer.getRemovedHfIndexes().sort().reverse();
    const changes = this.engine.batch(() => {
      descendingHfColumns.forEach(hfColumn => {
        this.engine.removeColumns(this.sheetId, [hfColumn, 1]);
      });
    });
    this.renderDependentSheets(changes);
  }

  /**
   * `afterDetachChild` hook callback.
   * Used to sync the data of the rows detached in the Nested Rows plugin with the engine's dataset.
   *
   * @private
   * @param {object} parent An object representing the parent from which the element was detached.
   * @param {object} element The detached element.
   * @param {number} finalElementRowIndex The final row index of the detached element.
   */
  onAfterDetachChild(parent, element, finalElementRowIndex) {
    var _element$__children;
    _classPrivateFieldSet(this, _internalOperationPending, true);
    const rowsData = this.hot.getSourceDataArray(finalElementRowIndex, 0, finalElementRowIndex + (((_element$__children = element.__children) === null || _element$__children === void 0 ? void 0 : _element$__children.length) || 0), this.hot.countSourceCols());
    _classPrivateFieldSet(this, _internalOperationPending, false);
    rowsData.forEach((row, relativeRowIndex) => {
      row.forEach((value, colIndex) => {
        this.engine.setCellContents({
          col: colIndex,
          row: finalElementRowIndex + relativeRowIndex,
          sheet: this.sheetId
        }, [[value]]);
      });
    });
  }

  /**
   * Called when a value is updated in the engine.
   *
   * @private
   * @fires Hooks#afterFormulasValuesUpdate
   * @param {Array} changes The values and location of applied changes.
   */
  onEngineValuesUpdated(changes) {
    this.hot.runHooks('afterFormulasValuesUpdate', changes);
  }

  /**
   * Called when a named expression is added to the engine instance.
   *
   * @private
   * @fires Hooks#afterNamedExpressionAdded
   * @param {string} namedExpressionName The name of the added expression.
   * @param {Array} changes The values and location of applied changes.
   */
  onEngineNamedExpressionsAdded(namedExpressionName, changes) {
    this.hot.runHooks('afterNamedExpressionAdded', namedExpressionName, changes);
  }

  /**
   * Called when a named expression is removed from the engine instance.
   *
   * @private
   * @fires Hooks#afterNamedExpressionRemoved
   * @param {string} namedExpressionName The name of the removed expression.
   * @param {Array} changes The values and location of applied changes.
   */
  onEngineNamedExpressionsRemoved(namedExpressionName, changes) {
    this.hot.runHooks('afterNamedExpressionRemoved', namedExpressionName, changes);
  }

  /**
   * Called when a new sheet is added to the engine instance.
   *
   * @private
   * @fires Hooks#afterSheetAdded
   * @param {string} addedSheetDisplayName The name of the added sheet.
   */
  onEngineSheetAdded(addedSheetDisplayName) {
    this.hot.runHooks('afterSheetAdded', addedSheetDisplayName);
  }

  /**
   * Called when a sheet in the engine instance is renamed.
   *
   * @private
   * @fires Hooks#afterSheetRenamed
   * @param {string} oldDisplayName The old name of the sheet.
   * @param {string} newDisplayName The new name of the sheet.
   */
  onEngineSheetRenamed(oldDisplayName, newDisplayName) {
    this.hot.runHooks('afterSheetRenamed', oldDisplayName, newDisplayName);
  }

  /**
   * Called when a sheet is removed from the engine instance.
   *
   * @private
   * @fires Hooks#afterSheetRemoved
   * @param {string} removedSheetDisplayName The removed sheet name.
   * @param {Array} changes The values and location of applied changes.
   */
  onEngineSheetRemoved(removedSheetDisplayName, changes) {
    this.hot.runHooks('afterSheetRemoved', removedSheetDisplayName, changes);
  }
}
exports.Formulas = Formulas;