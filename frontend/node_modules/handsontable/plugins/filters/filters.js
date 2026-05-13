"use strict";

exports.__esModule = true;
require("core-js/modules/es.array.push.js");
var _base = require("../base");
var _array = require("../../helpers/array");
var _templateLiteralTag = require("../../helpers/templateLiteralTag");
var _console = require("../../helpers/console");
var _number = require("../../helpers/number");
var _eventManager = _interopRequireDefault(require("../../eventManager"));
var _element = require("../../helpers/dom/element");
var _predefinedItems = require("../contextMenu/predefinedItems");
var constants = _interopRequireWildcard(require("../../i18n/constants"));
var _condition = _interopRequireDefault(require("./component/condition"));
var _operators = _interopRequireDefault(require("./component/operators"));
var _value = _interopRequireDefault(require("./component/value"));
var _actionBar = _interopRequireDefault(require("./component/actionBar"));
var _conditionCollection = _interopRequireDefault(require("./conditionCollection"));
var _dataFilter = _interopRequireDefault(require("./dataFilter"));
var _conditionUpdateObserver = _interopRequireDefault(require("./conditionUpdateObserver"));
var _utils = require("./utils");
var _constants2 = require("./constants");
var _translations = require("../../translations");
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
const PLUGIN_KEY = 'filters';
exports.PLUGIN_KEY = PLUGIN_KEY;
const PLUGIN_PRIORITY = 250;

/**
 * @plugin Filters
 * @class Filters
 *
 * @description
 * The plugin allows filtering the table data either by the built-in component or with the API.
 *
 * See [the filtering demo](@/guides/columns/column-filter.md) for examples.
 *
 * @example
 * ::: only-for javascript
 * ```js
 * const container = document.getElementById('example');
 * const hot = new Handsontable(container, {
 *   data: getData(),
 *   colHeaders: true,
 *   rowHeaders: true,
 *   dropdownMenu: true,
 *   filters: true
 * });
 * ```
 * :::
 *
 * ::: only-for react
 * ```jsx
 * <HotTable
 *   data={getData()}
 *   colHeaders={true}
 *   rowHeaders={true}
 *   dropdownMenu={true}
 *   filters={true}
 * />
 * ```
 * :::
 */
exports.PLUGIN_PRIORITY = PLUGIN_PRIORITY;
class Filters extends _base.BasePlugin {
  static get PLUGIN_KEY() {
    return PLUGIN_KEY;
  }
  static get PLUGIN_PRIORITY() {
    return PLUGIN_PRIORITY;
  }
  static get PLUGIN_DEPS() {
    return ['plugin:DropdownMenu', 'plugin:HiddenRows', 'cell-type:checkbox'];
  }
  constructor(hotInstance) {
    super(hotInstance);
    /**
     * Instance of {@link EventManager}.
     *
     * @private
     * @type {EventManager}
     */
    this.eventManager = new _eventManager.default(this);
    /**
     * Instance of {@link DropdownMenu}.
     *
     * @private
     * @type {DropdownMenu}
     */
    this.dropdownMenuPlugin = null;
    /**
     * Instance of {@link ConditionCollection}.
     *
     * @private
     * @type {ConditionCollection}
     */
    this.conditionCollection = null;
    /**
     * Instance of {@link ConditionUpdateObserver}.
     *
     * @private
     * @type {ConditionUpdateObserver}
     */
    this.conditionUpdateObserver = null;
    /**
     * Map, where key is component identifier and value represent `BaseComponent` element or it derivatives.
     *
     * @private
     * @type {Map}
     */
    this.components = new Map([['filter_by_condition', null], ['filter_operators', null], ['filter_by_condition2', null], ['filter_by_value', null], ['filter_action_bar', null]]);
    /**
     * Map of skipped rows by plugin.
     *
     * @private
     * @type {null|TrimmingMap}
     */
    this.filtersRowsMap = null;

    // One listener for the enable/disable functionality
    this.hot.addHook('afterGetColHeader', (col, TH) => this.onAfterGetColHeader(col, TH));
  }

  /**
   * Checks if the plugin is enabled in the handsontable settings. This method is executed in {@link Hooks#beforeInit}
   * hook and if it returns `true` then the {@link Filters#enablePlugin} method is called.
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
    if (this.enabled) {
      return;
    }
    this.filtersRowsMap = this.hot.rowIndexMapper.registerMap(this.pluginName, new _translations.TrimmingMap());
    this.dropdownMenuPlugin = this.hot.getPlugin('dropdownMenu');
    const dropdownSettings = this.hot.getSettings().dropdownMenu;
    const menuContainer = dropdownSettings && dropdownSettings.uiContainer || this.hot.rootDocument.body;
    const addConfirmationHooks = component => {
      component.addLocalHook('accept', () => this.onActionBarSubmit('accept'));
      component.addLocalHook('cancel', () => this.onActionBarSubmit('cancel'));
      component.addLocalHook('change', command => this.onComponentChange(component, command));
      return component;
    };
    const filterByConditionLabel = () => `${this.hot.getTranslatedPhrase(constants.FILTERS_DIVS_FILTER_BY_CONDITION)}:`;
    const filterValueLabel = () => `${this.hot.getTranslatedPhrase(constants.FILTERS_DIVS_FILTER_BY_VALUE)}:`;
    if (!this.components.get('filter_by_condition')) {
      const conditionComponent = new _condition.default(this.hot, {
        id: 'filter_by_condition',
        name: filterByConditionLabel,
        addSeparator: false,
        menuContainer
      });
      conditionComponent.addLocalHook('afterClose', () => this.onSelectUIClosed());
      this.components.set('filter_by_condition', addConfirmationHooks(conditionComponent));
    }
    if (!this.components.get('filter_operators')) {
      this.components.set('filter_operators', new _operators.default(this.hot, {
        id: 'filter_operators',
        name: 'Operators'
      }));
    }
    if (!this.components.get('filter_by_condition2')) {
      const conditionComponent = new _condition.default(this.hot, {
        id: 'filter_by_condition2',
        name: '',
        addSeparator: true,
        menuContainer
      });
      conditionComponent.addLocalHook('afterClose', () => this.onSelectUIClosed());
      this.components.set('filter_by_condition2', addConfirmationHooks(conditionComponent));
    }
    if (!this.components.get('filter_by_value')) {
      this.components.set('filter_by_value', addConfirmationHooks(new _value.default(this.hot, {
        id: 'filter_by_value',
        name: filterValueLabel
      })));
    }
    if (!this.components.get('filter_action_bar')) {
      this.components.set('filter_action_bar', addConfirmationHooks(new _actionBar.default(this.hot, {
        id: 'filter_action_bar',
        name: 'Action bar'
      })));
    }
    if (!this.conditionCollection) {
      this.conditionCollection = new _conditionCollection.default(this.hot);
    }
    if (!this.conditionUpdateObserver) {
      this.conditionUpdateObserver = new _conditionUpdateObserver.default(this.hot, this.conditionCollection, physicalColumn => this.getDataMapAtColumn(physicalColumn));
      this.conditionUpdateObserver.addLocalHook('update', conditionState => this.updateComponents(conditionState));
    }
    this.components.forEach(component => component.show());
    this.addHook('beforeDropdownMenuSetItems', items => this.onBeforeDropdownMenuSetItems(items));
    this.addHook('afterDropdownMenuDefaultOptions', defaultOptions => this.onAfterDropdownMenuDefaultOptions(defaultOptions));
    this.addHook('afterDropdownMenuShow', () => this.onAfterDropdownMenuShow());
    this.addHook('afterDropdownMenuHide', () => this.onAfterDropdownMenuHide());
    this.addHook('afterChange', changes => this.onAfterChange(changes));

    // Temp. solution (extending menu items bug in contextMenu/dropdownMenu)
    if (this.hot.getSettings().dropdownMenu && this.dropdownMenuPlugin) {
      this.dropdownMenuPlugin.disablePlugin();
      this.dropdownMenuPlugin.enablePlugin();
    }
    super.enablePlugin();
  }

  /**
   * Disables the plugin functionality for this Handsontable instance.
   */
  disablePlugin() {
    if (this.enabled) {
      var _this$dropdownMenuPlu;
      if ((_this$dropdownMenuPlu = this.dropdownMenuPlugin) !== null && _this$dropdownMenuPlu !== void 0 && _this$dropdownMenuPlu.enabled) {
        this.dropdownMenuPlugin.menu.clearLocalHooks();
      }
      this.components.forEach((component, key) => {
        component.destroy();
        this.components.set(key, null);
      });
      this.conditionCollection.destroy();
      this.conditionCollection = null;
      this.hot.rowIndexMapper.unregisterMap(this.pluginName);
    }
    super.disablePlugin();
  }

  /* eslint-disable jsdoc/require-description-complete-sentence */
  /**
   * @memberof Filters#
   * @function addCondition
   * @description
   * Adds condition to the conditions collection at specified column index.
   *
   * Possible predefined conditions:
   *  * `begins_with` - Begins with
   *  * `between` - Between
   *  * `by_value` - By value
   *  * `contains` - Contains
   *  * `date_after` - After a date
   *  * `date_before` - Before a date
   *  * `date_today` - Today
   *  * `date_tomorrow` - Tomorrow
   *  * `date_yesterday` - Yesterday
   *  * `empty` - Empty
   *  * `ends_with` - Ends with
   *  * `eq` - Equal
   *  * `gt` - Greater than
   *  * `gte` - Greater than or equal
   *  * `lt` - Less than
   *  * `lte` - Less than or equal
   *  * `none` - None (no filter)
   *  * `not_between` - Not between
   *  * `not_contains` - Not contains
   *  * `not_empty` - Not empty
   *  * `neq` - Not equal.
   *
   * Possible operations on collection of conditions:
   *  * `conjunction` - [**Conjunction**](https://en.wikipedia.org/wiki/Logical_conjunction) on conditions collection (by default), i.e. for such operation: <br/> c1 AND c2 AND c3 AND c4 ... AND cn === TRUE, where c1 ... cn are conditions.
   *  * `disjunction` - [**Disjunction**](https://en.wikipedia.org/wiki/Logical_disjunction) on conditions collection, i.e. for such operation: <br/> c1 OR c2 OR c3 OR c4 ... OR cn === TRUE, where c1, c2, c3, c4 ... cn are conditions.
   *  * `disjunctionWithExtraCondition` - **Disjunction** on first `n - 1`\* conditions from collection with an extra requirement computed from the last condition, i.e. for such operation: <br/> c1 OR c2 OR c3 OR c4 ... OR cn-1 AND cn === TRUE, where c1, c2, c3, c4 ... cn are conditions.
   *
   * \* when `n` is collection size; it's used i.e. for one operation introduced from UI (when choosing from filter's drop-down menu two conditions with OR operator between them, mixed with choosing values from the multiple choice select)
   *
   * **Note**: Mind that you cannot mix different types of operations (for instance, if you use `conjunction`, use it consequently for a particular column).
   *
   * @example
   * ::: only-for javascript
   * ```js
   * const container = document.getElementById('example');
   * const hot = new Handsontable(container, {
   *   data: getData(),
   *   filters: true
   * });
   *
   * // access to filters plugin instance
   * const filtersPlugin = hot.getPlugin('filters');
   *
   * // add filter "Greater than" 95 to column at index 1
   * filtersPlugin.addCondition(1, 'gt', [95]);
   * filtersPlugin.filter();
   *
   * // add filter "By value" to column at index 1
   * // in this case all value's that don't match will be filtered.
   * filtersPlugin.addCondition(1, 'by_value', [['ing', 'ed', 'as', 'on']]);
   * filtersPlugin.filter();
   *
   * // add filter "Begins with" with value "de" AND "Not contains" with value "ing"
   * filtersPlugin.addCondition(1, 'begins_with', ['de'], 'conjunction');
   * filtersPlugin.addCondition(1, 'not_contains', ['ing'], 'conjunction');
   * filtersPlugin.filter();
   *
   * // add filter "Begins with" with value "de" OR "Not contains" with value "ing"
   * filtersPlugin.addCondition(1, 'begins_with', ['de'], 'disjunction');
   * filtersPlugin.addCondition(1, 'not_contains', ['ing'], 'disjunction');
   * filtersPlugin.filter();
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
   *   filters={true}
   * />
   *
   * // access to filters plugin instance
   * const hot = hotRef.current.hotInstance;
   * const filtersPlugin = hot.getPlugin('filters');
   *
   * // add filter "Greater than" 95 to column at index 1
   * filtersPlugin.addCondition(1, 'gt', [95]);
   * filtersPlugin.filter();
   *
   * // add filter "By value" to column at index 1
   * // in this case all value's that don't match will be filtered.
   * filtersPlugin.addCondition(1, 'by_value', [['ing', 'ed', 'as', 'on']]);
   * filtersPlugin.filter();
   *
   * // add filter "Begins with" with value "de" AND "Not contains" with value "ing"
   * filtersPlugin.addCondition(1, 'begins_with', ['de'], 'conjunction');
   * filtersPlugin.addCondition(1, 'not_contains', ['ing'], 'conjunction');
   * filtersPlugin.filter();
   *
   * // add filter "Begins with" with value "de" OR "Not contains" with value "ing"
   * filtersPlugin.addCondition(1, 'begins_with', ['de'], 'disjunction');
   * filtersPlugin.addCondition(1, 'not_contains', ['ing'], 'disjunction');
   * filtersPlugin.filter();
   * ```
   * :::
   *
   * @param {number} column Visual column index.
   * @param {string} name Condition short name.
   * @param {Array} args Condition arguments.
   * @param {string} [operationId=conjunction] `id` of operation which is performed on the column.
   */
  /* eslint-enable jsdoc/require-description-complete-sentence */
  addCondition(column, name, args) {
    let operationId = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : _constants2.OPERATION_AND;
    const physicalColumn = this.hot.toPhysicalColumn(column);
    this.conditionCollection.addCondition(physicalColumn, {
      command: {
        key: name
      },
      args
    }, operationId);
  }

  /**
   * Removes conditions at specified column index.
   *
   * @param {number} column Visual column index.
   */
  removeConditions(column) {
    const physicalColumn = this.hot.toPhysicalColumn(column);
    this.conditionCollection.removeConditions(physicalColumn);
  }

  /**
   * Clears all conditions previously added to the collection for the specified column index or, if the column index
   * was not passed, clear the conditions for all columns.
   *
   * @param {number} [column] Visual column index.
   */
  clearConditions(column) {
    if (column === void 0) {
      this.conditionCollection.clean();
    } else {
      const physicalColumn = this.hot.toPhysicalColumn(column);
      this.conditionCollection.removeConditions(physicalColumn);
    }
  }

  /**
   * Filters data based on added filter conditions.
   *
   * @fires Hooks#beforeFilter
   * @fires Hooks#afterFilter
   */
  filter() {
    const dataFilter = this._createDataFilter();
    const needToFilter = !this.conditionCollection.isEmpty();
    let visibleVisualRows = [];
    const conditions = this.conditionCollection.exportAllConditions();
    const allowFiltering = this.hot.runHooks('beforeFilter', conditions);
    if (allowFiltering !== false) {
      if (needToFilter) {
        const trimmedRows = [];
        this.hot.batchExecution(() => {
          this.filtersRowsMap.clear();
          visibleVisualRows = (0, _array.arrayMap)(dataFilter.filter(), rowData => rowData.meta.visualRow);
          const visibleVisualRowsAssertion = (0, _utils.createArrayAssertion)(visibleVisualRows);
          (0, _number.rangeEach)(this.hot.countSourceRows() - 1, row => {
            if (!visibleVisualRowsAssertion(row)) {
              trimmedRows.push(row);
            }
          });
          (0, _array.arrayEach)(trimmedRows, physicalRow => {
            this.filtersRowsMap.setValueAtIndex(physicalRow, true);
          });
        }, true);
        if (!visibleVisualRows.length) {
          this.hot.deselectCell();
        }
      } else {
        this.filtersRowsMap.clear();
      }
    }
    this.hot.runHooks('afterFilter', conditions);
    this.hot.view.adjustElementsSize(true);
    this.hot.render();
    this.clearColumnSelection();
  }

  /**
   * Gets last selected column index.
   *
   * @returns {{visualIndex: number, physicalIndex: number} | null} Returns `null` when a column is
   * not selected. Otherwise, returns an object with `visualIndex` and `physicalIndex` properties containing
   * the index of the column.
   */
  getSelectedColumn() {
    var _this$hot$getSelected;
    const highlight = (_this$hot$getSelected = this.hot.getSelectedRangeLast()) === null || _this$hot$getSelected === void 0 ? void 0 : _this$hot$getSelected.highlight;
    if (!highlight) {
      return null;
    }
    return {
      visualIndex: highlight.col,
      physicalIndex: this.hot.toPhysicalColumn(highlight.col)
    };
  }

  /**
   * Clears column selection.
   *
   * @private
   */
  clearColumnSelection() {
    const selectedColumn = this.getSelectedColumn();
    if (selectedColumn !== null) {
      this.hot.selectCell(0, selectedColumn.visualIndex);
    }
  }

  /**
   * Returns handsontable source data with cell meta based on current selection.
   *
   * @param {number} [column] The physical column index. By default column index accept the value of the selected column.
   * @returns {Array} Returns array of objects where keys as row index.
   */
  getDataMapAtColumn(column) {
    const visualColumn = this.hot.toVisualColumn(column);
    const data = [];
    (0, _array.arrayEach)(this.hot.getSourceDataAtCol(visualColumn), (value, rowIndex) => {
      var _this$hot$getDataAtCe;
      const {
        row,
        col,
        visualCol,
        visualRow,
        type,
        instance,
        dateFormat,
        locale
      } = this.hot.getCellMeta(rowIndex, visualColumn);
      const dataValue = (_this$hot$getDataAtCe = this.hot.getDataAtCell(this.hot.toVisualRow(rowIndex), visualColumn)) !== null && _this$hot$getDataAtCe !== void 0 ? _this$hot$getDataAtCe : value;
      data.push({
        meta: {
          row,
          col,
          visualCol,
          visualRow,
          type,
          instance,
          dateFormat,
          locale
        },
        value: (0, _utils.toEmptyString)(dataValue)
      });
    });
    return data;
  }

  /**
   * `afterChange` listener.
   *
   * @private
   * @param {Array} changes Array of changes.
   */
  onAfterChange(changes) {
    if (changes) {
      (0, _array.arrayEach)(changes, change => {
        const [, prop] = change;
        const columnIndex = this.hot.propToCol(prop);
        if (this.conditionCollection.hasConditions(columnIndex)) {
          this.updateValueComponentCondition(columnIndex);
        }
      });
    }
  }

  /**
   * Update the condition of ValueComponent, based on the handled changes.
   *
   * @private
   * @param {number} columnIndex Column index of handled ValueComponent condition.
   */
  updateValueComponentCondition(columnIndex) {
    const dataAtCol = this.hot.getDataAtCol(columnIndex);
    const selectedValues = (0, _utils.unifyColumnValues)(dataAtCol);
    this.conditionUpdateObserver.updateStatesAtColumn(columnIndex, selectedValues);
  }

  /**
   * Restores components to its saved state.
   *
   * @private
   * @param {Array} components List of components.
   */
  restoreComponents(components) {
    var _this$getSelectedColu;
    const physicalIndex = (_this$getSelectedColu = this.getSelectedColumn()) === null || _this$getSelectedColu === void 0 ? void 0 : _this$getSelectedColu.physicalIndex;
    components.forEach(component => {
      if (component.isHidden()) {
        return;
      }
      component.restoreState(physicalIndex);
    });
    this.updateDependentComponentsVisibility();
  }

  /**
   * After dropdown menu show listener.
   *
   * @private
   */
  onAfterDropdownMenuShow() {
    this.restoreComponents(Array.from(this.components.values()));
  }

  /**
   * After dropdown menu hide listener.
   *
   * @private
   */
  onAfterDropdownMenuHide() {
    this.components.get('filter_by_condition').getSelectElement().closeOptions();
    this.components.get('filter_by_condition2').getSelectElement().closeOptions();
  }

  /**
   * Before dropdown menu set menu items listener.
   *
   * @private
   */
  onBeforeDropdownMenuSetItems() {
    if (this.dropdownMenuPlugin) {
      this.dropdownMenuPlugin.menu.addLocalHook('afterOpen', () => {
        this.dropdownMenuPlugin.menu.hotMenu.updateSettings({
          hiddenRows: true
        });
      });
    }
  }

  /**
   * After dropdown menu default options listener.
   *
   * @private
   * @param {object} defaultOptions ContextMenu default item options.
   */
  onAfterDropdownMenuDefaultOptions(defaultOptions) {
    defaultOptions.items.push({
      name: _predefinedItems.SEPARATOR
    });
    this.components.forEach(component => {
      defaultOptions.items.push(component.getMenuItemDescriptor());
    });
  }

  /**
   * Get an operation, based on the number and types of arguments (where arguments are states of components).
   *
   * @param {string} suggestedOperation Operation which was chosen by user from UI.
   * @param {object} byConditionState1 State of first condition component.
   * @param {object} byConditionState2 State of second condition component.
   * @param {object} byValueState State of value component.
   * @private
   * @returns {string}
   */
  getOperationBasedOnArguments(suggestedOperation, byConditionState1, byConditionState2, byValueState) {
    let operation = suggestedOperation;
    if (operation === _constants2.OPERATION_OR && byConditionState1.command.key !== _constants2.CONDITION_NONE && byConditionState2.command.key !== _constants2.CONDITION_NONE && byValueState.command.key !== _constants2.CONDITION_NONE) {
      operation = _constants2.OPERATION_OR_THEN_VARIABLE;
    } else if (byValueState.command.key !== _constants2.CONDITION_NONE) {
      if (byConditionState1.command.key === _constants2.CONDITION_NONE || byConditionState2.command.key === _constants2.CONDITION_NONE) {
        operation = _constants2.OPERATION_AND;
      }
    }
    return operation;
  }

  /**
   * On action bar submit listener.
   *
   * @private
   * @param {string} submitType The submit type.
   */
  onActionBarSubmit(submitType) {
    var _this$dropdownMenuPlu3;
    if (submitType === 'accept') {
      const selectedColumn = this.getSelectedColumn();
      if (selectedColumn === null) {
        var _this$dropdownMenuPlu2;
        (_this$dropdownMenuPlu2 = this.dropdownMenuPlugin) === null || _this$dropdownMenuPlu2 === void 0 || _this$dropdownMenuPlu2.close();
        return;
      }
      const {
        physicalIndex
      } = selectedColumn;
      const byConditionState1 = this.components.get('filter_by_condition').getState();
      const byConditionState2 = this.components.get('filter_by_condition2').getState();
      const byValueState = this.components.get('filter_by_value').getState();
      const operation = this.getOperationBasedOnArguments(this.components.get('filter_operators').getActiveOperationId(), byConditionState1, byConditionState2, byValueState);
      this.conditionUpdateObserver.groupChanges();
      let columnStackPosition = this.conditionCollection.getColumnStackPosition(physicalIndex);
      if (columnStackPosition === -1) {
        columnStackPosition = void 0;
      }
      this.conditionCollection.removeConditions(physicalIndex);
      if (byConditionState1.command.key !== _constants2.CONDITION_NONE) {
        this.conditionCollection.addCondition(physicalIndex, byConditionState1, operation, columnStackPosition);
        if (byConditionState2.command.key !== _constants2.CONDITION_NONE) {
          this.conditionCollection.addCondition(physicalIndex, byConditionState2, operation, columnStackPosition);
        }
      }
      if (byValueState.command.key !== _constants2.CONDITION_NONE) {
        this.conditionCollection.addCondition(physicalIndex, byValueState, operation, columnStackPosition);
      }
      this.conditionUpdateObserver.flush();
      this.components.forEach(component => component.saveState(physicalIndex));
      this.filtersRowsMap.clear();
      this.filter();
    }
    (_this$dropdownMenuPlu3 = this.dropdownMenuPlugin) === null || _this$dropdownMenuPlu3 === void 0 || _this$dropdownMenuPlu3.close();
  }

  /**
   * On component change listener.
   *
   * @private
   * @param {BaseComponent} component Component inheriting BaseComponent.
   * @param {object} command Menu item object (command).
   */
  onComponentChange(component, command) {
    this.updateDependentComponentsVisibility();
    if (component.constructor === _condition.default && !command.inputsCount) {
      this.setListeningDropdownMenu();
    }
  }

  /**
   * On component SelectUI closed listener.
   *
   * @private
   */
  onSelectUIClosed() {
    this.setListeningDropdownMenu();
  }

  /**
   * Listen to the keyboard input on document body and forward events to instance of Handsontable
   * created by DropdownMenu plugin.
   *
   * @private
   */
  setListeningDropdownMenu() {
    if (this.dropdownMenuPlugin) {
      this.dropdownMenuPlugin.setListening();
    }
  }

  /**
   * Updates visibility of some of the components, based on the state of the parent component.
   *
   * @private
   */
  updateDependentComponentsVisibility() {
    const component = this.components.get('filter_by_condition');
    const {
      command
    } = component.getState();
    const componentsToShow = [this.components.get('filter_by_condition2'), this.components.get('filter_operators')];
    if (command.showOperators) {
      this.showComponents(...componentsToShow);
    } else {
      this.hideComponents(...componentsToShow);
    }
  }

  /**
   * On after get column header listener.
   *
   * @private
   * @param {number} col Visual column index.
   * @param {HTMLTableCellElement} TH Header's TH element.
   */
  onAfterGetColHeader(col, TH) {
    const physicalColumn = this.hot.toPhysicalColumn(col);
    if (this.enabled && this.conditionCollection.hasConditions(physicalColumn)) {
      (0, _element.addClass)(TH, 'htFiltersActive');
    } else {
      (0, _element.removeClass)(TH, 'htFiltersActive');
    }
  }

  /**
   * Creates DataFilter instance based on condition collection.
   *
   * @private
   * @param {ConditionCollection} conditionCollection Condition collection object.
   * @returns {DataFilter}
   */
  _createDataFilter() {
    let conditionCollection = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : this.conditionCollection;
    return new _dataFilter.default(conditionCollection, physicalColumn => this.getDataMapAtColumn(physicalColumn));
  }

  /**
   * It updates the components state. The state is triggered by ConditionUpdateObserver, which
   * reacts to any condition added to the condition collection. It may be added through the UI
   * components or by API call.
   *
   * @private
   * @param {object} conditionsState An object with the state generated by UI components.
   */
  updateComponents(conditionsState) {
    var _this$dropdownMenuPlu4;
    if (!((_this$dropdownMenuPlu4 = this.dropdownMenuPlugin) !== null && _this$dropdownMenuPlu4 !== void 0 && _this$dropdownMenuPlu4.enabled)) {
      return;
    }
    const {
      editedConditionStack: {
        conditions,
        column
      }
    } = conditionsState;
    const conditionsByValue = conditions.filter(condition => condition.name === _constants2.CONDITION_BY_VALUE);
    const conditionsWithoutByValue = conditions.filter(condition => condition.name !== _constants2.CONDITION_BY_VALUE);
    if (conditionsByValue.length >= 2 || conditionsWithoutByValue.length >= 3) {
      (0, _console.warn)((0, _templateLiteralTag.toSingleLine)`The filter conditions have been applied properly, but couldn’t be displayed visually.\x20
        The overall amount of conditions exceed the capability of the dropdown menu.\x20
        For more details see the documentation.`);
    } else {
      const operationType = this.conditionCollection.getOperation(column);
      this.components.get('filter_by_condition').updateState(conditionsWithoutByValue[0], column);
      this.components.get('filter_by_condition2').updateState(conditionsWithoutByValue[1], column);
      this.components.get('filter_operators').updateState(operationType, column);
      this.components.get('filter_by_value').updateState(conditionsState);
    }
  }

  /**
   * Returns indexes of passed components inside list of `dropdownMenu` items.
   *
   * @private
   * @param {...BaseComponent} components List of components.
   * @returns {Array}
   */
  getIndexesOfComponents() {
    const indexes = [];
    if (!this.dropdownMenuPlugin) {
      return indexes;
    }
    const menu = this.dropdownMenuPlugin.menu;
    for (var _len = arguments.length, components = new Array(_len), _key = 0; _key < _len; _key++) {
      components[_key] = arguments[_key];
    }
    (0, _array.arrayEach)(components, component => {
      (0, _array.arrayEach)(menu.menuItems, (item, index) => {
        if (item.key === component.getMenuItemDescriptor().key) {
          indexes.push(index);
        }
      });
    });
    return indexes;
  }

  /**
   * Changes visibility of component.
   *
   * @private
   * @param {boolean} visible Determine if components should be visible.
   * @param {...BaseComponent} components List of components.
   */
  changeComponentsVisibility() {
    let visible = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : true;
    if (!this.dropdownMenuPlugin) {
      return;
    }
    const menu = this.dropdownMenuPlugin.menu;
    const hotMenu = menu.hotMenu;
    const hiddenRows = hotMenu.getPlugin('hiddenRows');
    for (var _len2 = arguments.length, components = new Array(_len2 > 1 ? _len2 - 1 : 0), _key2 = 1; _key2 < _len2; _key2++) {
      components[_key2 - 1] = arguments[_key2];
    }
    const indexes = this.getIndexesOfComponents(...components);
    if (visible) {
      hiddenRows.showRows(indexes);
    } else {
      hiddenRows.hideRows(indexes);
    }
    hotMenu.render();
  }

  /**
   * Hides components of filters `dropdownMenu`.
   *
   * @private
   * @param {...BaseComponent} components List of components.
   */
  hideComponents() {
    for (var _len3 = arguments.length, components = new Array(_len3), _key3 = 0; _key3 < _len3; _key3++) {
      components[_key3] = arguments[_key3];
    }
    this.changeComponentsVisibility(false, ...components);
  }

  /**
   * Shows components of filters `dropdownMenu`.
   *
   * @private
   * @param {...BaseComponent} components List of components.
   */
  showComponents() {
    for (var _len4 = arguments.length, components = new Array(_len4), _key4 = 0; _key4 < _len4; _key4++) {
      components[_key4] = arguments[_key4];
    }
    this.changeComponentsVisibility(true, ...components);
  }

  /**
   * Destroys the plugin instance.
   */
  destroy() {
    if (this.enabled) {
      this.components.forEach((component, key) => {
        if (component !== null) {
          component.destroy();
          this.components.set(key, null);
        }
      });
      this.conditionCollection.destroy();
      this.conditionUpdateObserver.destroy();
      this.hot.rowIndexMapper.unregisterMap(this.pluginName);
    }
    super.destroy();
  }
}
exports.Filters = Filters;