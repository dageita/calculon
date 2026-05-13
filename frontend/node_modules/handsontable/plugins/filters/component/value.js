"use strict";

exports.__esModule = true;
require("core-js/modules/es.array.push.js");
var _element = require("../../../helpers/dom/element");
var _event = require("../../../helpers/dom/event");
var _array = require("../../../helpers/array");
var _unicode = require("../../../helpers/unicode");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _utils = require("../utils");
var _base = _interopRequireDefault(require("./_base"));
var _multipleSelect = _interopRequireDefault(require("../ui/multipleSelect"));
var _constants2 = require("../constants");
var _conditionRegisterer = require("../conditionRegisterer");
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
/**
 * @private
 * @class ValueComponent
 */
class ValueComponent extends _base.default {
  constructor(hotInstance, options) {
    super(hotInstance, {
      id: options.id,
      stateless: false
    });
    this.name = options.name;
    this.elements.push(new _multipleSelect.default(this.hot));
    this.registerHooks();
  }

  /**
   * Register all necessary hooks.
   *
   * @private
   */
  registerHooks() {
    this.getMultipleSelectElement().addLocalHook('keydown', event => this.onInputKeyDown(event));
  }

  /**
   * Set state of the component.
   *
   * @param {object} value The component value.
   */
  setState(value) {
    this.reset();
    if (value && value.command.key === _constants2.CONDITION_BY_VALUE) {
      const select = this.getMultipleSelectElement();
      select.setItems(value.itemsSnapshot);
      select.setValue(value.args[0]);
      select.setLocale(value.locale);
    }
  }

  /**
   * Export state of the component (get selected filter and filter arguments).
   *
   * @returns {object} Returns object where `command` key keeps used condition filter and `args` key its arguments.
   */
  getState() {
    const select = this.getMultipleSelectElement();
    const availableItems = select.getItems();
    return {
      command: {
        key: select.isSelectedAllValues() || !availableItems.length ? _constants2.CONDITION_NONE : _constants2.CONDITION_BY_VALUE
      },
      args: [select.getValue()],
      itemsSnapshot: availableItems
    };
  }

  /**
   * Update state of component.
   *
   * @param {object} stateInfo Information about state containing stack of edited column,
   * stack of dependent conditions, data factory and optional condition arguments change. It's described by object containing keys:
   * `editedConditionStack`, `dependentConditionStacks`, `visibleDataFactory` and `conditionArgsChange`.
   */
  updateState(stateInfo) {
    const updateColumnState = (physicalColumn, conditions, conditionArgsChange, filteredRowsFactory, conditionsStack) => {
      const [firstByValueCondition] = (0, _array.arrayFilter)(conditions, condition => condition.name === _constants2.CONDITION_BY_VALUE);
      const state = {};
      const defaultBlankCellValue = this.hot.getTranslatedPhrase(C.FILTERS_VALUES_BLANK_CELLS);
      if (firstByValueCondition) {
        const rowValues = (0, _utils.unifyColumnValues)((0, _array.arrayMap)(filteredRowsFactory(physicalColumn, conditionsStack), row => row.value));
        if (conditionArgsChange) {
          firstByValueCondition.args[0] = conditionArgsChange;
        }
        const selectedValues = [];
        const itemsSnapshot = (0, _utils.intersectValues)(rowValues, firstByValueCondition.args[0], defaultBlankCellValue, item => {
          if (item.checked) {
            selectedValues.push(item.value);
          }
        });
        const column = stateInfo.editedConditionStack.column;
        state.locale = this.hot.getCellMeta(0, column).locale;
        state.args = [selectedValues];
        state.command = (0, _conditionRegisterer.getConditionDescriptor)(_constants2.CONDITION_BY_VALUE);
        state.itemsSnapshot = itemsSnapshot;
      } else {
        state.args = [];
        state.command = (0, _conditionRegisterer.getConditionDescriptor)(_constants2.CONDITION_NONE);
      }
      this.state.setValueAtIndex(physicalColumn, state);
    };
    updateColumnState(stateInfo.editedConditionStack.column, stateInfo.editedConditionStack.conditions, stateInfo.conditionArgsChange, stateInfo.filteredRowsFactory);

    // Update the next "by_value" component (filter column conditions added after this condition).
    // Its list of values has to be updated. As the new values by default are unchecked,
    // the further component update is unnecessary.
    if (stateInfo.dependentConditionStacks.length) {
      updateColumnState(stateInfo.dependentConditionStacks[0].column, stateInfo.dependentConditionStacks[0].conditions, stateInfo.conditionArgsChange, stateInfo.filteredRowsFactory, stateInfo.editedConditionStack);
    }
  }

  /**
   * Get multiple select element.
   *
   * @returns {MultipleSelectUI}
   */
  getMultipleSelectElement() {
    return this.elements.filter(element => element instanceof _multipleSelect.default)[0];
  }

  /**
   * Get object descriptor for menu item entry.
   *
   * @returns {object}
   */
  getMenuItemDescriptor() {
    return {
      key: this.id,
      name: this.name,
      isCommand: false,
      disableSelection: true,
      hidden: () => this.isHidden(),
      renderer: (hot, wrapper, row, col, prop, value) => {
        (0, _element.addClass)(wrapper.parentNode, 'htFiltersMenuValue');
        const label = this.hot.rootDocument.createElement('div');
        (0, _element.addClass)(label, 'htFiltersMenuLabel');
        label.textContent = value;
        wrapper.appendChild(label);
        if (!wrapper.parentNode.hasAttribute('ghost-table')) {
          (0, _array.arrayEach)(this.elements, ui => wrapper.appendChild(ui.element));
        }
        return wrapper;
      }
    };
  }

  /**
   * Reset elements to their initial state.
   */
  reset() {
    const defaultBlankCellValue = this.hot.getTranslatedPhrase(C.FILTERS_VALUES_BLANK_CELLS);
    const values = (0, _utils.unifyColumnValues)(this._getColumnVisibleValues());
    const items = (0, _utils.intersectValues)(values, values, defaultBlankCellValue);
    this.getMultipleSelectElement().setItems(items);
    super.reset();
    this.getMultipleSelectElement().setValue(values);
    const selectedColumn = this.hot.getPlugin('filters').getSelectedColumn();
    if (selectedColumn !== null) {
      this.getMultipleSelectElement().setLocale(this.hot.getCellMeta(0, selectedColumn.visualIndex).locale);
    }
  }

  /**
   * Key down listener.
   *
   * @private
   * @param {Event} event The DOM event object.
   */
  onInputKeyDown(event) {
    if ((0, _unicode.isKey)(event.keyCode, 'ESCAPE')) {
      this.runLocalHooks('cancel');
      (0, _event.stopImmediatePropagation)(event);
    }
  }

  /**
   * Get data for currently selected column.
   *
   * @returns {Array}
   * @private
   */
  _getColumnVisibleValues() {
    const selectedColumn = this.hot.getPlugin('filters').getSelectedColumn();
    if (selectedColumn === null) {
      return [];
    }
    return (0, _array.arrayMap)(this.hot.getDataAtCol(selectedColumn.visualIndex), v => (0, _utils.toEmptyString)(v));
  }
}
var _default = ValueComponent;
exports.default = _default;