"use strict";

exports.__esModule = true;
require("core-js/modules/es.array.push.js");
var _element = require("../../../helpers/dom/element");
var _event = require("../../../helpers/dom/event");
var _array = require("../../../helpers/array");
var _unicode = require("../../../helpers/unicode");
var _object = require("../../../helpers/object");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _base = _interopRequireDefault(require("./_base"));
var _constants2 = _interopRequireWildcard(require("../constants"));
var _input = _interopRequireDefault(require("../ui/input"));
var _select = _interopRequireDefault(require("../ui/select"));
var _conditionRegisterer = require("../conditionRegisterer");
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
/**
 * @private
 * @class ConditionComponent
 */
class ConditionComponent extends _base.default {
  constructor(hotInstance, options) {
    super(hotInstance, {
      id: options.id,
      stateless: false
    });
    this.name = options.name;
    this.addSeparator = options.addSeparator;
    this.elements.push(new _select.default(this.hot, {
      menuContainer: options.menuContainer
    }));
    this.elements.push(new _input.default(this.hot, {
      placeholder: C.FILTERS_BUTTONS_PLACEHOLDER_VALUE
    }));
    this.elements.push(new _input.default(this.hot, {
      placeholder: C.FILTERS_BUTTONS_PLACEHOLDER_SECOND_VALUE
    }));
    this.registerHooks();
  }

  /**
   * Register all necessary hooks.
   *
   * @private
   */
  registerHooks() {
    this.getSelectElement().addLocalHook('select', command => this.onConditionSelect(command));
    this.getSelectElement().addLocalHook('afterClose', () => this.onSelectUIClosed());
    (0, _array.arrayEach)(this.getInputElements(), input => {
      input.addLocalHook('keydown', event => this.onInputKeyDown(event));
    });
  }

  /**
   * Set state of the component.
   *
   * @param {object} value State to restore.
   */
  setState(value) {
    this.reset();
    if (!value) {
      return;
    }
    const copyOfCommand = (0, _object.clone)(value.command);
    if (copyOfCommand.name.startsWith(C.FILTERS_CONDITIONS_NAMESPACE)) {
      copyOfCommand.name = this.hot.getTranslatedPhrase(copyOfCommand.name);
    }
    this.getSelectElement().setValue(copyOfCommand);
    (0, _array.arrayEach)(value.args, (arg, index) => {
      if (index > copyOfCommand.inputsCount - 1) {
        return false;
      }
      const element = this.getInputElement(index);
      element.setValue(arg);
      element[copyOfCommand.inputsCount > index ? 'show' : 'hide']();
      if (!index) {
        setTimeout(() => element.focus(), 10);
      }
    });
  }

  /**
   * Export state of the component (get selected filter and filter arguments).
   *
   * @returns {object} Returns object where `command` key keeps used condition filter and `args` key its arguments.
   */
  getState() {
    const command = this.getSelectElement().getValue() || (0, _conditionRegisterer.getConditionDescriptor)(_constants2.CONDITION_NONE);
    const args = [];
    (0, _array.arrayEach)(this.getInputElements(), (element, index) => {
      if (command.inputsCount > index) {
        args.push(element.getValue());
      }
    });
    return {
      command,
      args
    };
  }

  /**
   * Update state of component.
   *
   * @param {object} condition The condition object.
   * @param {object} condition.command The command object with condition name as `key` property.
   * @param {Array} condition.args An array of values to compare.
   * @param {number} column Physical column index.
   */
  updateState(condition, column) {
    const command = condition ? (0, _conditionRegisterer.getConditionDescriptor)(condition.name) : (0, _conditionRegisterer.getConditionDescriptor)(_constants2.CONDITION_NONE);
    this.state.setValueAtIndex(column, {
      command,
      args: condition ? condition.args : []
    });
    if (!condition) {
      (0, _array.arrayEach)(this.getInputElements(), element => element.setValue(null));
    }
  }

  /**
   * Get select element.
   *
   * @returns {SelectUI}
   */
  getSelectElement() {
    return this.elements.filter(element => element instanceof _select.default)[0];
  }

  /**
   * Get input element.
   *
   * @param {number} index Index an array of elements.
   * @returns {InputUI}
   */
  getInputElement() {
    let index = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : 0;
    return this.getInputElements()[index];
  }

  /**
   * Get input elements.
   *
   * @returns {Array}
   */
  getInputElements() {
    return this.elements.filter(element => element instanceof _input.default);
  }

  /**
   * Get menu object descriptor.
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
        (0, _element.addClass)(wrapper.parentNode, 'htFiltersMenuCondition');
        if (this.addSeparator) {
          (0, _element.addClass)(wrapper.parentNode, 'border');
        }
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
    const selectedColumn = this.hot.getPlugin('filters').getSelectedColumn();
    let items = [(0, _conditionRegisterer.getConditionDescriptor)(_constants2.CONDITION_NONE)];
    if (selectedColumn !== null) {
      const {
        visualIndex
      } = selectedColumn;
      items = (0, _constants2.default)(this.hot.getDataType(0, visualIndex, this.hot.countRows(), visualIndex));
    }
    (0, _array.arrayEach)(this.getInputElements(), element => element.hide());
    this.getSelectElement().setItems(items);
    super.reset();
    // Select element as default 'None'
    this.getSelectElement().setValue(items[0]);
  }

  /**
   * On condition select listener.
   *
   * @private
   * @param {object} command Menu item object (command).
   */
  onConditionSelect(command) {
    (0, _array.arrayEach)(this.getInputElements(), (element, index) => {
      element[command.inputsCount > index ? 'show' : 'hide']();
      if (index === 0) {
        setTimeout(() => element.focus(), 10);
      }
    });
    this.runLocalHooks('change', command);
  }

  /**
   * On component SelectUI closed listener.
   *
   * @private
   */
  onSelectUIClosed() {
    this.runLocalHooks('afterClose');
  }

  /**
   * Key down listener.
   *
   * @private
   * @param {Event} event The DOM event object.
   */
  onInputKeyDown(event) {
    if ((0, _unicode.isKey)(event.keyCode, 'ENTER')) {
      this.runLocalHooks('accept');
      (0, _event.stopImmediatePropagation)(event);
    } else if ((0, _unicode.isKey)(event.keyCode, 'ESCAPE')) {
      this.runLocalHooks('cancel');
      (0, _event.stopImmediatePropagation)(event);
    }
  }
}
var _default = ConditionComponent;
exports.default = _default;