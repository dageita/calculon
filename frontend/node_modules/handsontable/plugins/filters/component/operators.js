"use strict";

exports.__esModule = true;
require("core-js/modules/es.array.push.js");
require("core-js/modules/es.error.cause.js");
var _element = require("../../../helpers/dom/element");
var _array = require("../../../helpers/array");
var _templateLiteralTag = require("../../../helpers/templateLiteralTag");
var _base = _interopRequireDefault(require("./_base"));
var _logicalOperationRegisterer = require("../logicalOperationRegisterer");
var _conjunction = require("../logicalOperations/conjunction");
var _disjunction = require("../logicalOperations/disjunction");
var _disjunctionWithExtraCondition = require("../logicalOperations/disjunctionWithExtraCondition");
var _radioInput = _interopRequireDefault(require("../ui/radioInput"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
const SELECTED_AT_START_ELEMENT_INDEX = 0;

/**
 * @private
 * @class OperatorsComponent
 */
class OperatorsComponent extends _base.default {
  constructor(hotInstance, options) {
    super(hotInstance, {
      id: options.id,
      stateless: false
    });
    this.name = options.name;
    this.buildOperatorsElement();
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
      renderer: (hot, wrapper) => {
        (0, _element.addClass)(wrapper.parentNode, 'htFiltersMenuOperators');
        if (!wrapper.parentNode.hasAttribute('ghost-table')) {
          (0, _array.arrayEach)(this.elements, ui => wrapper.appendChild(ui.element));
        }
        return wrapper;
      }
    };
  }

  /**
   * Add RadioInputUI elements to component.
   *
   * @private
   */
  buildOperatorsElement() {
    const operationKeys = [_conjunction.OPERATION_ID, _disjunction.OPERATION_ID];
    (0, _array.arrayEach)(operationKeys, operation => {
      const radioInput = new _radioInput.default(this.hot, {
        name: 'operator',
        label: {
          htmlFor: operation,
          textContent: (0, _logicalOperationRegisterer.getOperationName)(operation)
        },
        value: operation,
        checked: operation === operationKeys[SELECTED_AT_START_ELEMENT_INDEX],
        id: operation
      });
      radioInput.addLocalHook('change', event => this.onRadioInputChange(event));
      this.elements.push(radioInput);
    });
  }

  /**
   * Set state of operators component to check radio input at specific `index`.
   *
   * @param {number} searchedIndex Index of radio input to check.
   */
  setChecked(searchedIndex) {
    if (this.elements.length < searchedIndex) {
      throw Error((0, _templateLiteralTag.toSingleLine)`Radio button with index ${searchedIndex} doesn't exist.`);
    }
    (0, _array.arrayEach)(this.elements, (element, index) => {
      element.setChecked(index === searchedIndex);
    });
  }

  /**
   * Get `id` of active operator.
   *
   * @returns {string}
   */
  getActiveOperationId() {
    const operationElement = this.elements.find(element => element instanceof _radioInput.default && element.isChecked());
    if (operationElement) {
      return operationElement.getValue();
    }
    return _conjunction.OPERATION_ID;
  }

  /**
   * Export state of the component (get selected operator).
   *
   * @returns {string} Returns `id` of selected operator.
   */
  getState() {
    return this.getActiveOperationId();
  }

  /**
   * Set state of the component.
   *
   * @param {object} value State to restore.
   */
  setState(value) {
    this.reset();
    if (value && this.getActiveOperationId() !== value) {
      (0, _array.arrayEach)(this.elements, element => {
        element.setChecked(element.getValue() === value);
      });
    }
  }

  /**
   * Update state of component.
   *
   * @param {string} [operationId='conjunction'] Id of selected operation.
   * @param {number} column Physical column index.
   */
  updateState() {
    let operationId = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : _conjunction.OPERATION_ID;
    let column = arguments.length > 1 ? arguments[1] : undefined;
    let selectedOperationId = operationId;
    if (selectedOperationId === _disjunctionWithExtraCondition.OPERATION_ID) {
      selectedOperationId = _disjunction.OPERATION_ID;
    }
    this.state.setValueAtIndex(column, selectedOperationId);
  }

  /**
   * Reset elements to their initial state.
   */
  reset() {
    this.setChecked(SELECTED_AT_START_ELEMENT_INDEX);
  }

  /**
   * OnChange listener.
   *
   * @private
   * @param {Event} event The DOM event object.
   */
  onRadioInputChange(event) {
    this.setState(event.target.value);
  }
}
var _default = OperatorsComponent;
exports.default = _default;