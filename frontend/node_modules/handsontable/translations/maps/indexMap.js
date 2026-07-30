"use strict";

exports.__esModule = true;
require("core-js/modules/es.array.push.js");
var _number = require("../../helpers/number");
var _object = require("../../helpers/object");
var _function = require("../../helpers/function");
var _localHooks = _interopRequireDefault(require("../../mixins/localHooks"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
/**
 * Map for storing mappings from an index to a value.
 *
 * @class IndexMap
 */
class IndexMap {
  constructor() {
    let initValueOrFn = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : null;
    /**
     * List of values for particular indexes.
     *
     * @private
     * @type {Array}
     */
    this.indexedValues = [];
    /**
     * Initial value or function for each existing index.
     *
     * @private
     * @type {*}
     */
    this.initValueOrFn = initValueOrFn;
  }

  /**
   * Get full list of values for particular indexes.
   *
   * @returns {Array}
   */
  getValues() {
    return this.indexedValues;
  }

  /**
   * Get value for the particular index.
   *
   * @param {number} index Index for which value is got.
   * @returns {*}
   */
  getValueAtIndex(index) {
    const values = this.indexedValues;
    if (index < values.length) {
      return values[index];
    }
  }

  /**
   * Set new values for particular indexes.
   *
   * Note: Please keep in mind that `change` hook triggered by the method may not update cache of a collection immediately.
   *
   * @param {Array} values List of set values.
   */
  setValues(values) {
    this.indexedValues = values.slice();
    this.runLocalHooks('change');
  }

  /**
   * Set new value for the particular index.
   *
   * @param {number} index The index.
   * @param {*} value The value to save.
   *
   * Note: Please keep in mind that it is not possible to set value beyond the map (not respecting already set
   * map's size). Please use the `setValues` method when you would like to extend the map.
   * Note: Please keep in mind that `change` hook triggered by the method may not update cache of a collection immediately.
   *
   * @returns {boolean}
   */
  setValueAtIndex(index, value) {
    if (index < this.indexedValues.length) {
      this.indexedValues[index] = value;
      this.runLocalHooks('change');
      return true;
    }
    return false;
  }

  /**
   * Clear all values to the defaults.
   */
  clear() {
    this.setDefaultValues();
  }

  /**
   * Get length of the index map.
   *
   * @returns {number}
   */
  getLength() {
    return this.getValues().length;
  }

  /**
   * Set default values for elements from `0` to `n`, where `n` is equal to the handled variable.
   *
   * Note: Please keep in mind that `change` hook triggered by the method may not update cache of a collection immediately.
   *
   * @private
   * @param {number} [length] Length of list.
   */
  setDefaultValues() {
    let length = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : this.indexedValues.length;
    this.indexedValues.length = 0;
    if ((0, _function.isFunction)(this.initValueOrFn)) {
      (0, _number.rangeEach)(length - 1, index => this.indexedValues.push(this.initValueOrFn(index)));
    } else {
      (0, _number.rangeEach)(length - 1, () => this.indexedValues.push(this.initValueOrFn));
    }
    this.runLocalHooks('change');
  }

  /**
   * Initialize list with default values for particular indexes.
   *
   * @private
   * @param {number} length New length of indexed list.
   * @returns {IndexMap}
   */
  init(length) {
    this.setDefaultValues(length);
    this.runLocalHooks('init');
    return this;
  }

  /**
   * Add values to the list.
   *
   * Note: Please keep in mind that `change` hook triggered by the method may not update cache of a collection immediately.
   *
   * @private
   */
  insert() {
    this.runLocalHooks('change');
  }

  /**
   * Remove values from the list.
   *
   * Note: Please keep in mind that `change` hook triggered by the method may not update cache of a collection immediately.
   *
   * @private
   */
  remove() {
    this.runLocalHooks('change');
  }

  /**
   * Destroys the Map instance.
   */
  destroy() {
    this.clearLocalHooks();
    this.indexedValues = null;
    this.initValueOrFn = null;
  }
}
exports.IndexMap = IndexMap;
(0, _object.mixin)(IndexMap, _localHooks.default);