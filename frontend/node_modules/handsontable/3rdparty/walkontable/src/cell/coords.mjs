import "core-js/modules/es.error.cause.js";
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
var _isRtl = /*#__PURE__*/new WeakMap();
/* eslint-disable jsdoc/require-description-complete-sentence */
/**
 * @description
 *
 * The `CellCoords` class holds the coordinates (`row`, `col`) of a single cell.
 *
 * It also contains methods for validating the coordinates
 * and retrieving them as an object.
 *
 * To import the `CellCoords` class:
 *
 * ```js
 * import Handsontable, { CellCoords } from '/handsontable';
 *
 * // or, using modules
 * import Handsontable, { CellCoords } from '/handsontable/base';
 * ```
 */
class CellCoords {
  constructor(row, column) {
    let isRtl = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : false;
    /**
     * A visual row index.
     *
     * @type {number}
     */
    _defineProperty(this, "row", null);
    /**
     * A visual column index.
     *
     * @type {number}
     */
    _defineProperty(this, "col", null);
    /**
     * @type {boolean}
     */
    _classPrivateFieldInitSpec(this, _isRtl, {
      writable: true,
      value: false
    });
    _classPrivateFieldSet(this, _isRtl, isRtl);
    if (typeof row !== 'undefined' && typeof column !== 'undefined') {
      this.row = row;
      this.col = column;
    }
  }

  /**
   * Checks if the coordinates in your `CellCoords` instance are valid
   * in the context of a given Walkontable instance.
   *
   * The `row` index:
   * - Can't be negative.
   * - Can't be higher than the total number of rows in the Walkontable instance.
   *
   * The `col` index:
   * - Can't be negative.
   * - Can't be higher than the total number of columns in the Walkontable instance.
   *
   * @param {Walkontable} wot A Walkontable instance.
   * @returns {boolean} `true`: The coordinates are valid.
   */
  isValid(wot) {
    // check if the row and column indexes are valid (0 or higher)
    if (this.row < 0 || this.col < 0) {
      return false;
    }
    // check if the selection fits in the total of rows and columns
    if (this.row >= wot.getSetting('totalRows') || this.col >= wot.getSetting('totalColumns')) {
      return false;
    }
    return true;
  }

  /**
   * Checks if another set of coordinates (`cellCoords`)
   * is equal to the coordinates in your `CellCoords` instance.
   *
   * @param {CellCoords} cellCoords Coordinates to check.
   * @returns {boolean}
   */
  isEqual(cellCoords) {
    if (cellCoords === this) {
      return true;
    }
    return this.row === cellCoords.row && this.col === cellCoords.col;
  }

  /**
   * Checks if another set of coordinates (`testedCoords`)
   * is south-east of the coordinates in your `CellCoords` instance.
   *
   * @param {CellCoords} testedCoords Coordinates to check.
   * @returns {boolean}
   */
  isSouthEastOf(testedCoords) {
    return this.row >= testedCoords.row && (_classPrivateFieldGet(this, _isRtl) ? this.col <= testedCoords.col : this.col >= testedCoords.col);
  }

  /**
   * Checks if another set of coordinates (`testedCoords`)
   * is north-west of the coordinates in your `CellCoords` instance.
   *
   * @param {CellCoords} testedCoords Coordinates to check.
   * @returns {boolean}
   */
  isNorthWestOf(testedCoords) {
    return this.row <= testedCoords.row && (_classPrivateFieldGet(this, _isRtl) ? this.col >= testedCoords.col : this.col <= testedCoords.col);
  }

  /**
   * Checks if another set of coordinates (`testedCoords`)
   * is south-west of the coordinates in your `CellCoords` instance.
   *
   * @param {CellCoords} testedCoords Coordinates to check.
   * @returns {boolean}
   */
  isSouthWestOf(testedCoords) {
    return this.row >= testedCoords.row && (_classPrivateFieldGet(this, _isRtl) ? this.col >= testedCoords.col : this.col <= testedCoords.col);
  }

  /**
   * Checks if another set of coordinates (`testedCoords`)
   * is north-east of the coordinates in your `CellCoords` instance.
   *
   * @param {CellCoords} testedCoords Coordinates to check.
   * @returns {boolean}
   */
  isNorthEastOf(testedCoords) {
    return this.row <= testedCoords.row && (_classPrivateFieldGet(this, _isRtl) ? this.col <= testedCoords.col : this.col >= testedCoords.col);
  }

  /**
   * Normalizes the coordinates in your `CellCoords` instance to the nearest valid position.
   *
   * Coordinates that point to headers (negative values) are normalized to `0`.
   *
   * @returns {CellCoords}
   */
  normalize() {
    this.row = this.row === null ? this.row : Math.max(this.row, 0);
    this.col = this.col === null ? this.col : Math.max(this.col, 0);
    return this;
  }

  /**
   * Clones your `CellCoords` instance.
   *
   * @returns {CellCoords}
   */
  clone() {
    return new CellCoords(this.row, this.col, _classPrivateFieldGet(this, _isRtl));
  }

  /**
   * Converts your `CellCoords` instance into an object literal with `row` and `col` properties.
   *
   * @returns {{row: number, col: number}} An object literal with `row` and `col` properties.
   */
  toObject() {
    return {
      row: this.row,
      col: this.col
    };
  }
}
export default CellCoords;