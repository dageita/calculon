"use strict";

exports.__esModule = true;
exports.normalizeRanges = normalizeRanges;
require("core-js/modules/es.array.push.js");
require("core-js/modules/es.error.cause.js");
var _array = require("../../helpers/array");
var _number = require("../../helpers/number");
function _classPrivateMethodInitSpec(obj, privateSet) { _checkPrivateRedeclaration(obj, privateSet); privateSet.add(obj); }
function _classPrivateFieldInitSpec(obj, privateMap, value) { _checkPrivateRedeclaration(obj, privateMap); privateMap.set(obj, value); }
function _checkPrivateRedeclaration(obj, privateCollection) { if (privateCollection.has(obj)) { throw new TypeError("Cannot initialize the same private elements twice on an object"); } }
function _classPrivateMethodGet(receiver, privateSet, fn) { if (!privateSet.has(receiver)) { throw new TypeError("attempted to get private field on non-instance"); } return fn; }
function _classPrivateFieldGet(receiver, privateMap) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "get"); return _classApplyDescriptorGet(receiver, descriptor); }
function _classApplyDescriptorGet(receiver, descriptor) { if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }
function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "set"); _classApplyDescriptorSet(receiver, descriptor, value); return value; }
function _classExtractFieldDescriptor(receiver, privateMap, action) { if (!privateMap.has(receiver)) { throw new TypeError("attempted to " + action + " private field on non-instance"); } return privateMap.get(receiver); }
function _classApplyDescriptorSet(receiver, descriptor, value) { if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } }
/**
 * The utils class produces the selection ranges in the `{startRow, startCol, endRow, endCol}` format
 * based on the current table selection. The CopyPaste plugin consumes that ranges to generate
 * appropriate data ready to copy to the clipboard.
 *
 * @private
 */
var _selectedRange = /*#__PURE__*/new WeakMap();
var _countRows = /*#__PURE__*/new WeakMap();
var _countColumns = /*#__PURE__*/new WeakMap();
var _rowsLimit = /*#__PURE__*/new WeakMap();
var _columnsLimit = /*#__PURE__*/new WeakMap();
var _countColumnHeaders = /*#__PURE__*/new WeakMap();
var _trimColumnsRange = /*#__PURE__*/new WeakSet();
var _trimRowsRange = /*#__PURE__*/new WeakSet();
class CopyableRangesFactory {
  /* eslint-disable jsdoc/require-description-complete-sentence */
  /**
   * @param {{
   *   countRows: function(): number,
   *   countColumns: function(): number,
   *   rowsLimit: function(): number,
   *   columnsLimit: function(): number,
   *   countColumnHeaders: function(): number
   * }} dependencies The utils class dependencies.
   */
  constructor(_ref) {
    let {
      countRows,
      countColumns,
      rowsLimit,
      columnsLimit,
      countColumnHeaders
    } = _ref;
    /**
     * Trimmed the rows range to the limit.
     *
     * @param {*} startRow The lowest row index in the range.
     * @param {*} endRow The highest row index in the range.
     * @returns {number} Returns trimmed row index if it exceeds the limit.
     */
    _classPrivateMethodInitSpec(this, _trimRowsRange);
    /**
     * Trimmed the columns range to the limit.
     *
     * @param {*} startColumn The lowest column index in the range.
     * @param {*} endColumn The highest column index in the range.
     * @returns {number} Returns trimmed column index if it exceeds the limit.
     */
    _classPrivateMethodInitSpec(this, _trimColumnsRange);
    /**
     * @type {CellRange}
     */
    _classPrivateFieldInitSpec(this, _selectedRange, {
      writable: true,
      value: void 0
    });
    /**
     * @type {function(): number}
     */
    _classPrivateFieldInitSpec(this, _countRows, {
      writable: true,
      value: void 0
    });
    /**
     * @type {function(): number}
     */
    _classPrivateFieldInitSpec(this, _countColumns, {
      writable: true,
      value: void 0
    });
    /**
     * @type {function(): number}
     */
    _classPrivateFieldInitSpec(this, _rowsLimit, {
      writable: true,
      value: void 0
    });
    /**
     * @type {function(): number}
     */
    _classPrivateFieldInitSpec(this, _columnsLimit, {
      writable: true,
      value: void 0
    });
    /**
     * @type {function(): number}
     */
    _classPrivateFieldInitSpec(this, _countColumnHeaders, {
      writable: true,
      value: void 0
    });
    _classPrivateFieldSet(this, _countRows, countRows);
    _classPrivateFieldSet(this, _countColumns, countColumns);
    _classPrivateFieldSet(this, _rowsLimit, rowsLimit);
    _classPrivateFieldSet(this, _columnsLimit, columnsLimit);
    _classPrivateFieldSet(this, _countColumnHeaders, countColumnHeaders);
  }
  /* eslint-enable jsdoc/require-description-complete-sentence */

  /**
   * Sets the selection range to be processed.
   *
   * @param {CellRange} selectedRange The selection range represented by the CellRange class.
   */
  setSelectedRange(selectedRange) {
    _classPrivateFieldSet(this, _selectedRange, selectedRange);
  }

  /**
   * Returns a new coords object within the dataset range (cells) with `startRow`, `startCol`, `endRow`
   * and `endCol` keys.
   *
   * @returns {{startRow: number, startCol: number, endRow: number, endCol: number} | null}
   */
  getCellsRange() {
    if (_classPrivateFieldGet(this, _countRows).call(this) === 0 || _classPrivateFieldGet(this, _countColumns).call(this) === 0) {
      return null;
    }
    const {
      row: startRow,
      col: startCol
    } = _classPrivateFieldGet(this, _selectedRange).getTopStartCorner();
    const {
      row: endRow,
      col: endCol
    } = _classPrivateFieldGet(this, _selectedRange).getBottomEndCorner();
    const finalEndRow = _classPrivateMethodGet(this, _trimRowsRange, _trimRowsRange2).call(this, startRow, endRow);
    const finalEndCol = _classPrivateMethodGet(this, _trimColumnsRange, _trimColumnsRange2).call(this, startCol, endCol);
    const isRangeTrimmed = endRow !== finalEndRow || endCol !== finalEndCol;
    return {
      isRangeTrimmed,
      startRow,
      startCol,
      endRow: finalEndRow,
      endCol: finalEndCol
    };
  }

  /**
   * Returns a new coords object within the most-bottom column headers range with `startRow`,
   * `startCol`, `endRow` and `endCol` keys.
   *
   * @returns {{startRow: number, startCol: number, endRow: number, endCol: number} | null}
   */
  getMostBottomColumnHeadersRange() {
    if (_classPrivateFieldGet(this, _countColumns).call(this) === 0 || _classPrivateFieldGet(this, _countColumnHeaders).call(this) === 0) {
      return null;
    }
    const {
      col: startCol
    } = _classPrivateFieldGet(this, _selectedRange).getTopStartCorner();
    const {
      col: endCol
    } = _classPrivateFieldGet(this, _selectedRange).getBottomEndCorner();
    const finalEndCol = _classPrivateMethodGet(this, _trimColumnsRange, _trimColumnsRange2).call(this, startCol, endCol);
    const isRangeTrimmed = endCol !== finalEndCol;
    return {
      isRangeTrimmed,
      startRow: -1,
      startCol,
      endRow: -1,
      endCol: finalEndCol
    };
  }

  /**
   * Returns a new coords object within all column headers layers (including nested headers) range with
   * `startRow`, `startCol`, `endRow` and `endCol` keys.
   *
   * @returns {{startRow: number, startCol: number, endRow: number, endCol: number} | null}
   */
  getAllColumnHeadersRange() {
    if (_classPrivateFieldGet(this, _countColumns).call(this) === 0 || _classPrivateFieldGet(this, _countColumnHeaders).call(this) === 0) {
      return null;
    }
    const {
      col: startCol
    } = _classPrivateFieldGet(this, _selectedRange).getTopStartCorner();
    const {
      col: endCol
    } = _classPrivateFieldGet(this, _selectedRange).getBottomEndCorner();
    const finalEndCol = _classPrivateMethodGet(this, _trimColumnsRange, _trimColumnsRange2).call(this, startCol, endCol);
    const isRangeTrimmed = endCol !== finalEndCol;
    return {
      isRangeTrimmed,
      startRow: -_classPrivateFieldGet(this, _countColumnHeaders).call(this),
      startCol,
      endRow: -1,
      endCol: finalEndCol
    };
  }
}

/**
 * Returns an object with `rows` and `columns` keys. The arrays contains sorted indexes
 * generated according to the given `ranges` array.
 *
 * @param {Array<{startRow: number, startCol: number, endRow: number, endCol: number}>} ranges The range to process.
 * @returns {{rows: number[], columns: number[]}}
 */
exports.CopyableRangesFactory = CopyableRangesFactory;
function _trimColumnsRange2(startColumn, endColumn) {
  return Math.min(endColumn, Math.max(startColumn + _classPrivateFieldGet(this, _columnsLimit).call(this) - 1, startColumn));
}
function _trimRowsRange2(startRow, endRow) {
  return Math.min(endRow, Math.max(startRow + _classPrivateFieldGet(this, _rowsLimit).call(this) - 1, startRow));
}
function normalizeRanges(ranges) {
  const rows = [];
  const columns = [];
  (0, _array.arrayEach)(ranges, range => {
    const minRow = Math.min(range.startRow, range.endRow);
    const maxRow = Math.max(range.startRow, range.endRow);
    (0, _number.rangeEach)(minRow, maxRow, row => {
      if (rows.indexOf(row) === -1) {
        rows.push(row);
      }
    });
    const minColumn = Math.min(range.startCol, range.endCol);
    const maxColumn = Math.max(range.startCol, range.endCol);
    (0, _number.rangeEach)(minColumn, maxColumn, column => {
      if (columns.indexOf(column) === -1) {
        columns.push(column);
      }
    });
  });
  return {
    rows,
    columns
  };
}