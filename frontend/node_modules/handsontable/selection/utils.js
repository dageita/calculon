"use strict";

exports.__esModule = true;
exports.detectSelectionType = detectSelectionType;
exports.isValidCoord = isValidCoord;
exports.normalizeSelectionFactory = normalizeSelectionFactory;
exports.transformSelectionToColumnDistance = transformSelectionToColumnDistance;
exports.transformSelectionToRowDistance = transformSelectionToRowDistance;
require("core-js/modules/es.error.cause.js");
require("core-js/modules/es.array.push.js");
var _src = require("./../3rdparty/walkontable/src");
var _array = require("./../helpers/array");
var _mixed = require("./../helpers/mixed");
const SELECTION_TYPE_UNRECOGNIZED = 0;
exports.SELECTION_TYPE_UNRECOGNIZED = SELECTION_TYPE_UNRECOGNIZED;
const SELECTION_TYPE_EMPTY = 1;
exports.SELECTION_TYPE_EMPTY = SELECTION_TYPE_EMPTY;
const SELECTION_TYPE_ARRAY = 2;
exports.SELECTION_TYPE_ARRAY = SELECTION_TYPE_ARRAY;
const SELECTION_TYPE_OBJECT = 3;
exports.SELECTION_TYPE_OBJECT = SELECTION_TYPE_OBJECT;
const SELECTION_TYPES = [SELECTION_TYPE_OBJECT, SELECTION_TYPE_ARRAY];
exports.SELECTION_TYPES = SELECTION_TYPES;
const ARRAY_TYPE_PATTERN = [['number'], ['number', 'string'], ['number', 'undefined'], ['number', 'string', 'undefined']];
const rootCall = Symbol('root');
const childCall = Symbol('child');

/**
 * Detect selection schema structure.
 *
 * @param {*} selectionRanges The selected range or and array of selected ranges. This type of data is produced by
 * `hot.getSelected()`, `hot.getSelectedLast()`, `hot.getSelectedRange()`
 * and `hot.getSelectedRangeLast()` methods.
 * @param {symbol} _callSymbol The symbol object which indicates source of the helper invocation.
 * @returns {number} Returns a number that specifies the type of detected selection schema. If selection schema type
 * is unrecognized than it returns `0`.
 */
function detectSelectionType(selectionRanges) {
  let _callSymbol = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : rootCall;
  if (_callSymbol !== rootCall && _callSymbol !== childCall) {
    throw new Error('The second argument is used internally only and cannot be overwritten.');
  }
  const isArray = Array.isArray(selectionRanges);
  const isRootCall = _callSymbol === rootCall;
  let result = SELECTION_TYPE_UNRECOGNIZED;
  if (isArray) {
    const firstItem = selectionRanges[0];
    if (selectionRanges.length === 0) {
      result = SELECTION_TYPE_EMPTY;
    } else if (isRootCall && firstItem instanceof _src.CellRange) {
      result = SELECTION_TYPE_OBJECT;
    } else if (isRootCall && Array.isArray(firstItem)) {
      result = detectSelectionType(firstItem, childCall);
    } else if (selectionRanges.length >= 2 && selectionRanges.length <= 4) {
      const isArrayType = !selectionRanges.some((value, index) => !ARRAY_TYPE_PATTERN[index].includes(typeof value));
      if (isArrayType) {
        result = SELECTION_TYPE_ARRAY;
      }
    }
  }
  return result;
}

/**
 * Factory function designed for normalization data schema from different data structures of the selection ranges.
 *
 * @param {number} type Selection type which will be processed.
 * @param {object} [options] The normalization options.
 * @param {boolean} [options.keepDirection=false] If `true`, the coordinates which contain the direction of the
 *                                                selected cells won't be changed. Otherwise, the selection will be
 *                                                normalized to values starting from top-left to bottom-right.
 * @param {Function} [options.propToCol] Pass the converting function (usually `datamap.propToCol`) if the column
 *                                       defined as props should be normalized to the numeric values.
 * @returns {number[]} Returns normalized data about selected range as an array (`[rowStart, columnStart, rowEnd, columnEnd]`).
 */
function normalizeSelectionFactory(type) {
  let {
    keepDirection = false,
    propToCol
  } = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};
  if (!SELECTION_TYPES.includes(type)) {
    throw new Error('Unsupported selection ranges schema type was provided.');
  }
  return function (selection) {
    const isObjectType = type === SELECTION_TYPE_OBJECT;
    let rowStart = isObjectType ? selection.from.row : selection[0];
    let columnStart = isObjectType ? selection.from.col : selection[1];
    let rowEnd = isObjectType ? selection.to.row : selection[2];
    let columnEnd = isObjectType ? selection.to.col : selection[3];
    if (typeof propToCol === 'function') {
      if (typeof columnStart === 'string') {
        columnStart = propToCol(columnStart);
      }
      if (typeof columnEnd === 'string') {
        columnEnd = propToCol(columnEnd);
      }
    }
    if ((0, _mixed.isUndefined)(rowEnd)) {
      rowEnd = rowStart;
    }
    if ((0, _mixed.isUndefined)(columnEnd)) {
      columnEnd = columnStart;
    }
    if (!keepDirection) {
      const origRowStart = rowStart;
      const origColumnStart = columnStart;
      const origRowEnd = rowEnd;
      const origColumnEnd = columnEnd;
      rowStart = Math.min(origRowStart, origRowEnd);
      columnStart = Math.min(origColumnStart, origColumnEnd);
      rowEnd = Math.max(origRowStart, origRowEnd);
      columnEnd = Math.max(origColumnStart, origColumnEnd);
    }
    return [rowStart, columnStart, rowEnd, columnEnd];
  };
}

/**
 * Function transform selection ranges (produced by `hot.getSelected()` and `hot.getSelectedRange()`) to normalized
 * data structure. It merges repeated ranges into consecutive coordinates. The returned structure
 * contains an array of arrays. The single item contains at index 0 visual column index from the selection was
 * started and at index 1 distance as a count of selected columns.
 *
 * @param {Array[]|CellRange[]} selectionRanges Selection ranges produced by Handsontable.
 * @returns {Array[]} Returns an array of arrays with ranges defines in that schema:
 *                   `[[visualColumnStart, distance], [visualColumnStart, distance], ...]`.
 *                   The column distances are always created starting from the left (zero index) to the
 *                   right (the latest column index).
 */
function transformSelectionToColumnDistance(selectionRanges) {
  const selectionType = detectSelectionType(selectionRanges);
  if (selectionType === SELECTION_TYPE_UNRECOGNIZED || selectionType === SELECTION_TYPE_EMPTY) {
    return [];
  }
  const selectionSchemaNormalizer = normalizeSelectionFactory(selectionType);
  const unorderedIndexes = new Set();

  // Iterate through all ranges and collect all column indexes which are not saved yet.
  (0, _array.arrayEach)(selectionRanges, selection => {
    const [, columnStart,, columnEnd] = selectionSchemaNormalizer(selection);
    const columnNonHeaderStart = Math.max(columnStart, 0);
    const amount = columnEnd - columnNonHeaderStart + 1;
    (0, _array.arrayEach)(Array.from(new Array(amount), (_, i) => columnNonHeaderStart + i), index => {
      if (!unorderedIndexes.has(index)) {
        unorderedIndexes.add(index);
      }
    });
  });

  // Sort indexes in ascending order to easily detecting non-consecutive columns.
  const orderedIndexes = Array.from(unorderedIndexes).sort((a, b) => a - b);
  const normalizedColumnRanges = (0, _array.arrayReduce)(orderedIndexes, (acc, visualColumnIndex, index, array) => {
    if (index !== 0 && visualColumnIndex === array[index - 1] + 1) {
      acc[acc.length - 1][1] += 1;
    } else {
      acc.push([visualColumnIndex, 1]);
    }
    return acc;
  }, []);
  return normalizedColumnRanges;
}

/**
 * Function transform selection ranges (produced by `hot.getSelected()` and `hot.getSelectedRange()`) to normalized
 * data structure. It merges repeated ranges into consecutive coordinates. The returned structure
 * contains an array of arrays. The single item contains at index 0 visual column index from the selection was
 * started and at index 1 distance as a count of selected columns.
 *
 * @param {Array[]|CellRange[]} selectionRanges Selection ranges produced by Handsontable.
 * @returns {Array[]} Returns an array of arrays with ranges defines in that schema:
 *                   `[[visualColumnStart, distance], [visualColumnStart, distance], ...]`.
 *                   The column distances are always created starting from the left (zero index) to the
 *                   right (the latest column index).
 */
function transformSelectionToRowDistance(selectionRanges) {
  const selectionType = detectSelectionType(selectionRanges);
  if (selectionType === SELECTION_TYPE_UNRECOGNIZED || selectionType === SELECTION_TYPE_EMPTY) {
    return [];
  }
  const selectionSchemaNormalizer = normalizeSelectionFactory(selectionType);
  const unorderedIndexes = new Set();

  // Iterate through all ranges and collect all column indexes which are not saved yet.
  (0, _array.arrayEach)(selectionRanges, selection => {
    const [rowStart,, rowEnd] = selectionSchemaNormalizer(selection);
    const rowNonHeaderStart = Math.max(rowStart, 0);
    const amount = rowEnd - rowNonHeaderStart + 1;
    (0, _array.arrayEach)(Array.from(new Array(amount), (_, i) => rowNonHeaderStart + i), index => {
      if (!unorderedIndexes.has(index)) {
        unorderedIndexes.add(index);
      }
    });
  });

  // Sort indexes in ascending order to easily detecting non-consecutive columns.
  const orderedIndexes = Array.from(unorderedIndexes).sort((a, b) => a - b);
  const normalizedRowRanges = (0, _array.arrayReduce)(orderedIndexes, (acc, rowIndex, index, array) => {
    if (index !== 0 && rowIndex === array[index - 1] + 1) {
      acc[acc.length - 1][1] += 1;
    } else {
      acc.push([rowIndex, 1]);
    }
    return acc;
  }, []);
  return normalizedRowRanges;
}

/**
 * Check if passed value can be treated as valid cell coordinate. The second argument is
 * used to check if the value doesn't exceed the defined max table rows/columns count.
 *
 * @param {number} coord The coordinate to validate (row index or column index).
 * @param {number} maxTableItemsCount The value that declares the maximum coordinate that is still validatable.
 * @returns {boolean}
 */
function isValidCoord(coord) {
  let maxTableItemsCount = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : Infinity;
  return typeof coord === 'number' && coord >= 0 && coord < maxTableItemsCount;
}