"use strict";

exports.__esModule = true;
exports.addItem = addItem;
exports.predefinedItems = predefinedItems;
var _object = require("../../helpers/object");
var _alignment = _interopRequireWildcard(require("./predefinedItems/alignment"));
exports.ALIGNMENT = _alignment.KEY;
var _clearColumn = _interopRequireWildcard(require("./predefinedItems/clearColumn"));
exports.CLEAR_COLUMN = _clearColumn.KEY;
var _columnLeft = _interopRequireWildcard(require("./predefinedItems/columnLeft"));
exports.COLUMN_LEFT = _columnLeft.KEY;
var _columnRight = _interopRequireWildcard(require("./predefinedItems/columnRight"));
exports.COLUMN_RIGHT = _columnRight.KEY;
var _readOnly = _interopRequireWildcard(require("./predefinedItems/readOnly"));
exports.READ_ONLY = _readOnly.KEY;
var _redo = _interopRequireWildcard(require("./predefinedItems/redo"));
exports.REDO = _redo.KEY;
var _removeColumn = _interopRequireWildcard(require("./predefinedItems/removeColumn"));
exports.REMOVE_COLUMN = _removeColumn.KEY;
var _removeRow = _interopRequireWildcard(require("./predefinedItems/removeRow"));
exports.REMOVE_ROW = _removeRow.KEY;
var _rowAbove = _interopRequireWildcard(require("./predefinedItems/rowAbove"));
exports.ROW_ABOVE = _rowAbove.KEY;
var _rowBelow = _interopRequireWildcard(require("./predefinedItems/rowBelow"));
exports.ROW_BELOW = _rowBelow.KEY;
var _separator = _interopRequireWildcard(require("./predefinedItems/separator"));
exports.SEPARATOR = _separator.KEY;
var _noItems = _interopRequireWildcard(require("./predefinedItems/noItems"));
exports.NO_ITEMS = _noItems.KEY;
var _undo = _interopRequireWildcard(require("./predefinedItems/undo"));
exports.UNDO = _undo.KEY;
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const ITEMS = [_rowAbove.KEY, _rowBelow.KEY, _columnLeft.KEY, _columnRight.KEY, _clearColumn.KEY, _removeRow.KEY, _removeColumn.KEY, _undo.KEY, _redo.KEY, _readOnly.KEY, _alignment.KEY, _separator.KEY, _noItems.KEY];
exports.ITEMS = ITEMS;
const _predefinedItems = {
  [_separator.KEY]: _separator.default,
  [_noItems.KEY]: _noItems.default,
  [_rowAbove.KEY]: _rowAbove.default,
  [_rowBelow.KEY]: _rowBelow.default,
  [_columnLeft.KEY]: _columnLeft.default,
  [_columnRight.KEY]: _columnRight.default,
  [_clearColumn.KEY]: _clearColumn.default,
  [_removeRow.KEY]: _removeRow.default,
  [_removeColumn.KEY]: _removeColumn.default,
  [_undo.KEY]: _undo.default,
  [_redo.KEY]: _redo.default,
  [_readOnly.KEY]: _readOnly.default,
  [_alignment.KEY]: _alignment.default
};

/**
 * Gets new object with all predefined menu items.
 *
 * @returns {object}
 */
function predefinedItems() {
  const items = {};
  (0, _object.objectEach)(_predefinedItems, (itemFactory, key) => {
    items[key] = itemFactory();
  });
  return items;
}

/**
 * Add new predefined menu item to the collection.
 *
 * @param {string} key Menu command id.
 * @param {object} item Object command descriptor.
 */
function addItem(key, item) {
  if (ITEMS.indexOf(key) === -1) {
    _predefinedItems[key] = item;
  }
}