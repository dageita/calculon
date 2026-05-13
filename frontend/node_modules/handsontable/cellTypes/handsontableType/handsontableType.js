"use strict";

exports.__esModule = true;
var _handsontableEditor = require("../../editors/handsontableEditor");
var _autocompleteRenderer = require("../../renderers/autocompleteRenderer");
const CELL_TYPE = 'handsontable';
exports.CELL_TYPE = CELL_TYPE;
const HandsontableCellType = {
  CELL_TYPE,
  editor: _handsontableEditor.HandsontableEditor,
  // displays small gray arrow on right side of the cell
  renderer: _autocompleteRenderer.autocompleteRenderer
};
exports.HandsontableCellType = HandsontableCellType;