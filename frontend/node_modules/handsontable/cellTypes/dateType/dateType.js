"use strict";

exports.__esModule = true;
var _dateEditor = require("../../editors/dateEditor");
var _autocompleteRenderer = require("../../renderers/autocompleteRenderer");
var _dateValidator = require("../../validators/dateValidator");
const CELL_TYPE = 'date';
exports.CELL_TYPE = CELL_TYPE;
const DateCellType = {
  CELL_TYPE,
  editor: _dateEditor.DateEditor,
  // displays small gray arrow on right side of the cell
  renderer: _autocompleteRenderer.autocompleteRenderer,
  validator: _dateValidator.dateValidator
};
exports.DateCellType = DateCellType;