"use strict";

exports.__esModule = true;
var _dropdownEditor = require("../../editors/dropdownEditor");
var _autocompleteRenderer = require("../../renderers/autocompleteRenderer");
var _autocompleteValidator = require("../../validators/autocompleteValidator");
const CELL_TYPE = 'dropdown';
exports.CELL_TYPE = CELL_TYPE;
const DropdownCellType = {
  CELL_TYPE,
  editor: _dropdownEditor.DropdownEditor,
  // displays small gray arrow on right side of the cell
  renderer: _autocompleteRenderer.autocompleteRenderer,
  validator: _autocompleteValidator.autocompleteValidator
};
exports.DropdownCellType = DropdownCellType;