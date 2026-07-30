"use strict";

exports.__esModule = true;
var _checkboxEditor = require("../../editors/checkboxEditor");
var _checkboxRenderer = require("../../renderers/checkboxRenderer");
const CELL_TYPE = 'checkbox';
exports.CELL_TYPE = CELL_TYPE;
const CheckboxCellType = {
  CELL_TYPE,
  editor: _checkboxEditor.CheckboxEditor,
  renderer: _checkboxRenderer.checkboxRenderer
};
exports.CheckboxCellType = CheckboxCellType;