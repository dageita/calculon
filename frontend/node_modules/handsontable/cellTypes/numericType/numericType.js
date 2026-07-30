"use strict";

exports.__esModule = true;
var _numericEditor = require("../../editors/numericEditor");
var _numericRenderer = require("../../renderers/numericRenderer");
var _numericValidator = require("../../validators/numericValidator");
const CELL_TYPE = 'numeric';
exports.CELL_TYPE = CELL_TYPE;
const NumericCellType = {
  CELL_TYPE,
  editor: _numericEditor.NumericEditor,
  renderer: _numericRenderer.numericRenderer,
  validator: _numericValidator.numericValidator,
  dataType: 'number'
};
exports.NumericCellType = NumericCellType;