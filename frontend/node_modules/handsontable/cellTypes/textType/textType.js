"use strict";

exports.__esModule = true;
var _textEditor = require("../../editors/textEditor");
var _textRenderer = require("../../renderers/textRenderer");
const CELL_TYPE = 'text';
exports.CELL_TYPE = CELL_TYPE;
const TextCellType = {
  CELL_TYPE,
  editor: _textEditor.TextEditor,
  renderer: _textRenderer.textRenderer
};
exports.TextCellType = TextCellType;