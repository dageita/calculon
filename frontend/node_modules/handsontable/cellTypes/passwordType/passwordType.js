"use strict";

exports.__esModule = true;
var _passwordEditor = require("../../editors/passwordEditor");
var _passwordRenderer = require("../../renderers/passwordRenderer");
const CELL_TYPE = 'password';
exports.CELL_TYPE = CELL_TYPE;
const PasswordCellType = {
  CELL_TYPE,
  editor: _passwordEditor.PasswordEditor,
  renderer: _passwordRenderer.passwordRenderer,
  copyable: false
};
exports.PasswordCellType = PasswordCellType;