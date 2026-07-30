"use strict";

exports.__esModule = true;
var _textEditor = require("../textEditor");
const EDITOR_TYPE = 'numeric';

/**
 * @private
 * @class NumericEditor
 */
exports.EDITOR_TYPE = EDITOR_TYPE;
class NumericEditor extends _textEditor.TextEditor {
  static get EDITOR_TYPE() {
    return EDITOR_TYPE;
  }
}
exports.NumericEditor = NumericEditor;