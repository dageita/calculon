"use strict";

exports.__esModule = true;
exports.passwordRenderer = passwordRenderer;
var _textRenderer = require("../textRenderer");
var _element = require("../../helpers/dom/element");
var _number = require("../../helpers/number");
const RENDERER_TYPE = 'password';

/**
 * @private
 * @param {Core} instance The Handsontable instance.
 * @param {HTMLTableCellElement} TD The rendered cell element.
 * @param {number} row The visual row index.
 * @param {number} col The visual column index.
 * @param {number|string} prop The column property (passed when datasource is an array of objects).
 * @param {*} value The rendered value.
 * @param {object} cellProperties The cell meta object ({@see Core#getCellMeta}).
 */
exports.RENDERER_TYPE = RENDERER_TYPE;
function passwordRenderer(instance, TD, row, col, prop, value, cellProperties) {
  _textRenderer.textRenderer.apply(this, [instance, TD, row, col, prop, value, cellProperties]);
  const hashLength = cellProperties.hashLength || TD.innerHTML.length;
  const hashSymbol = cellProperties.hashSymbol || '*';
  let hash = '';
  (0, _number.rangeEach)(hashLength - 1, () => {
    hash += hashSymbol;
  });
  (0, _element.fastInnerHTML)(TD, hash);
}
passwordRenderer.RENDERER_TYPE = RENDERER_TYPE;