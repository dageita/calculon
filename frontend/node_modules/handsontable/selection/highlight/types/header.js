"use strict";

exports.__esModule = true;
var _constants = require("../constants");
var _visualSelection = _interopRequireDefault(require("../visualSelection"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
/**
 * Creates the new instance of Selection, responsible for highlighting row and column headers. This type of selection
 * can occur multiple times.
 *
 * @param {object} highlightParams A configuration object to create a highlight.
 * @param {string} highlightParams.headerClassName Highlighted headers' class name.
 * @param {string} highlightParams.rowClassName Highlighted row' class name.
 * @param {string} highlightParams.columnClassName Highlighted column' class name.
 * @returns {Selection}
 */
function createHighlight(_ref) {
  let {
    headerClassName,
    rowClassName,
    columnClassName,
    ...restOptions
  } = _ref;
  const s = new _visualSelection.default({
    className: 'highlight',
    highlightHeaderClassName: headerClassName,
    highlightRowClassName: rowClassName,
    highlightColumnClassName: columnClassName,
    ...restOptions,
    highlightOnlyClosestHeader: true,
    selectionType: _constants.HEADER_TYPE
  });
  return s;
}
var _default = createHighlight;
exports.default = _default;