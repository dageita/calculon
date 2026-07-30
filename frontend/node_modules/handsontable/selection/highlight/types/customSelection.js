"use strict";

exports.__esModule = true;
var _constants = require("../constants");
var _visualSelection = _interopRequireDefault(require("../visualSelection"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
/**
 * Creates the new instance of Selection responsible for highlighting currently selected cell. This type of selection
 * can present on the table only one at the time.
 *
 * @param {object} highlightParams A configuration object to create a highlight.
 * @param {object} highlightParams.border Border configuration.
 * @param {object} highlightParams.visualCellRange Function to translate visual to renderable coords.
 * @returns {Selection}
 */
function createHighlight(_ref) {
  let {
    border,
    visualCellRange,
    ...restOptions
  } = _ref;
  const s = new _visualSelection.default({
    ...border,
    ...restOptions,
    selectionType: _constants.CUSTOM_SELECTION_TYPE
  }, visualCellRange);
  return s;
}
var _default = createHighlight;
exports.default = _default;