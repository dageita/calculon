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
 * @param {Function} highlightParams.cellCornerVisible Function to determine if cell's corner should be visible.
 * @returns {Selection}
 */
function createHighlight(_ref) {
  let {
    cellCornerVisible,
    ...restOptions
  } = _ref;
  const s = new _visualSelection.default({
    className: 'current',
    border: {
      width: 2,
      color: '#4b89ff',
      cornerVisible: cellCornerVisible
    },
    ...restOptions,
    selectionType: _constants.CELL_TYPE
  });
  return s;
}
var _default = createHighlight;
exports.default = _default;