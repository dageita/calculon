"use strict";

exports.__esModule = true;
var _constants = require("../constants");
var _visualSelection = _interopRequireDefault(require("../visualSelection"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
/**
 * @param {object} highlightParams A configuration object to create a highlight.
 * @param {string} highlightParams.activeHeaderClassName Highlighted headers' class name.
 * @returns {Selection}
 */
function createHighlight(_ref) {
  let {
    activeHeaderClassName,
    ...restOptions
  } = _ref;
  const s = new _visualSelection.default({
    highlightHeaderClassName: activeHeaderClassName,
    ...restOptions,
    selectionType: _constants.ACTIVE_HEADER_TYPE
  });
  return s;
}
var _default = createHighlight;
exports.default = _default;