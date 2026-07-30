"use strict";

exports.__esModule = true;
var _constants = require("../constants");
var _visualSelection = _interopRequireDefault(require("../visualSelection"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
/**
 * Creates the new instance of Selection responsible for highlighting area of the selected multiple cells.
 *
 * @param {object} highlightParams A configuration object to create a highlight.
 * @param {number} highlightParams.layerLevel Layer level.
 * @param {object} highlightParams.areaCornerVisible Function to determine if area's corner should be visible.
 * @returns {Selection}
 */
function createHighlight(_ref) {
  let {
    layerLevel,
    areaCornerVisible,
    ...restOptions
  } = _ref;
  const s = new _visualSelection.default({
    className: 'area',
    markIntersections: true,
    layerLevel: Math.min(layerLevel, 7),
    border: {
      width: 1,
      color: '#4b89ff',
      cornerVisible: areaCornerVisible
    },
    ...restOptions,
    selectionType: _constants.AREA_TYPE
  });
  return s;
}
var _default = createHighlight;
exports.default = _default;