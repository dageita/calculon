"use strict";

exports.__esModule = true;
exports.createHighlight = createHighlight;
var _staticRegister = _interopRequireDefault(require("./../../../utils/staticRegister"));
var _constants = require("../constants");
var _activeHeader = _interopRequireDefault(require("./activeHeader"));
var _area = _interopRequireDefault(require("./area"));
var _cell = _interopRequireDefault(require("./cell"));
var _customSelection = _interopRequireDefault(require("./customSelection"));
var _fill = _interopRequireDefault(require("./fill"));
var _header = _interopRequireDefault(require("./header"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
const {
  register,
  getItem
} = (0, _staticRegister.default)('highlight/types');
register(_constants.ACTIVE_HEADER_TYPE, _activeHeader.default);
register(_constants.AREA_TYPE, _area.default);
register(_constants.CELL_TYPE, _cell.default);
register(_constants.CUSTOM_SELECTION_TYPE, _customSelection.default);
register(_constants.FILL_TYPE, _fill.default);
register(_constants.HEADER_TYPE, _header.default);

/**
 * @param {string} highlightType The selection type.
 * @param {object} options The selection options.
 * @returns {Selection}
 */
function createHighlight(highlightType, options) {
  return getItem(highlightType)({
    type: highlightType,
    ...options
  });
}