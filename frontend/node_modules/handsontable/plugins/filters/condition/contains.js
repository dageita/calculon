"use strict";

exports.__esModule = true;
exports.condition = condition;
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _mixed = require("../../../helpers/mixed");
var _conditionRegisterer = require("../conditionRegisterer");
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const CONDITION_NAME = 'contains';

/**
 * @param {object} dataRow The object which holds and describes the single cell value.
 * @param {Array} inputValues An array of values to compare with.
 * @param {*} inputValues."0" A value to check if it occurs in the row's data.
 * @returns {boolean}
 */
exports.CONDITION_NAME = CONDITION_NAME;
function condition(dataRow, _ref) {
  let [value] = _ref;
  return (0, _mixed.stringify)(dataRow.value).toLocaleLowerCase(dataRow.meta.locale).indexOf((0, _mixed.stringify)(value)) >= 0;
}
(0, _conditionRegisterer.registerCondition)(CONDITION_NAME, condition, {
  name: C.FILTERS_CONDITIONS_CONTAINS,
  inputsCount: 1,
  showOperators: true
});