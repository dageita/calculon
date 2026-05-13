"use strict";

exports.__esModule = true;
exports.condition = condition;
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _conditionRegisterer = require("../conditionRegisterer");
var _after = require("./date/after");
var _before = require("./date/before");
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const CONDITION_NAME = 'between';

/**
 * @param {object} dataRow The object which holds and describes the single cell value.
 * @param {Array} inputValues An array of values to compare with.
 * @param {number} inputValues."0" The minimum value of the range.
 * @param {number} inputValues."1" The maximum value of the range.
 * @returns {boolean}
 */
exports.CONDITION_NAME = CONDITION_NAME;
function condition(dataRow, _ref) {
  let [from, to] = _ref;
  let fromValue = from;
  let toValue = to;
  if (dataRow.meta.type === 'numeric') {
    const _from = parseFloat(fromValue, 10);
    const _to = parseFloat(toValue, 10);
    fromValue = Math.min(_from, _to);
    toValue = Math.max(_from, _to);
  } else if (dataRow.meta.type === 'date') {
    const dateBefore = (0, _conditionRegisterer.getCondition)(_before.CONDITION_NAME, [toValue]);
    const dateAfter = (0, _conditionRegisterer.getCondition)(_after.CONDITION_NAME, [fromValue]);
    return dateBefore(dataRow) && dateAfter(dataRow);
  }
  return dataRow.value >= fromValue && dataRow.value <= toValue;
}
(0, _conditionRegisterer.registerCondition)(CONDITION_NAME, condition, {
  name: C.FILTERS_CONDITIONS_BETWEEN,
  inputsCount: 2,
  showOperators: true
});