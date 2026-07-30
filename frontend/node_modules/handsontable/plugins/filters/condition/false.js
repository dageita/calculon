"use strict";

exports.__esModule = true;
exports.condition = condition;
var _conditionRegisterer = require("../conditionRegisterer");
const CONDITION_NAME = 'false';

/**
 * @returns {boolean}
 */
exports.CONDITION_NAME = CONDITION_NAME;
function condition() {
  return false;
}
(0, _conditionRegisterer.registerCondition)(CONDITION_NAME, condition, {
  name: 'False'
});