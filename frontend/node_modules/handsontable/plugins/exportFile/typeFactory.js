"use strict";

exports.__esModule = true;
exports.default = typeFactory;
var _csv = _interopRequireDefault(require("./types/csv"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
const TYPE_CSV = 'csv';
exports.TYPE_CSV = TYPE_CSV;
const TYPE_EXCEL = 'excel'; // TODO
exports.TYPE_EXCEL = TYPE_EXCEL;
const TYPE_PDF = 'pdf'; // TODO
exports.TYPE_PDF = TYPE_PDF;
const EXPORT_TYPES = {
  [TYPE_CSV]: _csv.default
};

/**
 * @private
 * @param {string} type The exporter type.
 * @param {DataProvider} dataProvider The data provider.
 * @param {object} options Constructor options for exporter class.
 * @returns {BaseType|null}
 */
exports.EXPORT_TYPES = EXPORT_TYPES;
function typeFactory(type, dataProvider, options) {
  if (typeof EXPORT_TYPES[type] === 'function') {
    return new EXPORT_TYPES[type](dataProvider, options);
  }
  return null;
}