"use strict";

exports.__esModule = true;
var _object = require("../../../helpers/object");
var _string = require("../../../helpers/string");
/**
 * @private
 */
class BaseType {
  /**
   * Default options.
   *
   * @returns {object}
   */
  static get DEFAULT_OPTIONS() {
    return {
      mimeType: 'text/plain',
      fileExtension: 'txt',
      filename: 'Handsontable [YYYY]-[MM]-[DD]',
      encoding: 'utf-8',
      bom: false,
      columnHeaders: false,
      rowHeaders: false,
      exportHiddenColumns: false,
      exportHiddenRows: false,
      range: []
    };
  }
  constructor(dataProvider, options) {
    /**
     * Data provider.
     *
     * @type {DataProvider}
     */
    this.dataProvider = dataProvider;
    /**
     * Format type class options.
     *
     * @type {object}
     */
    this.options = this._mergeOptions(options);
    this.dataProvider.setOptions(this.options);
  }

  /**
   * Merge options provided by users with defaults.
   *
   * @param {object} options An object with options to merge with.
   * @returns {object} Returns new options object.
   */
  _mergeOptions(options) {
    let _options = (0, _object.clone)(this.constructor.DEFAULT_OPTIONS);
    const date = new Date();
    _options = (0, _object.extend)((0, _object.clone)(BaseType.DEFAULT_OPTIONS), _options);
    _options = (0, _object.extend)(_options, options);
    _options.filename = (0, _string.substitute)(_options.filename, {
      YYYY: date.getFullYear(),
      MM: `${date.getMonth() + 1}`.padStart(2, '0'),
      DD: `${date.getDate()}`.padStart(2, '0')
    });
    return _options;
  }
}
var _default = BaseType;
exports.default = _default;