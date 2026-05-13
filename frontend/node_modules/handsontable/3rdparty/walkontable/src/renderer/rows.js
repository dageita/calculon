"use strict";

exports.__esModule = true;
var _console = require("./../../../../helpers/console");
var _templateLiteralTag = require("./../../../../helpers/templateLiteralTag");
var _orderView = require("./../utils/orderView");
var _base = _interopRequireDefault(require("./_base"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
let performanceWarningAppeared = false;

/**
 * Rows renderer responsible for managing (inserting, tracking, rendering) TR elements belongs to TBODY.
 *
 *   <tbody> (root node)
 *     ├ <tr>   \
 *     ├ <tr>    \
 *     ├ <tr>     - RowsRenderer
 *     ├ <tr>    /
 *     └ <tr>   /.
 *
 * @class {RowsRenderer}
 */
class RowsRenderer extends _base.default {
  constructor(rootNode) {
    super('TR', rootNode);
    /**
     * Cache for OrderView classes connected to specified node.
     *
     * @type {WeakMap}
     */
    this.orderView = new _orderView.OrderView(rootNode, sourceRowIndex => this.nodesPool.obtain(sourceRowIndex), this.nodeType);
  }

  /**
   * Returns currently rendered node.
   *
   * @param {string} visualIndex Visual index of the rendered node (it always goeas from 0 to N).
   * @returns {HTMLTableRowElement}
   */
  getRenderedNode(visualIndex) {
    return this.orderView.getNode(visualIndex);
  }

  /**
   * Renders the cells.
   */
  render() {
    const {
      rowsToRender
    } = this.table;
    if (!performanceWarningAppeared && rowsToRender > 1000) {
      performanceWarningAppeared = true;
      (0, _console.warn)((0, _templateLiteralTag.toSingleLine)`Performance tip: Handsontable rendered more than 1000 visible rows. Consider limiting\x20
        the number of rendered rows by specifying the table height and/or turning off the "renderAllRows" option.`);
    }
    this.orderView.setSize(rowsToRender).setOffset(this.table.renderedRowToSource(0)).start();
    for (let visibleRowIndex = 0; visibleRowIndex < rowsToRender; visibleRowIndex++) {
      this.orderView.render();
    }
    this.orderView.end();
  }
}
exports.default = RowsRenderer;