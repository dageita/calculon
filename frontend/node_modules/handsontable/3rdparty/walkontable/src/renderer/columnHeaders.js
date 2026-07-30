"use strict";

exports.__esModule = true;
var _element = require("./../../../../helpers/dom/element");
var _base = _interopRequireDefault(require("./_base"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
/**
 * Column headers renderer responsible for managing (inserting, tracking, rendering) TR and TH elements.
 *
 *   <thead> (root node)
 *     ├ <tr>   \
 *     ├ <tr>    \
 *     ├ <tr>     - ColumnHeadersRenderer
 *     ├ <tr>    /
 *     └ <tr>   /.
 *
 * @class {ColumnHeadersRenderer}
 */
class ColumnHeadersRenderer extends _base.default {
  constructor(rootNode) {
    super(null, rootNode); // NodePool is not implemented for this renderer yet
  }

  /**
   * Adjusts the number of the rendered elements.
   */
  adjust() {
    const {
      columnHeadersCount,
      rowHeadersCount
    } = this.table;
    let TR = this.rootNode.firstChild;
    if (columnHeadersCount) {
      const {
        columnsToRender
      } = this.table;
      const allColumnsToRender = columnsToRender + rowHeadersCount;
      for (let i = 0, len = columnHeadersCount; i < len; i++) {
        TR = this.rootNode.childNodes[i];
        if (!TR) {
          TR = this.table.rootDocument.createElement('tr');
          this.rootNode.appendChild(TR);
        }
        this.renderedNodes = TR.childNodes.length;
        while (this.renderedNodes < allColumnsToRender) {
          TR.appendChild(this.table.rootDocument.createElement('th'));
          this.renderedNodes += 1;
        }
        while (this.renderedNodes > allColumnsToRender) {
          TR.removeChild(TR.lastChild);
          this.renderedNodes -= 1;
        }
      }
      const theadChildrenLength = this.rootNode.childNodes.length;
      if (theadChildrenLength > columnHeadersCount) {
        for (let i = columnHeadersCount; i < theadChildrenLength; i++) {
          this.rootNode.removeChild(this.rootNode.lastChild);
        }
      }
    } else if (TR) {
      (0, _element.empty)(TR);
    }
  }

  /**
   * Renders the TH elements.
   */
  render() {
    const {
      columnHeadersCount
    } = this.table;
    for (let rowHeaderIndex = 0; rowHeaderIndex < columnHeadersCount; rowHeaderIndex += 1) {
      const {
        columnHeaderFunctions,
        columnsToRender,
        rowHeadersCount
      } = this.table;
      const TR = this.rootNode.childNodes[rowHeaderIndex];
      for (let renderedColumnIndex = -1 * rowHeadersCount; renderedColumnIndex < columnsToRender; renderedColumnIndex += 1) {
        // eslint-disable-line max-len
        const sourceColumnIndex = this.table.renderedColumnToSource(renderedColumnIndex);
        const TH = TR.childNodes[renderedColumnIndex + rowHeadersCount];
        TH.className = '';
        TH.removeAttribute('style');
        columnHeaderFunctions[rowHeaderIndex](sourceColumnIndex, TH, rowHeaderIndex);
      }
    }
  }
}
exports.default = ColumnHeadersRenderer;