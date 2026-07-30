"use strict";

exports.__esModule = true;
require("core-js/modules/es.error.cause.js");
var _element = require("../../../helpers/dom/element");
function _defineProperty(obj, key, value) { key = _toPropertyKey(key); if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _toPropertyKey(arg) { var key = _toPrimitive(arg, "string"); return typeof key === "symbol" ? key : String(key); }
function _toPrimitive(input, hint) { if (typeof input !== "object" || input === null) return input; var prim = input[Symbol.toPrimitive]; if (prim !== undefined) { var res = prim.call(input, hint || "default"); if (typeof res !== "object") return res; throw new TypeError("@@toPrimitive must return a primitive value."); } return (hint === "string" ? String : Number)(input); }
/**
 * @class Scroll
 */
class Scroll {
  /**
   * @param {ScrollDao} dataAccessObject Tha data access object.
   */
  constructor(dataAccessObject) {
    /**
     * The data access object.
     *
     * @protected
     * @type {ScrollDao}
     */
    _defineProperty(this, "dataAccessObject", void 0);
    /**
     * Holds the last column reached by the scroll, which determines the scroll snapping direction
     * (left or right) for a next horizontal scroll.
     *
     * @protected
     * @type {number}
     */
    _defineProperty(this, "lastScrolledColumnPos", -1);
    /**
     * Holds the last row reached by the scroll, which determines the scroll snapping direction
     * (top or bottom) for a next vertical scroll.
     *
     * @protected
     * @type {number}
     */
    _defineProperty(this, "lastScrolledRowPos", -1);
    this.dataAccessObject = dataAccessObject;
  }

  /**
   * Scrolls viewport to a cell.
   *
   * @param {CellCoords} coords The cell coordinates.
   * @param {boolean} [snapToTop] If `true`, viewport is scrolled to show the cell on the top of the table.
   * @param {boolean} [snapToRight] If `true`, viewport is scrolled to show the cell on the right of the table.
   * @param {boolean} [snapToBottom] If `true`, viewport is scrolled to show the cell on the bottom of the table.
   * @param {boolean} [snapToLeft] If `true`, viewport is scrolled to show the cell on the left of the table.
   * @returns {boolean}
   */
  scrollViewport(coords, snapToTop, snapToRight, snapToBottom, snapToLeft) {
    const scrolledHorizontally = this.scrollViewportHorizontally(coords.col, snapToRight, snapToLeft);
    const scrolledVertically = this.scrollViewportVertically(coords.row, snapToTop, snapToBottom);
    return scrolledHorizontally || scrolledVertically;
  }

  /**
   * Scrolls viewport to a column.
   *
   * @param {number} column Visual column index.
   * @param {boolean} [snapToRight] If `true`, viewport is scrolled to show the cell on the right of the table.
   * @param {boolean} [snapToLeft] If `true`, viewport is scrolled to show the cell on the left of the table.
   * @returns {boolean}
   */
  scrollViewportHorizontally(column, snapToRight, snapToLeft) {
    const {
      drawn,
      totalColumns
    } = this.dataAccessObject;

    // do not scroll the viewport when the column points to a range outside of the dataset
    if (!drawn || !Number.isInteger(column) || column < 0 || column > totalColumns) {
      return false;
    }
    const firstVisibleColumn = this.getFirstVisibleColumn();
    const lastVisibleColumn = this.getLastVisibleColumn();
    const autoSnapping = snapToRight === void 0 && snapToLeft === void 0;
    const {
      fixedColumnsStart,
      inlineStartOverlay
    } = this.dataAccessObject;

    // for auto-snapping (both snap* arguments are undefined) do not scroll the viewport
    // when the columns points to the overlays
    if (autoSnapping && column < fixedColumnsStart) {
      return false;
    }
    let result = false;

    // if there is no fully visible columns use the supporting variable (lastScrolledColumnPos) to
    // determine the snapping direction (left or right)
    if (firstVisibleColumn === -1) {
      result = inlineStartOverlay.scrollTo(column, autoSnapping ? column > this.lastScrolledColumnPos : snapToRight);
    } else if (autoSnapping && (column < firstVisibleColumn || column > lastVisibleColumn) || !autoSnapping) {
      // if there is at least one fully visible column determine the snapping direction based on
      // that columns or by snapToRight/snapToLeft flags, if provided.
      result = inlineStartOverlay.scrollTo(column, autoSnapping ? column > lastVisibleColumn : snapToRight);
    }
    if (result) {
      this.lastScrolledColumnPos = column;
    }
    return result;
  }

  /**
   * Scrolls viewport to a row.
   *
   * @param {number} row Visual row index.
   * @param {boolean} [snapToTop] If `true`, viewport is scrolled to show the cell on the top of the table.
   * @param {boolean} [snapToBottom] If `true`, viewport is scrolled to show the cell on the bottom of the table.
   * @returns {boolean}
   */
  scrollViewportVertically(row, snapToTop, snapToBottom) {
    const {
      drawn,
      totalRows
    } = this.dataAccessObject;

    // do not scroll the viewport when the row points to a range outside of the dataset
    if (!drawn || !Number.isInteger(row) || row < 0 || row > totalRows) {
      return false;
    }
    const firstVisibleRow = this.getFirstVisibleRow();
    const lastVisibleRow = this.getLastVisibleRow();
    const autoSnapping = snapToTop === void 0 && snapToBottom === void 0;
    const {
      fixedRowsBottom,
      fixedRowsTop,
      topOverlay
    } = this.dataAccessObject;

    // for auto-snapping (both snap* arguments are undefined) do not scroll the viewport
    // when the rows points to the overlays
    if (autoSnapping && (row < fixedRowsTop || row > totalRows - fixedRowsBottom - 1)) {
      return false;
    }
    let result = false;

    // if there is no fully visible rows use the supporting variable (lastScrolledRowPos) to
    // determine the snapping direction (top or bottom)
    if (firstVisibleRow === -1) {
      result = topOverlay.scrollTo(row, autoSnapping ? row > this.lastScrolledRowPos : snapToBottom);
    } else if (autoSnapping && (row < firstVisibleRow || row > lastVisibleRow) || !autoSnapping) {
      // if there is at least one fully visible row determine the snapping direction based on
      // that rows or by snapToTop/snapToBottom flags, if provided.
      result = topOverlay.scrollTo(row, autoSnapping ? row > lastVisibleRow : snapToBottom);
    }
    if (result) {
      this.lastScrolledRowPos = row;
    }
    return result;
  }

  /**
   * Get first visible row based on virtual dom and how table is visible in browser window viewport.
   *
   * @returns {number}
   */
  getFirstVisibleRow() {
    const {
      topOverlay,
      wtTable,
      wtViewport,
      totalRows,
      fixedRowsTop,
      rootWindow
    } = this.dataAccessObject;
    let firstVisibleRow = wtTable.getFirstVisibleRow();
    if (topOverlay.mainTableScrollableElement === rootWindow) {
      const rootElementOffset = (0, _element.offset)(wtTable.wtRootElement);
      const totalTableHeight = (0, _element.innerHeight)(wtTable.hider);
      const windowHeight = (0, _element.innerHeight)(rootWindow);
      const windowScrollTop = (0, _element.getScrollTop)(rootWindow, rootWindow);

      // Only calculate firstVisibleRow when table didn't filled (from up) whole viewport space
      if (rootElementOffset.top + totalTableHeight - windowHeight <= windowScrollTop) {
        let rowsHeight = wtViewport.getColumnHeaderHeight();
        rowsHeight += topOverlay.sumCellSizes(0, fixedRowsTop);
        for (let row = totalRows; row > 0; row--) {
          rowsHeight += topOverlay.sumCellSizes(row - 1, row);
          if (rootElementOffset.top + totalTableHeight - rowsHeight <= windowScrollTop) {
            // Return physical row + 1
            firstVisibleRow = row;
            break;
          }
        }
      }
    }
    return firstVisibleRow;
  }

  /**
   * Get last visible row based on virtual dom and how table is visible in browser window viewport.
   *
   * @returns {number}
   */
  getLastVisibleRow() {
    const {
      topOverlay,
      wtTable,
      wtViewport,
      totalRows,
      rootWindow
    } = this.dataAccessObject;
    let lastVisibleRow = wtTable.getLastVisibleRow();
    if (topOverlay.mainTableScrollableElement === rootWindow) {
      const rootElementOffset = (0, _element.offset)(wtTable.wtRootElement);
      const windowScrollTop = (0, _element.getScrollTop)(rootWindow, rootWindow);

      // Only calculate lastVisibleRow when table didn't filled (from bottom) whole viewport space
      if (rootElementOffset.top > windowScrollTop) {
        const windowHeight = (0, _element.innerHeight)(rootWindow);
        let rowsHeight = wtViewport.getColumnHeaderHeight();
        for (let row = 1; row <= totalRows; row++) {
          rowsHeight += topOverlay.sumCellSizes(row - 1, row);
          if (rootElementOffset.top + rowsHeight - windowScrollTop >= windowHeight) {
            // Return physical row - 1 (-2 because rangeEach gives row index + 1 - sumCellSizes requirements)
            lastVisibleRow = row - 2;
            break;
          }
        }
      }
    }
    return lastVisibleRow;
  }

  /**
   * Get first visible column based on virtual dom and how table is visible in browser window viewport.
   *
   * @returns {number}
   */
  getFirstVisibleColumn() {
    const {
      inlineStartOverlay,
      wtTable,
      wtViewport,
      totalColumns,
      rootWindow
    } = this.dataAccessObject;
    let firstVisibleColumn = wtTable.getFirstVisibleColumn();
    if (inlineStartOverlay.mainTableScrollableElement === rootWindow) {
      const rootElementOffset = (0, _element.offset)(wtTable.wtRootElement);
      const totalTableWidth = (0, _element.innerWidth)(wtTable.hider);
      const windowWidth = (0, _element.innerWidth)(rootWindow);
      const windowScrollLeft = Math.abs((0, _element.getScrollLeft)(rootWindow, rootWindow));

      // Only calculate firstVisibleColumn when table didn't filled (from left) whole viewport space
      if (rootElementOffset.left + totalTableWidth - windowWidth <= windowScrollLeft) {
        let columnsWidth = wtViewport.getRowHeaderWidth();
        for (let column = totalColumns; column > 0; column--) {
          columnsWidth += inlineStartOverlay.sumCellSizes(column - 1, column);
          if (rootElementOffset.left + totalTableWidth - columnsWidth <= windowScrollLeft) {
            // Return physical column + 1
            firstVisibleColumn = column;
            break;
          }
        }
      }
    }
    return firstVisibleColumn;
  }

  /**
   * Get last visible column based on virtual dom and how table is visible in browser window viewport.
   *
   * @returns {number}
   */
  getLastVisibleColumn() {
    const {
      wtSettings,
      inlineStartOverlay,
      wtTable,
      wtViewport,
      totalColumns,
      rootWindow
    } = this.dataAccessObject;
    let lastVisibleColumn = wtTable.getLastVisibleColumn();
    if (inlineStartOverlay.mainTableScrollableElement === rootWindow) {
      const isRtl = wtSettings.getSetting('rtlMode');
      let inlineStartRootElementOffset = null;
      if (isRtl) {
        const tableRect = wtTable.TABLE.getBoundingClientRect();
        const rootDocument = this.dataAccessObject.rootWindow.document;
        const docOffsetWidth = rootDocument.documentElement.offsetWidth;
        inlineStartRootElementOffset = Math.abs(tableRect.right - docOffsetWidth);
      } else {
        const rootElementOffset = (0, _element.offset)(wtTable.wtRootElement);
        inlineStartRootElementOffset = rootElementOffset.left;
      }
      const windowScrollLeft = Math.abs((0, _element.getScrollLeft)(rootWindow, rootWindow));

      // Only calculate lastVisibleColumn when table didn't filled (from right) whole viewport space
      if (inlineStartRootElementOffset > windowScrollLeft) {
        const windowWidth = (0, _element.innerWidth)(rootWindow);
        let columnsWidth = wtViewport.getRowHeaderWidth();
        for (let column = 1; column <= totalColumns; column++) {
          columnsWidth += inlineStartOverlay.sumCellSizes(column - 1, column);
          if (inlineStartRootElementOffset + columnsWidth - windowScrollLeft >= windowWidth) {
            // Return physical column - 1 (-2 because rangeEach gives column index + 1 - sumCellSizes requirements)
            lastVisibleColumn = column - 2;
            break;
          }
        }
      }
    }
    return lastVisibleColumn;
  }
}
var _default = Scroll;
exports.default = _default;