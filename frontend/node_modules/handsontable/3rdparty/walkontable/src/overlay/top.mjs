import "core-js/modules/es.error.cause.js";
function _defineProperty(obj, key, value) { key = _toPropertyKey(key); if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _toPropertyKey(arg) { var key = _toPrimitive(arg, "string"); return typeof key === "symbol" ? key : String(key); }
function _toPrimitive(input, hint) { if (typeof input !== "object" || input === null) return input; var prim = input[Symbol.toPrimitive]; if (prim !== undefined) { var res = prim.call(input, hint || "default"); if (typeof res !== "object") return res; throw new TypeError("@@toPrimitive must return a primitive value."); } return (hint === "string" ? String : Number)(input); }
import { addClass, getScrollbarWidth, getScrollTop, getWindowScrollLeft, hasClass, outerHeight, removeClass, setOverlayPosition, resetCssTransform } from "../../../../helpers/dom/element.mjs";
import TopOverlayTable from "./../table/top.mjs";
import { Overlay } from "./_base.mjs";
import { CLONE_TOP } from "./constants.mjs";
/**
 * @class TopOverlay
 */
export class TopOverlay extends Overlay {
  /**
   * @param {Walkontable} wotInstance The Walkontable instance. @TODO refactoring: check if can be deleted.
   * @param {FacadeGetter} facadeGetter Function which return proper facade.
   * @param {Settings} wtSettings The Walkontable settings.
   * @param {DomBindings} domBindings Dom elements bound to the current instance.
   */
  constructor(wotInstance, facadeGetter, wtSettings, domBindings) {
    super(wotInstance, facadeGetter, CLONE_TOP, wtSettings, domBindings);
    /**
     * Cached value which holds the previous value of the `fixedRowsTop` option.
     * It is used as a comparison value that can be used to detect changes in this value.
     *
     * @type {number}
     */
    _defineProperty(this, "cachedFixedRowsTop", -1);
    this.cachedFixedRowsTop = this.wtSettings.getSetting('fixedRowsTop');
  }

  /**
   * Factory method to create a subclass of `Table` that is relevant to this overlay.
   *
   * @see Table#constructor
   * @param {...*} args Parameters that will be forwarded to the `Table` constructor.
   * @returns {TopOverlayTable}
   */
  createTable() {
    for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
      args[_key] = arguments[_key];
    }
    return new TopOverlayTable(...args);
  }

  /**
   * Checks if overlay should be fully rendered.
   *
   * @returns {boolean}
   */
  shouldBeRendered() {
    return this.wtSettings.getSetting('shouldRenderTopOverlay');
  }

  /**
   * Updates the top overlay position.
   *
   * @returns {boolean}
   */
  resetFixedPosition() {
    if (!this.needFullRender || !this.wot.wtTable.holder.parentNode) {
      // removed from DOM
      return false;
    }
    const overlayRoot = this.clone.wtTable.holder.parentNode;
    const {
      rootWindow
    } = this.domBindings;
    const preventOverflow = this.wtSettings.getSetting('preventOverflow');
    let overlayPosition = 0;
    let skipInnerBorderAdjusting = false;
    if (this.trimmingContainer === rootWindow && (!preventOverflow || preventOverflow !== 'vertical')) {
      const {
        wtTable
      } = this.wot;
      const hiderRect = wtTable.hider.getBoundingClientRect();
      const bottom = Math.ceil(hiderRect.bottom);
      const rootHeight = overlayRoot.offsetHeight;

      // This checks if the overlay is going to an infinite loop caused by added (or removed)
      // `innerBorderTop` class name. Toggling the class name shifts the viewport by 1px and
      // triggers the `scroll` event. It causes the table to render. The new render cycle takes into,
      // account the shift and toggles the class name again. This causes the next loops. This
      // happens only on Chrome (#7256).
      //
      // When we detect that the table bottom position is the same as the overlay bottom,
      // do not toggle the class name.
      //
      // This workaround will be able to be cleared after merging the SVG borders, which introduces
      // frozen lines (no more `innerBorderTop` workaround).
      skipInnerBorderAdjusting = bottom === rootHeight;
      overlayPosition = this.getOverlayOffset();
      setOverlayPosition(overlayRoot, '0px', `${overlayPosition}px`);
    } else {
      overlayPosition = this.getScrollPosition();
      resetCssTransform(overlayRoot);
    }
    const positionChanged = this.adjustHeaderBordersPosition(overlayPosition, skipInnerBorderAdjusting);
    this.adjustElementsSize();
    return positionChanged;
  }

  /**
   * Sets the main overlay's vertical scroll position.
   *
   * @param {number} pos The scroll position.
   * @returns {boolean}
   */
  setScrollPosition(pos) {
    const rootWindow = this.domBindings.rootWindow;
    let result = false;
    if (this.mainTableScrollableElement === rootWindow && rootWindow.scrollY !== pos) {
      rootWindow.scrollTo(getWindowScrollLeft(rootWindow), pos);
      result = true;
    } else if (this.mainTableScrollableElement.scrollTop !== pos) {
      this.mainTableScrollableElement.scrollTop = pos;
      result = true;
    }
    return result;
  }

  /**
   * Triggers onScroll hook callback.
   */
  onScroll() {
    this.wtSettings.getSetting('onScrollHorizontally');
  }

  /**
   * Calculates total sum cells height.
   *
   * @param {number} from Row index which calculates started from.
   * @param {number} to Row index where calculation is finished.
   * @returns {number} Height sum.
   */
  sumCellSizes(from, to) {
    const defaultRowHeight = this.wtSettings.getSetting('defaultRowHeight');
    let row = from;
    let sum = 0;
    while (row < to) {
      const height = this.wot.wtTable.getRowHeight(row);
      sum += height === void 0 ? defaultRowHeight : height;
      row += 1;
    }
    return sum;
  }

  /**
   * Adjust overlay root element, childs and master table element sizes (width, height).
   *
   * @param {boolean} [force=false] When `true`, it adjusts the DOM nodes sizes for that overlay.
   */
  adjustElementsSize() {
    let force = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : false;
    this.updateTrimmingContainer();
    if (this.needFullRender || force) {
      this.adjustRootElementSize();
      this.adjustRootChildrenSize();
    }
  }

  /**
   * Adjust overlay root element size (width and height).
   */
  adjustRootElementSize() {
    const {
      wtTable
    } = this.wot;
    const {
      rootDocument,
      rootWindow
    } = this.domBindings;
    const scrollbarWidth = getScrollbarWidth(rootDocument);
    const overlayRoot = this.clone.wtTable.holder.parentNode;
    const overlayRootStyle = overlayRoot.style;
    const preventOverflow = this.wtSettings.getSetting('preventOverflow');
    if (this.trimmingContainer !== rootWindow || preventOverflow === 'horizontal') {
      let width = this.wot.wtViewport.getWorkspaceWidth();
      if (this.wot.wtOverlays.hasScrollbarRight) {
        width -= scrollbarWidth;
      }
      width = Math.min(width, wtTable.wtRootElement.scrollWidth);
      overlayRootStyle.width = `${width}px`;
    } else {
      overlayRootStyle.width = '';
    }
    this.clone.wtTable.holder.style.width = overlayRootStyle.width;
    let tableHeight = outerHeight(this.clone.wtTable.TABLE);
    if (!this.wot.wtTable.hasDefinedSize()) {
      tableHeight = 0;
    }
    overlayRootStyle.height = `${tableHeight}px`;
  }

  /**
   * Adjust overlay root childs size.
   */
  adjustRootChildrenSize() {
    var _selections$getCell$g;
    const {
      holder
    } = this.clone.wtTable;
    const {
      selections
    } = this.wot;
    const facade = this.facadeGetter();
    const selectionCornerOffset = Math.abs((_selections$getCell$g = selections === null || selections === void 0 ? void 0 : selections.getCell().getBorder(facade).cornerCenterPointOffset) !== null && _selections$getCell$g !== void 0 ? _selections$getCell$g : 0);
    this.clone.wtTable.hider.style.width = this.hider.style.width;
    holder.style.width = holder.parentNode.style.width;
    // Add selection corner protruding part to the holder total height to make sure that
    // borders' corner won't be cut after vertical scroll (#6937).
    holder.style.height = `${parseInt(holder.parentNode.style.height, 10) + selectionCornerOffset}px`;
  }

  /**
   * Adjust the overlay dimensions and position.
   */
  applyToDOM() {
    const total = this.wtSettings.getSetting('totalRows');
    if (typeof this.wot.wtViewport.rowsRenderCalculator.startPosition === 'number') {
      this.spreader.style.top = `${this.wot.wtViewport.rowsRenderCalculator.startPosition}px`;
    } else if (total === 0) {
      // can happen if there are 0 rows
      this.spreader.style.top = '0';
    } else {
      throw new Error('Incorrect value of the rowsRenderCalculator');
    }
    this.spreader.style.bottom = '';
    if (this.needFullRender) {
      this.syncOverlayOffset();
    }
  }

  /**
   * Synchronize calculated left position to an element.
   */
  syncOverlayOffset() {
    const styleProperty = this.isRtl() ? 'right' : 'left';
    const {
      spreader
    } = this.clone.wtTable;
    if (typeof this.wot.wtViewport.columnsRenderCalculator.startPosition === 'number') {
      spreader.style[styleProperty] = `${this.wot.wtViewport.columnsRenderCalculator.startPosition}px`;
    } else {
      spreader.style[styleProperty] = '';
    }
  }

  /**
   * Scrolls vertically to a row.
   *
   * @param {number} sourceRow Row index which you want to scroll to.
   * @param {boolean} [bottomEdge] If `true`, scrolls according to the bottom edge (top edge is by default).
   * @returns {boolean}
   */
  scrollTo(sourceRow, bottomEdge) {
    const {
      wot,
      wtSettings
    } = this;
    const sourceInstance = wot.cloneSource ? wot.cloneSource : wot;
    const mainHolder = sourceInstance.wtTable.holder;
    let newY = this.getTableParentOffset();
    let scrollbarCompensation = 0;
    if (bottomEdge) {
      const rowHeight = this.wot.wtTable.getRowHeight(sourceRow);
      const viewportHeight = this.wot.wtViewport.getViewportHeight();
      if (rowHeight > viewportHeight) {
        bottomEdge = false;
      }
    }
    if (bottomEdge && mainHolder.offsetHeight !== mainHolder.clientHeight) {
      scrollbarCompensation = getScrollbarWidth(this.domBindings.rootDocument);
    }
    if (bottomEdge) {
      const fixedRowsBottom = wtSettings.getSetting('fixedRowsBottom');
      const totalRows = wtSettings.getSetting('totalRows');
      newY += this.sumCellSizes(0, sourceRow + 1);
      newY -= wot.wtViewport.getViewportHeight() - this.sumCellSizes(totalRows - fixedRowsBottom, totalRows);
      // Fix 1 pixel offset when cell is selected
      newY += 1;
    } else {
      newY += this.sumCellSizes(wtSettings.getSetting('fixedRowsTop'), sourceRow);
    }
    newY += scrollbarCompensation;
    return this.setScrollPosition(newY);
  }

  /**
   * Gets table parent top position.
   *
   * @returns {number}
   */
  getTableParentOffset() {
    if (this.mainTableScrollableElement === this.domBindings.rootWindow) {
      return this.wot.wtTable.holderOffset.top;
    }
    return 0;
  }

  /**
   * Gets the main overlay's vertical scroll position.
   *
   * @returns {number} Main table's vertical scroll position.
   */
  getScrollPosition() {
    return getScrollTop(this.mainTableScrollableElement, this.domBindings.rootWindow);
  }

  /**
   * Gets the main overlay's vertical overlay offset.
   *
   * @returns {number} Main table's vertical overlay offset.
   */
  getOverlayOffset() {
    const {
      rootWindow
    } = this.domBindings;
    const preventOverflow = this.wtSettings.getSetting('preventOverflow');
    let overlayOffset = 0;
    if (this.trimmingContainer === rootWindow && (!preventOverflow || preventOverflow !== 'vertical')) {
      const rootHeight = this.wot.wtTable.getTotalHeight();
      const overlayRootHeight = this.clone.wtTable.getTotalHeight();
      const maxOffset = rootHeight - overlayRootHeight;
      overlayOffset = Math.max(this.getScrollPosition() - this.getTableParentOffset(), 0);
      if (overlayOffset > maxOffset) {
        overlayOffset = 0;
      }
    }
    return overlayOffset;
  }

  /**
   * Adds css classes to hide the header border's header (cell-selection border hiding issue).
   *
   * @param {number} position Header Y position if trimming container is window or scroll top if not.
   * @param {boolean} [skipInnerBorderAdjusting=false] If `true` the inner border adjusting will be skipped.
   * @returns {boolean}
   */
  adjustHeaderBordersPosition(position) {
    let skipInnerBorderAdjusting = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : false;
    const masterParent = this.wot.wtTable.holder.parentNode;
    const totalColumns = this.wtSettings.getSetting('totalColumns');
    if (totalColumns) {
      removeClass(masterParent, 'emptyColumns');
    } else {
      addClass(masterParent, 'emptyColumns');
    }
    let positionChanged = false;
    if (!skipInnerBorderAdjusting) {
      const fixedRowsTop = this.wtSettings.getSetting('fixedRowsTop');
      const areFixedRowsTopChanged = this.cachedFixedRowsTop !== fixedRowsTop;
      const columnHeaders = this.wtSettings.getSetting('columnHeaders');
      if ((areFixedRowsTopChanged || fixedRowsTop === 0) && columnHeaders.length > 0) {
        const previousState = hasClass(masterParent, 'innerBorderTop');
        this.cachedFixedRowsTop = this.wtSettings.getSetting('fixedRowsTop');
        if (position || this.wtSettings.getSetting('totalRows') === 0) {
          addClass(masterParent, 'innerBorderTop');
          positionChanged = !previousState;
        } else {
          removeClass(masterParent, 'innerBorderTop');
          positionChanged = previousState;
        }
      }
    }
    return positionChanged;
  }
}