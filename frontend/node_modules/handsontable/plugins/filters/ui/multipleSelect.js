"use strict";

exports.__esModule = true;
require("core-js/modules/es.array.push.js");
var _element = require("../../../helpers/dom/element");
var _object = require("../../../helpers/object");
var _array = require("../../../helpers/array");
var _unicode = require("../../../helpers/unicode");
var _function = require("../../../helpers/function");
var _data = require("../../../helpers/data");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _event = require("../../../helpers/dom/event");
var _base = _interopRequireDefault(require("./_base"));
var _input = _interopRequireDefault(require("./input"));
var _link = _interopRequireDefault(require("./link"));
var _utils = require("../utils");
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const privatePool = new WeakMap();
const SHORTCUTS_GROUP = 'multipleSelect.itemBox';

/**
 * @private
 * @class MultipleSelectUI
 */
class MultipleSelectUI extends _base.default {
  static get DEFAULTS() {
    return (0, _object.clone)({
      className: 'htUIMultipleSelect',
      value: []
    });
  }
  constructor(hotInstance, options) {
    super(hotInstance, (0, _object.extend)(MultipleSelectUI.DEFAULTS, options));
    privatePool.set(this, {});
    /**
     * Input element.
     *
     * @type {InputUI}
     */
    this.searchInput = new _input.default(this.hot, {
      placeholder: C.FILTERS_BUTTONS_PLACEHOLDER_SEARCH,
      className: 'htUIMultipleSelectSearch'
    });
    /**
     * "Select all" UI element.
     *
     * @type {BaseUI}
     */
    this.selectAllUI = new _link.default(this.hot, {
      textContent: C.FILTERS_BUTTONS_SELECT_ALL,
      className: 'htUISelectAll'
    });
    /**
     * "Clear" UI element.
     *
     * @type {BaseUI}
     */
    this.clearAllUI = new _link.default(this.hot, {
      textContent: C.FILTERS_BUTTONS_CLEAR,
      className: 'htUIClearAll'
    });
    /**
     * List of available select options.
     *
     * @type {Array}
     */
    this.items = [];
    /**
     * Handsontable instance used as items list element.
     *
     * @type {Handsontable}
     */
    this.itemsBox = null;
    this.registerHooks();
  }

  /**
   * Register all necessary hooks.
   */
  registerHooks() {
    this.searchInput.addLocalHook('keydown', event => this.onInputKeyDown(event));
    this.searchInput.addLocalHook('input', event => this.onInput(event));
    this.selectAllUI.addLocalHook('click', event => this.onSelectAllClick(event));
    this.clearAllUI.addLocalHook('click', event => this.onClearAllClick(event));
  }

  /**
   * Set available options.
   *
   * @param {Array} items Array of objects with `checked` and `label` property.
   */
  setItems(items) {
    this.items = items;
    if (this.itemsBox) {
      this.itemsBox.loadData(this.items);
    }
  }

  /**
   * Set a locale for the component.
   *
   * @param {string} locale Locale used for filter actions performed on data, ie. `en-US`.
   */
  setLocale(locale) {
    this.locale = locale;
  }

  /**
   * Get a locale for the component.
   *
   * @returns {string}
   */
  getLocale() {
    return this.locale;
  }

  /**
   * Get all available options.
   *
   * @returns {Array}
   */
  getItems() {
    return [...this.items];
  }

  /**
   * Get element value.
   *
   * @returns {Array} Array of selected values.
   */
  getValue() {
    return itemsToValue(this.items);
  }

  /**
   * Check if all values listed in element are selected.
   *
   * @returns {boolean}
   */
  isSelectedAllValues() {
    return this.items.length === this.getValue().length;
  }

  /**
   * Build DOM structure.
   */
  build() {
    super.build();
    const {
      rootDocument
    } = this.hot;
    const itemsBoxWrapper = rootDocument.createElement('div');
    const selectionControl = new _base.default(this.hot, {
      className: 'htUISelectionControls',
      children: [this.selectAllUI, this.clearAllUI]
    });
    this._element.appendChild(this.searchInput.element);
    this._element.appendChild(selectionControl.element);
    this._element.appendChild(itemsBoxWrapper);
    const hotInitializer = wrapper => {
      if (!this._element) {
        return;
      }
      if (this.itemsBox) {
        this.itemsBox.destroy();
      }
      (0, _element.addClass)(wrapper, 'htUIMultipleSelectHot');
      // Constructs and initializes a new Handsontable instance
      this.itemsBox = new this.hot.constructor(wrapper, {
        data: this.items,
        columns: [{
          data: 'checked',
          type: 'checkbox',
          label: {
            property: 'visualValue',
            position: 'after'
          }
        }],
        beforeRenderer: (TD, row, col, prop, value, cellProperties) => {
          TD.title = cellProperties.instance.getDataAtRowProp(row, cellProperties.label.property);
        },
        maxCols: 1,
        autoWrapCol: true,
        height: 110,
        // Workaround for #151.
        colWidths: () => this.itemsBox.container.scrollWidth - (0, _element.getScrollbarWidth)(rootDocument),
        copyPaste: false,
        disableVisualSelection: 'area',
        fillHandle: false,
        fragmentSelection: 'cell',
        tabMoves: {
          row: 1,
          col: 0
        },
        layoutDirection: this.hot.isRtl() ? 'rtl' : 'ltr'
      });
      this.itemsBox.init();
      const shortcutManager = this.itemsBox.getShortcutManager();
      const gridContext = shortcutManager.getContext('grid');
      gridContext.addShortcut({
        // TODO: Is this shortcut really needed? We have one test for that case, but focus is performed programmatically.
        keys: [['Escape']],
        callback: event => {
          this.runLocalHooks('keydown', event, this);
        },
        group: SHORTCUTS_GROUP
      });
    };
    hotInitializer(itemsBoxWrapper);
    setTimeout(() => hotInitializer(itemsBoxWrapper), 100);
  }

  /**
   * Reset DOM structure.
   */
  reset() {
    this.searchInput.reset();
    this.selectAllUI.reset();
    this.clearAllUI.reset();
  }

  /**
   * Update DOM structure.
   */
  update() {
    if (!this.isBuilt()) {
      return;
    }
    this.itemsBox.loadData(valueToItems(this.items, this.options.value));
    super.update();
  }

  /**
   * Destroy instance.
   */
  destroy() {
    if (this.itemsBox) {
      this.itemsBox.destroy();
    }
    this.searchInput.destroy();
    this.clearAllUI.destroy();
    this.selectAllUI.destroy();
    this.searchInput = null;
    this.clearAllUI = null;
    this.selectAllUI = null;
    this.itemsBox = null;
    this.items = null;
    super.destroy();
  }

  /**
   * 'input' event listener for input element.
   *
   * @private
   * @param {Event} event DOM event.
   */
  onInput(event) {
    const value = event.target.value.toLocaleLowerCase(this.getLocale());
    let filteredItems;
    if (value === '') {
      filteredItems = [...this.items];
    } else {
      filteredItems = (0, _array.arrayFilter)(this.items, item => `${item.value}`.toLocaleLowerCase(this.getLocale()).indexOf(value) >= 0);
    }
    this.itemsBox.loadData(filteredItems);
  }

  /**
   * 'keydown' event listener for input element.
   *
   * @private
   * @param {Event} event DOM event.
   */
  onInputKeyDown(event) {
    this.runLocalHooks('keydown', event, this);
    const isKeyCode = (0, _function.partial)(_unicode.isKey, event.keyCode);
    if (isKeyCode('ARROW_DOWN|TAB') && !this.itemsBox.isListening()) {
      (0, _event.stopImmediatePropagation)(event);
      this.itemsBox.listen();
      this.itemsBox.selectCell(0, 0);
    }
  }

  /**
   * On click listener for "Select all" link.
   *
   * @private
   * @param {DOMEvent} event The mouse event object.
   */
  onSelectAllClick(event) {
    const changes = [];
    event.preventDefault();
    (0, _array.arrayEach)(this.itemsBox.getSourceData(), (row, rowIndex) => {
      row.checked = true;
      changes.push((0, _data.dataRowToChangesArray)(row, rowIndex)[0]);
    });
    this.itemsBox.setSourceDataAtCell(changes);
  }

  /**
   * On click listener for "Clear" link.
   *
   * @private
   * @param {DOMEvent} event The mouse event object.
   */
  onClearAllClick(event) {
    const changes = [];
    event.preventDefault();
    (0, _array.arrayEach)(this.itemsBox.getSourceData(), (row, rowIndex) => {
      row.checked = false;
      changes.push((0, _data.dataRowToChangesArray)(row, rowIndex)[0]);
    });
    this.itemsBox.setSourceDataAtCell(changes);
  }
}
var _default = MultipleSelectUI;
/**
 * Pick up object items based on selected values.
 *
 * @param {Array} availableItems Base collection to compare values.
 * @param {Array} selectedValue Flat array with selected values.
 * @returns {Array}
 */
exports.default = _default;
function valueToItems(availableItems, selectedValue) {
  const arrayAssertion = (0, _utils.createArrayAssertion)(selectedValue);
  return (0, _array.arrayMap)(availableItems, item => {
    item.checked = arrayAssertion(item.value);
    return item;
  });
}

/**
 * Convert all checked items into flat array.
 *
 * @param {Array} availableItems Base collection.
 * @returns {Array}
 */
function itemsToValue(availableItems) {
  const items = [];
  (0, _array.arrayEach)(availableItems, item => {
    if (item.checked) {
      items.push(item.value);
    }
  });
  return items;
}