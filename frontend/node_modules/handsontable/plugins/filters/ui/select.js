"use strict";

exports.__esModule = true;
var _menu = _interopRequireDefault(require("../../../plugins/contextMenu/menu"));
var _object = require("../../../helpers/object");
var _array = require("../../../helpers/array");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _predefinedItems = require("../../../plugins/contextMenu/predefinedItems");
var _base = _interopRequireDefault(require("./_base"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
const privatePool = new WeakMap();

/**
 * @private
 * @class SelectUI
 */
class SelectUI extends _base.default {
  static get DEFAULTS() {
    return (0, _object.clone)({
      className: 'htUISelect',
      wrapIt: false
    });
  }
  constructor(hotInstance, options) {
    super(hotInstance, (0, _object.extend)(SelectUI.DEFAULTS, options));
    privatePool.set(this, {});
    /**
     * Instance of {@link Menu}.
     *
     * @type {Menu}
     */
    this.menu = null;
    /**
     * List of available select options.
     *
     * @type {Array}
     */
    this.items = [];
    this.registerHooks();
  }

  /**
   * Register all necessary hooks.
   */
  registerHooks() {
    this.addLocalHook('click', () => this.onClick());
  }

  /**
   * Set options which can be selected in the list.
   *
   * @param {Array} items Array of objects with required keys `key` and `name`.
   */
  setItems(items) {
    this.items = this.translateNames(items);
    if (this.menu) {
      this.menu.setMenuItems(this.items);
    }
  }

  /**
   * Translate names of menu items.
   *
   * @param {Array} items Array of objects with required keys `key` and `name`.
   * @returns {Array} Items with translated `name` keys.
   */
  translateNames(items) {
    (0, _array.arrayEach)(items, item => {
      item.name = this.translateIfPossible(item.name);
    });
    return items;
  }

  /**
   * Build DOM structure.
   */
  build() {
    super.build();
    this.menu = new _menu.default(this.hot, {
      className: 'htSelectUI htFiltersConditionsMenu',
      keepInViewport: false,
      standalone: true,
      container: this.options.menuContainer
    });
    this.menu.setMenuItems(this.items);
    const caption = new _base.default(this.hot, {
      className: 'htUISelectCaption'
    });
    const dropdown = new _base.default(this.hot, {
      className: 'htUISelectDropdown'
    });
    const priv = privatePool.get(this);
    priv.caption = caption;
    priv.captionElement = caption.element;
    priv.dropdown = dropdown;
    (0, _array.arrayEach)([caption, dropdown], element => this._element.appendChild(element.element));
    this.menu.addLocalHook('select', command => this.onMenuSelect(command));
    this.menu.addLocalHook('afterClose', () => this.onMenuClosed());
    this.update();
  }

  /**
   * Update DOM structure.
   */
  update() {
    if (!this.isBuilt()) {
      return;
    }
    let conditionName;
    if (this.options.value) {
      conditionName = this.options.value.name;
    } else {
      conditionName = this.menu.hot.getTranslatedPhrase(C.FILTERS_CONDITIONS_NONE);
    }
    privatePool.get(this).captionElement.textContent = conditionName;
    super.update();
  }

  /**
   * Open select dropdown menu with available options.
   */
  openOptions() {
    const rect = this.element.getBoundingClientRect();
    if (this.menu) {
      this.menu.open();
      this.menu.setPosition({
        left: this.hot.isLtr() ? rect.left - 5 : rect.left - 31,
        top: rect.top - 1,
        width: rect.width,
        height: rect.height
      });
    }
  }

  /**
   * Close select dropdown menu.
   */
  closeOptions() {
    if (this.menu) {
      this.menu.close();
    }
  }

  /**
   * On menu selected listener.
   *
   * @private
   * @param {object} command Selected item.
   */
  onMenuSelect(command) {
    if (command.name !== _predefinedItems.SEPARATOR) {
      this.options.value = command;
      this.update();
      this.runLocalHooks('select', this.options.value);
    }
  }

  /**
   * On menu closed listener.
   *
   * @private
   */
  onMenuClosed() {
    this.runLocalHooks('afterClose');
  }

  /**
   * On element click listener.
   *
   * @private
   */
  onClick() {
    this.openOptions();
  }

  /**
   * Destroy instance.
   */
  destroy() {
    if (this.menu) {
      this.menu.destroy();
      this.menu = null;
    }
    const {
      caption,
      dropdown
    } = privatePool.get(this);
    if (caption) {
      caption.destroy();
    }
    if (dropdown) {
      dropdown.destroy();
    }
    super.destroy();
  }
}
var _default = SelectUI;
exports.default = _default;