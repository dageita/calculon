"use strict";

exports.__esModule = true;
var _object = require("../../../helpers/object");
var _localHooks = _interopRequireDefault(require("../../../mixins/localHooks"));
var _eventManager = _interopRequireDefault(require("../../../eventManager"));
var _element = require("../../../helpers/dom/element");
var _array = require("../../../helpers/array");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
const STATE_BUILT = 'built';
const STATE_BUILDING = 'building';
const EVENTS_TO_REGISTER = ['click', 'input', 'keydown', 'keypress', 'keyup', 'focus', 'blur', 'change'];

/**
 * @private
 */
class BaseUI {
  static get DEFAULTS() {
    return (0, _object.clone)({
      className: '',
      value: '',
      tagName: 'div',
      children: [],
      wrapIt: true
    });
  }
  constructor(hotInstance, options) {
    /**
     * Instance of Handsontable.
     *
     * @type {Core}
     */
    this.hot = hotInstance;
    /**
     * Instance of EventManager.
     *
     * @type {EventManager}
     */
    this.eventManager = new _eventManager.default(this);
    /**
     * List of element options.
     *
     * @type {object}
     */
    this.options = (0, _object.extend)(BaseUI.DEFAULTS, options);
    /**
     * Build root DOM element.
     *
     * @type {Element}
     * @private
     */
    this._element = this.hot.rootDocument.createElement(this.options.wrapIt ? 'div' : this.options.tagName);
    /**
     * Flag which determines build state of element.
     *
     * @type {string}
     */
    this.buildState = null;
  }

  /**
   * Set the element value.
   *
   * @param {*} value Set the component value.
   */
  setValue(value) {
    this.options.value = value;
    this.update();
  }

  /**
   * Get the element value.
   *
   * @returns {*}
   */
  getValue() {
    return this.options.value;
  }

  /**
   * Get element as a DOM object.
   *
   * @returns {Element}
   */
  get element() {
    if (this.buildState === STATE_BUILDING) {
      return this._element;
    }
    if (this.buildState === STATE_BUILT) {
      this.update();
      return this._element;
    }
    this.buildState = STATE_BUILDING;
    this.build();
    this.buildState = STATE_BUILT;
    return this._element;
  }

  /**
   * Check if element was built (built whole DOM structure).
   *
   * @returns {boolean}
   */
  isBuilt() {
    return this.buildState === STATE_BUILT;
  }

  /**
   * Translate value if it is possible. It's checked if value belongs to namespace of translated phrases.
   *
   * @param {*} value Value which will may be translated.
   * @returns {*} Translated value if translation was possible, original value otherwise.
   */
  translateIfPossible(value) {
    if (typeof value === 'string' && value.startsWith(C.FILTERS_NAMESPACE)) {
      return this.hot.getTranslatedPhrase(value);
    }
    return value;
  }

  /**
   * Build DOM structure.
   */
  build() {
    const registerEvent = (element, eventName) => {
      this.eventManager.addEventListener(element, eventName, event => this.runLocalHooks(eventName, event, this));
    };
    if (!this.buildState) {
      this.buildState = STATE_BUILDING;
    }
    if (this.options.className) {
      (0, _element.addClass)(this._element, this.options.className);
    }
    if (this.options.children.length) {
      (0, _array.arrayEach)(this.options.children, element => this._element.appendChild(element.element));
    } else if (this.options.wrapIt) {
      const element = this.hot.rootDocument.createElement(this.options.tagName);
      (0, _object.objectEach)(this.options, (value, key) => {
        if (element[key] !== void 0 && key !== 'className' && key !== 'tagName' && key !== 'children') {
          element[key] = this.translateIfPossible(value);
        }
      });
      this._element.appendChild(element);
      (0, _array.arrayEach)(EVENTS_TO_REGISTER, eventName => registerEvent(element, eventName));
    } else {
      (0, _array.arrayEach)(EVENTS_TO_REGISTER, eventName => registerEvent(this._element, eventName));
    }
  }

  /**
   * Update DOM structure.
   */
  update() {}

  /**
   * Reset to initial state.
   */
  reset() {
    this.options.value = '';
    this.update();
  }

  /**
   * Show element.
   */
  show() {
    this.element.style.display = '';
  }

  /**
   * Hide element.
   */
  hide() {
    this.element.style.display = 'none';
  }

  /**
   * Focus element.
   */
  focus() {}
  destroy() {
    this.eventManager.destroy();
    this.eventManager = null;
    this.hot = null;
    if (this._element.parentNode) {
      this._element.parentNode.removeChild(this._element);
    }
    this._element = null;
  }
}
(0, _object.mixin)(BaseUI, _localHooks.default);
var _default = BaseUI;
exports.default = _default;