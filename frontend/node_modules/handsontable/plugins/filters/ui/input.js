"use strict";

exports.__esModule = true;
var _element = require("../../../helpers/dom/element");
var _object = require("../../../helpers/object");
var _base = _interopRequireDefault(require("./_base"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
const privatePool = new WeakMap();

/**
 * @private
 * @class InputUI
 */
class InputUI extends _base.default {
  static get DEFAULTS() {
    return (0, _object.clone)({
      placeholder: '',
      type: 'text',
      tagName: 'input'
    });
  }
  constructor(hotInstance, options) {
    super(hotInstance, (0, _object.extend)(InputUI.DEFAULTS, options));
    privatePool.set(this, {});
    this.registerHooks();
  }

  /**
   * Register all necessary hooks.
   */
  registerHooks() {
    this.addLocalHook('click', () => this.onClick());
    this.addLocalHook('keyup', event => this.onKeyup(event));
  }

  /**
   * Build DOM structure.
   */
  build() {
    super.build();
    const priv = privatePool.get(this);
    const icon = this.hot.rootDocument.createElement('div');
    priv.input = this._element.firstChild;
    (0, _element.addClass)(this._element, 'htUIInput');
    (0, _element.addClass)(icon, 'htUIInputIcon');
    this._element.appendChild(icon);
    this.update();
  }

  /**
   * Update element.
   */
  update() {
    if (!this.isBuilt()) {
      return;
    }
    const input = privatePool.get(this).input;
    input.type = this.options.type;
    input.placeholder = this.translateIfPossible(this.options.placeholder);
    input.value = this.translateIfPossible(this.options.value);
  }

  /**
   * Focus element.
   */
  focus() {
    if (this.isBuilt()) {
      privatePool.get(this).input.focus();
    }
  }

  /**
   * OnClick listener.
   */
  onClick() {}

  /**
   * OnKeyup listener.
   *
   * @param {Event} event The mouse event object.
   */
  onKeyup(event) {
    this.options.value = event.target.value;
  }
}
var _default = InputUI;
exports.default = _default;