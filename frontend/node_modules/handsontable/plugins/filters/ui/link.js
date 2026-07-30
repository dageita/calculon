"use strict";

exports.__esModule = true;
var _object = require("../../../helpers/object");
var _base = _interopRequireDefault(require("./_base"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
const privatePool = new WeakMap();

/**
 * @private
 * @class LinkUI
 */
class LinkUI extends _base.default {
  static get DEFAULTS() {
    return (0, _object.clone)({
      href: '#',
      tagName: 'a'
    });
  }
  constructor(hotInstance, options) {
    super(hotInstance, (0, _object.extend)(LinkUI.DEFAULTS, options));
    privatePool.set(this, {});
  }

  /**
   * Build DOM structure.
   */
  build() {
    super.build();
    const priv = privatePool.get(this);
    priv.link = this._element.firstChild;
  }

  /**
   * Update element.
   */
  update() {
    if (!this.isBuilt()) {
      return;
    }
    privatePool.get(this).link.textContent = this.translateIfPossible(this.options.textContent);
  }
}
var _default = LinkUI;
exports.default = _default;