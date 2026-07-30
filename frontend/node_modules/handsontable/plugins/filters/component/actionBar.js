"use strict";

exports.__esModule = true;
require("core-js/modules/es.array.push.js");
var _element = require("../../../helpers/dom/element");
var _array = require("../../../helpers/array");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
var _base = _interopRequireDefault(require("./_base"));
var _input = _interopRequireDefault(require("../ui/input"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
/**
 * @private
 * @class ActionBarComponent
 */
class ActionBarComponent extends _base.default {
  static get BUTTON_OK() {
    return 'ok';
  }
  static get BUTTON_CANCEL() {
    return 'cancel';
  }
  constructor(hotInstance, options) {
    super(hotInstance, {
      id: options.id,
      stateless: true
    });
    this.name = options.name;
    this.elements.push(new _input.default(this.hot, {
      type: 'button',
      value: C.FILTERS_BUTTONS_OK,
      className: 'htUIButton htUIButtonOK',
      identifier: ActionBarComponent.BUTTON_OK
    }));
    this.elements.push(new _input.default(this.hot, {
      type: 'button',
      value: C.FILTERS_BUTTONS_CANCEL,
      className: 'htUIButton htUIButtonCancel',
      identifier: ActionBarComponent.BUTTON_CANCEL
    }));
    this.registerHooks();
  }

  /**
   * Register all necessary hooks.
   *
   * @private
   */
  registerHooks() {
    (0, _array.arrayEach)(this.elements, element => {
      element.addLocalHook('click', (event, button) => this.onButtonClick(event, button));
    });
  }

  /**
   * Get menu object descriptor.
   *
   * @returns {object}
   */
  getMenuItemDescriptor() {
    return {
      key: this.id,
      name: this.name,
      isCommand: false,
      disableSelection: true,
      hidden: () => this.isHidden(),
      renderer: (hot, wrapper) => {
        (0, _element.addClass)(wrapper.parentNode, 'htFiltersMenuActionBar');
        if (!wrapper.parentNode.hasAttribute('ghost-table')) {
          (0, _array.arrayEach)(this.elements, ui => wrapper.appendChild(ui.element));
        }
        return wrapper;
      }
    };
  }

  /**
   * Fire accept event.
   */
  accept() {
    this.runLocalHooks('accept');
  }

  /**
   * Fire cancel event.
   */
  cancel() {
    this.runLocalHooks('cancel');
  }

  /**
   * On button click listener.
   *
   * @private
   * @param {Event} event DOM event.
   * @param {InputUI} button InputUI object.
   */
  onButtonClick(event, button) {
    if (button.options.identifier === ActionBarComponent.BUTTON_OK) {
      this.accept();
    } else {
      this.cancel();
    }
  }
}
var _default = ActionBarComponent;
exports.default = _default;