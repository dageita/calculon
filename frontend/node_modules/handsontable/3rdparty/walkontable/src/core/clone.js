"use strict";

exports.__esModule = true;
require("core-js/modules/es.error.cause.js");
var _event = _interopRequireDefault(require("../event"));
var _base = _interopRequireDefault(require("./_base"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _defineProperty(obj, key, value) { key = _toPropertyKey(key); if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _toPropertyKey(arg) { var key = _toPrimitive(arg, "string"); return typeof key === "symbol" ? key : String(key); }
function _toPrimitive(input, hint) { if (typeof input !== "object" || input === null) return input; var prim = input[Symbol.toPrimitive]; if (prim !== undefined) { var res = prim.call(input, hint || "default"); if (typeof res !== "object") return res; throw new TypeError("@@toPrimitive must return a primitive value."); } return (hint === "string" ? String : Number)(input); }
/**
 * @class Walkontable
 */
class Clone extends _base.default {
  /**
   * @param {HTMLTableElement} table Main table.
   * @param {SettingsPure|Settings} settings The Walkontable settings.
   * @param {WalkontableCloneOptions} clone Clone data.
   */
  constructor(table, settings, clone) {
    super(table, settings);
    _defineProperty(this, "cloneSource", void 0);
    _defineProperty(this, "cloneOverlay", void 0);
    const facadeGetter = this.wtSettings.getSetting('facade', this);
    this.cloneSource = clone.source;
    this.cloneOverlay = clone.overlay;
    this.wtTable = this.cloneOverlay.createTable(this.getTableDao(), facadeGetter, this.domBindings, this.wtSettings);
    this.wtViewport = clone.viewport;
    this.selections = clone.selections;
    this.wtEvent = new _event.default(facadeGetter, this.domBindings, this.wtSettings, this.eventManager, this.wtTable, this.selections, clone.event);
    this.findOriginalHeaders();
  }
}
exports.default = Clone;