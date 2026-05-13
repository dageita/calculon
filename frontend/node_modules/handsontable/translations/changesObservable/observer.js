"use strict";

require("core-js/modules/es.error.cause.js");
exports.__esModule = true;
var _object = require("../../helpers/object");
var _localHooks = _interopRequireDefault(require("../../mixins/localHooks"));
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
function _classPrivateFieldInitSpec(obj, privateMap, value) { _checkPrivateRedeclaration(obj, privateMap); privateMap.set(obj, value); }
function _checkPrivateRedeclaration(obj, privateCollection) { if (privateCollection.has(obj)) { throw new TypeError("Cannot initialize the same private elements twice on an object"); } }
function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "set"); _classApplyDescriptorSet(receiver, descriptor, value); return value; }
function _classApplyDescriptorSet(receiver, descriptor, value) { if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } }
function _classPrivateFieldGet(receiver, privateMap) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "get"); return _classApplyDescriptorGet(receiver, descriptor); }
function _classExtractFieldDescriptor(receiver, privateMap, action) { if (!privateMap.has(receiver)) { throw new TypeError("attempted to " + action + " private field on non-instance"); } return privateMap.get(receiver); }
function _classApplyDescriptorGet(receiver, descriptor) { if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }
/**
 * The ChangesObserver module is an object that represents a disposable resource
 * provided by the ChangesObservable module.
 *
 * @class ChangesObserver
 */
var _currentInitialChanges = /*#__PURE__*/new WeakMap();
class ChangesObserver {
  constructor() {
    /**
     * The field holds initial changes that will be used to notify the callbacks added using
     * subscribe method. Regardless of the moment of listening for changes, the subscriber
     * will be notified once with all changes made before subscribing.
     *
     * @type {Array}
     */
    _classPrivateFieldInitSpec(this, _currentInitialChanges, {
      writable: true,
      value: []
    });
  }
  /**
   * Subscribes to the observer.
   *
   * @param {Function} callback A function that will be called when the new changes will appear.
   * @returns {ChangesObserver}
   */
  subscribe(callback) {
    this.addLocalHook('change', callback);
    this._write(_classPrivateFieldGet(this, _currentInitialChanges));
    return this;
  }

  /**
   * Unsubscribes all subscriptions. After the method call, the observer would not produce
   * any new events.
   *
   * @returns {ChangesObserver}
   */
  unsubscribe() {
    this.runLocalHooks('unsubscribe');
    this.clearLocalHooks();
    return this;
  }

  /**
   * The write method is executed by the ChangesObservable module. The module produces all
   * changes events that are distributed further by the observer.
   *
   * @private
   * @param {object} changes The chunk of changes produced by the ChangesObservable module.
   * @returns {ChangesObserver}
   */
  _write(changes) {
    if (changes.length > 0) {
      this.runLocalHooks('change', changes);
    }
    return this;
  }

  /**
   * The write method is executed by the ChangesObservable module. The module produces initial
   * changes that will be used to notify new subscribers.
   *
   * @private
   * @param {object} initialChanges The chunk of changes produced by the ChangesObservable module.
   */
  _writeInitialChanges(initialChanges) {
    _classPrivateFieldSet(this, _currentInitialChanges, initialChanges);
  }
}
exports.ChangesObserver = ChangesObserver;
(0, _object.mixin)(ChangesObserver, _localHooks.default);