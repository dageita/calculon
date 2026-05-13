import "core-js/modules/es.error.cause.js";
function _classPrivateFieldInitSpec(obj, privateMap, value) { _checkPrivateRedeclaration(obj, privateMap); privateMap.set(obj, value); }
function _checkPrivateRedeclaration(obj, privateCollection) { if (privateCollection.has(obj)) { throw new TypeError("Cannot initialize the same private elements twice on an object"); } }
function _classPrivateFieldGet(receiver, privateMap) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "get"); return _classApplyDescriptorGet(receiver, descriptor); }
function _classApplyDescriptorGet(receiver, descriptor) { if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }
function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "set"); _classApplyDescriptorSet(receiver, descriptor, value); return value; }
function _classExtractFieldDescriptor(receiver, privateMap, action) { if (!privateMap.has(receiver)) { throw new TypeError("attempted to " + action + " private field on non-instance"); } return privateMap.get(receiver); }
function _classApplyDescriptorSet(receiver, descriptor, value) { if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } }
import { ChangesObserver } from "./observer.mjs";
import { arrayDiff } from "./utils.mjs";
/**
 * The ChangesObservable module is an object that represents a resource that provides
 * the ability to observe the changes that happened in the index map indexes during
 * the code running.
 *
 * @private
 * @class ChangesObservable
 */
var _observers = /*#__PURE__*/new WeakMap();
var _indexMatrix = /*#__PURE__*/new WeakMap();
var _currentIndexState = /*#__PURE__*/new WeakMap();
var _isMatrixIndexesInitialized = /*#__PURE__*/new WeakMap();
var _initialIndexValue = /*#__PURE__*/new WeakMap();
export class ChangesObservable {
  constructor() {
    let {
      initialIndexValue
    } = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
    /**
     * The list of registered ChangesObserver instances.
     *
     * @type {ChangesObserver[]}
     */
    _classPrivateFieldInitSpec(this, _observers, {
      writable: true,
      value: new Set()
    });
    /**
     * An array with default values that act as a base array that will be compared with
     * the last saved index state. The changes are generated and immediately send through
     * the newly created ChangesObserver object. Thanks to that, the observer initially has
     * all information about what indexes are currently changed.
     *
     * @type {Array}
     */
    _classPrivateFieldInitSpec(this, _indexMatrix, {
      writable: true,
      value: []
    });
    /**
     * An array that holds the indexes state that is currently valid. The value is changed on every
     * index mapper cache update.
     *
     * @type {Array}
     */
    _classPrivateFieldInitSpec(this, _currentIndexState, {
      writable: true,
      value: []
    });
    /**
     * The flag determines if the observable is initialized or not. Not initialized object creates
     * index matrix once while emitting new changes.
     *
     * @type {boolean}
     */
    _classPrivateFieldInitSpec(this, _isMatrixIndexesInitialized, {
      writable: true,
      value: false
    });
    /**
     * The initial index value allows control from what value the index matrix array will be created.
     * Changing that value changes how the array diff generates the changes for the initial data
     * sent to the subscribers. For example, the changes can be triggered by detecting the changes
     * from `false` to `true` value or vice versa. Generally, it depends on which index map type
     * the Observable will work with. For "hiding" or "trimming" index types, it will be boolean
     * values. For various index maps, it can be anything, but I suspect that the most appropriate
     * initial value will be "undefined" in that case.
     *
     * @type {boolean}
     */
    _classPrivateFieldInitSpec(this, _initialIndexValue, {
      writable: true,
      value: false
    });
    _classPrivateFieldSet(this, _initialIndexValue, initialIndexValue !== null && initialIndexValue !== void 0 ? initialIndexValue : false);
  }

  /* eslint-disable jsdoc/require-description-complete-sentence */
  /**
   * Creates and returns a new instance of the ChangesObserver object. The resource
   * allows subscribing to the index changes that during the code running may change.
   * Changes are emitted as an array of the index change. Each change is represented
   * separately as an object with `op`, `index`, `oldValue`, and `newValue` props.
   *
   * For example:
   * ```
   * [
   *   { op: 'replace', index: 1, oldValue: false, newValue: true },
   *   { op: 'replace', index: 3, oldValue: false, newValue: true },
   *   { op: 'insert', index: 4, oldValue: false, newValue: true },
   * ]
   * // or when the new index map changes have less indexes
   * [
   *   { op: 'replace', index: 1, oldValue: false, newValue: true },
   *   { op: 'remove', index: 4, oldValue: false, newValue: true },
   * ]
   * ```
   *
   * @returns {ChangesObserver}
   */
  /* eslint-enable jsdoc/require-description-complete-sentence */
  createObserver() {
    const observer = new ChangesObserver();
    _classPrivateFieldGet(this, _observers).add(observer);
    observer.addLocalHook('unsubscribe', () => {
      _classPrivateFieldGet(this, _observers).delete(observer);
    });
    observer._writeInitialChanges(arrayDiff(_classPrivateFieldGet(this, _indexMatrix), _classPrivateFieldGet(this, _currentIndexState)));
    return observer;
  }

  /**
   * The method is an entry point for triggering new index map changes. Emitting the
   * changes triggers comparing algorithm which compares last saved state with a new
   * state. When there are some differences, the changes are sent to all subscribers.
   *
   * @param {Array} indexesState An array with index map state.
   */
  emit(indexesState) {
    let currentIndexState = _classPrivateFieldGet(this, _currentIndexState);
    if (!_classPrivateFieldGet(this, _isMatrixIndexesInitialized) || _classPrivateFieldGet(this, _indexMatrix).length !== indexesState.length) {
      if (indexesState.length === 0) {
        indexesState = new Array(currentIndexState.length).fill(_classPrivateFieldGet(this, _initialIndexValue));
      } else {
        _classPrivateFieldSet(this, _indexMatrix, new Array(indexesState.length).fill(_classPrivateFieldGet(this, _initialIndexValue)));
      }
      if (!_classPrivateFieldGet(this, _isMatrixIndexesInitialized)) {
        _classPrivateFieldSet(this, _isMatrixIndexesInitialized, true);
        currentIndexState = _classPrivateFieldGet(this, _indexMatrix);
      }
    }
    const changes = arrayDiff(currentIndexState, indexesState);
    _classPrivateFieldGet(this, _observers).forEach(observer => observer._write(changes));
    _classPrivateFieldSet(this, _currentIndexState, indexesState);
  }
}