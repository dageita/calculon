import "core-js/modules/es.error.cause.js";
function _classPrivateFieldInitSpec(obj, privateMap, value) { _checkPrivateRedeclaration(obj, privateMap); privateMap.set(obj, value); }
function _checkPrivateRedeclaration(obj, privateCollection) { if (privateCollection.has(obj)) { throw new TypeError("Cannot initialize the same private elements twice on an object"); } }
function _classPrivateFieldGet(receiver, privateMap) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "get"); return _classApplyDescriptorGet(receiver, descriptor); }
function _classApplyDescriptorGet(receiver, descriptor) { if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }
function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "set"); _classApplyDescriptorSet(receiver, descriptor, value); return value; }
function _classExtractFieldDescriptor(receiver, privateMap, action) { if (!privateMap.has(receiver)) { throw new TypeError("attempted to " + action + " private field on non-instance"); } return privateMap.get(receiver); }
function _classApplyDescriptorSet(receiver, descriptor, value) { if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } }
import AxisSyncer from "./axisSyncer.mjs";
/**
 * @private
 * @class IndexSyncer
 * @description
 *
 * Indexes synchronizer responsible for providing logic for syncing actions done on indexes for HOT to actions performed
 * on HF's.
 *
 */
var _rowIndexSyncer = /*#__PURE__*/new WeakMap();
var _columnIndexSyncer = /*#__PURE__*/new WeakMap();
var _postponeAction = /*#__PURE__*/new WeakMap();
var _isPerformingUndo = /*#__PURE__*/new WeakMap();
var _isPerformingRedo = /*#__PURE__*/new WeakMap();
var _engine = /*#__PURE__*/new WeakMap();
var _sheetId = /*#__PURE__*/new WeakMap();
class IndexSyncer {
  constructor(rowIndexMapper, columnIndexMapper, postponeAction) {
    /**
     * Indexes synchronizer for the axis of the rows.
     *
     * @private
     * @type {AxisSyncer}
     */
    _classPrivateFieldInitSpec(this, _rowIndexSyncer, {
      writable: true,
      value: void 0
    });
    /**
     * Indexes synchronizer for the axis of the columns.
     *
     * @private
     * @type {AxisSyncer}
     */
    _classPrivateFieldInitSpec(this, _columnIndexSyncer, {
      writable: true,
      value: void 0
    });
    /**
     * Method which will postpone execution of some action (needed when synchronization endpoint isn't setup yet).
     *
     * @private
     * @type {Function}
     */
    _classPrivateFieldInitSpec(this, _postponeAction, {
      writable: true,
      value: void 0
    });
    /**
     * Flag informing whether undo is already performed (we don't perform synchronization in such case).
     *
     * @private
     * @type {boolean}
     */
    _classPrivateFieldInitSpec(this, _isPerformingUndo, {
      writable: true,
      value: false
    });
    /**
     * Flag informing whether redo is already performed (we don't perform synchronization in such case).
     *
     * @private
     * @type {boolean}
     */
    _classPrivateFieldInitSpec(this, _isPerformingRedo, {
      writable: true,
      value: false
    });
    /**
     * The HF's engine instance which will be synced.
     *
     * @private
     * @type {HyperFormula|null}
     */
    _classPrivateFieldInitSpec(this, _engine, {
      writable: true,
      value: null
    });
    /**
     * HyperFormula's sheet name.
     *
     * @private
     * @type {string|null}
     */
    _classPrivateFieldInitSpec(this, _sheetId, {
      writable: true,
      value: null
    });
    _classPrivateFieldSet(this, _rowIndexSyncer, new AxisSyncer('row', rowIndexMapper, this));
    _classPrivateFieldSet(this, _columnIndexSyncer, new AxisSyncer('column', columnIndexMapper, this));
    _classPrivateFieldSet(this, _postponeAction, postponeAction);
  }

  /**
   * Gets index synchronizer for a particular axis.
   *
   * @param {'row'|'column'} indexType Type of indexes.
   * @returns {AxisSyncer}
   */
  getForAxis(indexType) {
    if (indexType === 'row') {
      return _classPrivateFieldGet(this, _rowIndexSyncer);
    }
    return _classPrivateFieldGet(this, _columnIndexSyncer);
  }

  /**
   * Sets flag informing whether an undo action is already performed (we don't execute synchronization in such case).
   *
   * @param {boolean} flagValue Boolean value for the flag.
   */
  setPerformUndo(flagValue) {
    _classPrivateFieldSet(this, _isPerformingUndo, flagValue);
  }

  /**
   * Sets flag informing whether a redo action is already performed (we don't execute synchronization in such case).
   *
   * @param {boolean} flagValue Boolean value for the flag.
   */
  setPerformRedo(flagValue) {
    _classPrivateFieldSet(this, _isPerformingRedo, flagValue);
  }

  /**
   * Gets information whether redo or undo action is already performed (we don't execute synchronization in such case).
   *
   * @private
   * @returns {boolean}
   */
  isPerformingUndoRedo() {
    return _classPrivateFieldGet(this, _isPerformingUndo) || _classPrivateFieldGet(this, _isPerformingRedo);
  }

  /**
   * Gets HyperFormula's sheet id.
   *
   * @returns {string|null}
   */
  getSheetId() {
    return _classPrivateFieldGet(this, _sheetId);
  }

  /**
   * Gets engine instance that will be used for handled instance of Handsontable.
   *
   * @type {HyperFormula|null}
   */
  getEngine() {
    return _classPrivateFieldGet(this, _engine);
  }

  /**
   * Gets method which will postpone execution of some action (needed when synchronization endpoint isn't setup yet).
   *
   * @returns {Function}
   */
  getPostponeAction() {
    return _classPrivateFieldGet(this, _postponeAction);
  }

  /**
   * Setups a synchronization endpoint.
   *
   * @param {HyperFormula|null} engine The HF's engine instance which will be synced.
   * @param {string|null} sheetId HyperFormula's sheet name.
   */
  setupSyncEndpoint(engine, sheetId) {
    _classPrivateFieldSet(this, _engine, engine);
    _classPrivateFieldSet(this, _sheetId, sheetId);
    _classPrivateFieldGet(this, _rowIndexSyncer).init();
    _classPrivateFieldGet(this, _columnIndexSyncer).init();
  }
}
export default IndexSyncer;