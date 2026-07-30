import "core-js/modules/es.array.push.js";
import "core-js/modules/es.error.cause.js";
function _classPrivateFieldInitSpec(obj, privateMap, value) { _checkPrivateRedeclaration(obj, privateMap); privateMap.set(obj, value); }
function _checkPrivateRedeclaration(obj, privateCollection) { if (privateCollection.has(obj)) { throw new TypeError("Cannot initialize the same private elements twice on an object"); } }
function _classPrivateFieldGet(receiver, privateMap) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "get"); return _classApplyDescriptorGet(receiver, descriptor); }
function _classApplyDescriptorGet(receiver, descriptor) { if (descriptor.get) { return descriptor.get.call(receiver); } return descriptor.value; }
function _classPrivateFieldSet(receiver, privateMap, value) { var descriptor = _classExtractFieldDescriptor(receiver, privateMap, "set"); _classApplyDescriptorSet(receiver, descriptor, value); return value; }
function _classExtractFieldDescriptor(receiver, privateMap, action) { if (!privateMap.has(receiver)) { throw new TypeError("attempted to " + action + " private field on non-instance"); } return privateMap.get(receiver); }
function _classApplyDescriptorSet(receiver, descriptor, value) { if (descriptor.set) { descriptor.set.call(receiver, value); } else { if (!descriptor.writable) { throw new TypeError("attempted to set read only private field"); } descriptor.value = value; } }
import { toUpperCaseFirst } from "../../../helpers/string.mjs";
/**
 * @private
 * @class IndexSyncer
 * @description
 *
 * Indexes synchronizer responsible for providing logic for particular axis. It respects an idea to represent trimmed
 * elements in HF's engine to perform formulas calculations on them. It also provides method for translation from visual
 * row/column indexes to HF's row/column indexes.
 */
var _axis = /*#__PURE__*/new WeakMap();
var _indexMapper = /*#__PURE__*/new WeakMap();
var _indexSyncer = /*#__PURE__*/new WeakMap();
var _indexesSequence = /*#__PURE__*/new WeakMap();
var _movedIndexes = /*#__PURE__*/new WeakMap();
var _finalIndex = /*#__PURE__*/new WeakMap();
var _removedIndexes = /*#__PURE__*/new WeakMap();
class AxisSyncer {
  constructor(axis, indexMapper, indexSyncer) {
    /**
     * The axis for which the actions are performed.
     *
     * @private
     * @type {'row'|'column'}
     */
    _classPrivateFieldInitSpec(this, _axis, {
      writable: true,
      value: void 0
    });
    /**
     * Reference to index mapper.
     *
     * @private
     * @type {IndexMapper}
     */
    _classPrivateFieldInitSpec(this, _indexMapper, {
      writable: true,
      value: void 0
    });
    /**
     * The index synchronizer for both axis (is storing some more general information).
     *
     * @private
     * @type {IndexSyncer}
     */
    _classPrivateFieldInitSpec(this, _indexSyncer, {
      writable: true,
      value: void 0
    });
    /**
     * Sequence of physical indexes stored for watching changes and calculating some transformations.
     *
     * @private
     * @type {Array<number>}
     */
    _classPrivateFieldInitSpec(this, _indexesSequence, {
      writable: true,
      value: []
    });
    /**
     * List of moved HF indexes, stored before performing move on HOT to calculate transformation needed on HF's engine.
     *
     * @private
     * @type {Array<number>}
     */
    _classPrivateFieldInitSpec(this, _movedIndexes, {
      writable: true,
      value: []
    });
    /**
     * Final HF's place where to move indexes, stored before performing move on HOT to calculate transformation needed on HF's engine.
     *
     * @private
     * @type {number|undefined}
     */
    _classPrivateFieldInitSpec(this, _finalIndex, {
      writable: true,
      value: void 0
    });
    /**
     * List of removed HF indexes, stored before performing removal on HOT to calculate transformation needed on HF's engine.
     *
     * @private
     * @type {Array<number>}
     */
    _classPrivateFieldInitSpec(this, _removedIndexes, {
      writable: true,
      value: []
    });
    _classPrivateFieldSet(this, _axis, axis);
    _classPrivateFieldSet(this, _indexMapper, indexMapper);
    _classPrivateFieldSet(this, _indexSyncer, indexSyncer);
  }

  /**
   * Sets removed HF indexes (it should be done right before performing move on HOT).
   *
   * @param {Array<number>} removedIndexes List of removed physical indexes.
   * @returns {Array<number>} List of removed visual indexes.
   */
  setRemovedHfIndexes(removedIndexes) {
    _classPrivateFieldSet(this, _removedIndexes, removedIndexes.map(physicalIndex => {
      const visualIndex = _classPrivateFieldGet(this, _indexMapper).getVisualFromPhysicalIndex(physicalIndex);
      return this.getHfIndexFromVisualIndex(visualIndex);
    }));
    return _classPrivateFieldGet(this, _removedIndexes);
  }

  /**
   * Gets removed HF indexes (right before performing removal on HOT).
   *
   * @returns {Array<number>} List of removed HF indexes.
   */
  getRemovedHfIndexes() {
    return _classPrivateFieldGet(this, _removedIndexes);
  }

  /**
   * Gets corresponding HyperFormula index for particular visual index. It's respecting the idea that HF's engine
   * is fed also with trimmed indexes (business requirements for formula result calculation also for trimmed elements).
   *
   * @param {number} visualIndex Visual index.
   * @returns {number}
   */
  getHfIndexFromVisualIndex(visualIndex) {
    const indexesSequence = _classPrivateFieldGet(this, _indexMapper).getIndexesSequence();
    const notTrimmedIndexes = _classPrivateFieldGet(this, _indexMapper).getNotTrimmedIndexes();
    return indexesSequence.indexOf(notTrimmedIndexes[visualIndex]);
  }

  /**
   * Synchronizes moves done on HOT to HF engine (based on previously calculated positions).
   *
   * @private
   * @param {Array<{from: number, to: number}>} moves Calculated HF's move positions.
   */
  syncMoves(moves) {
    const NUMBER_OF_MOVED_INDEXES = 1;
    const SYNC_MOVE_METHOD_NAME = `move${toUpperCaseFirst(_classPrivateFieldGet(this, _axis))}s`;
    _classPrivateFieldGet(this, _indexSyncer).getEngine().batch(() => {
      moves.forEach(move => {
        const moveToTheSamePosition = move.from !== move.to;
        // Moving from left to right (or top to bottom) to a line (drop index) right after already moved element.
        const anotherMoveWithoutEffect = move.from + 1 !== move.to;
        if (moveToTheSamePosition && anotherMoveWithoutEffect) {
          _classPrivateFieldGet(this, _indexSyncer).getEngine()[SYNC_MOVE_METHOD_NAME](_classPrivateFieldGet(this, _indexSyncer).getSheetId(), move.from, NUMBER_OF_MOVED_INDEXES, move.to);
        }
      });
    });
  }

  /**
   * Stores information about performed HOT moves for purpose of calculating where to move HF elements.
   *
   * @param {Array<number>} movedVisualIndexes Sequence of moved visual indexes for certain axis.
   * @param {number} visualFinalIndex Final visual place where to move HOT indexes.
   * @param {boolean} movePossible Indicates if it's possible to move HOT indexes to the desired position.
   */
  storeMovesInformation(movedVisualIndexes, visualFinalIndex, movePossible) {
    if (movePossible === false) {
      return;
    }
    _classPrivateFieldSet(this, _movedIndexes, movedVisualIndexes.map(index => this.getHfIndexFromVisualIndex(index)));
    _classPrivateFieldSet(this, _finalIndex, this.getHfIndexFromVisualIndex(visualFinalIndex));
  }

  /**
   * Gets first position where to move element (respecting the fact that some element will be sooner or later
   * taken out of the dataset in order to move them).
   *
   * @param {Array<number>} movedHfIndexes Sequence of moved HF indexes for certain axis.
   * @param {number} finalHfIndex Final HF place where to move rows.
   * @returns {number} HF's index informing where to move the first element.
   * @private
   */
  getMoveLine(movedHfIndexes, finalHfIndex) {
    const numberOfElements = _classPrivateFieldGet(this, _indexMapper).getNumberOfIndexes();
    const notMovedElements = Array.from(Array(numberOfElements).keys()).filter(index => movedHfIndexes.includes(index) === false);
    if (finalHfIndex === 0) {
      var _notMovedElements$fin;
      return (_notMovedElements$fin = notMovedElements[finalHfIndex]) !== null && _notMovedElements$fin !== void 0 ? _notMovedElements$fin : 0; // Moving before the first dataset's element.
    }

    return notMovedElements[finalHfIndex - 1] + 1; // Moving before another element.
  }

  /**
   * Gets initially calculated HF's move positions.
   *
   * @private
   * @param {Array<number>} movedHfIndexes Sequence of moved HF indexes for certain axis.
   * @param {number} finalHfIndex Final HF place where to move rows.
   * @returns {Array<{from: number, to: number}>} Initially calculated HF's move positions.
   */
  getInitiallyCalculatedMoves(movedHfIndexes, finalHfIndex) {
    let moveLine = this.getMoveLine(movedHfIndexes, finalHfIndex);
    const moves = [];
    movedHfIndexes.forEach(movedHfIndex => {
      const move = {
        from: movedHfIndex,
        to: moveLine
      };
      moves.forEach(previouslyMovedIndex => {
        const isMovingFromEndToStart = previouslyMovedIndex.from > previouslyMovedIndex.to;
        const isMovingElementBefore = previouslyMovedIndex.to <= move.from;
        const isMovingAfterElement = previouslyMovedIndex.from > move.from;
        if (isMovingAfterElement && isMovingElementBefore && isMovingFromEndToStart) {
          move.from += 1;
        }
      });

      // Moved element from right to left (or bottom to top).
      if (move.from >= moveLine) {
        moveLine += 1;
      }
      moves.push(move);
    });
    return moves;
  }

  /**
   * Gets finally calculated HF's move positions (after adjusting).
   *
   * @private
   * @param {Array<{from: number, to: number}>} moves Initially calculated HF's move positions.
   * @returns {Array<{from: number, to: number}>} Finally calculated HF's move positions (after adjusting).
   */
  adjustedCalculatedMoves(moves) {
    moves.forEach((move, index) => {
      const nextMoved = moves.slice(index + 1);
      nextMoved.forEach(nextMovedIndex => {
        const isMovingFromStartToEnd = nextMovedIndex.from < nextMovedIndex.to;
        if (nextMovedIndex.from > move.from && isMovingFromStartToEnd) {
          nextMovedIndex.from -= 1;
        }
      });
    });
    return moves;
  }

  /**
   * Calculating where to move HF elements and performing already calculated moves.
   *
   * @param {boolean} movePossible Indicates if it was possible to move HOT indexes to the desired position.
   * @param {boolean} orderChanged Indicates if order of HOT indexes was changed by move.
   */
  calculateAndSyncMoves(movePossible, orderChanged) {
    if (_classPrivateFieldGet(this, _indexSyncer).isPerformingUndoRedo()) {
      return;
    }
    if (movePossible === false || orderChanged === false) {
      return;
    }
    const calculatedMoves = this.adjustedCalculatedMoves(this.getInitiallyCalculatedMoves(_classPrivateFieldGet(this, _movedIndexes), _classPrivateFieldGet(this, _finalIndex)));
    if (_classPrivateFieldGet(this, _indexSyncer).getSheetId() === null) {
      _classPrivateFieldGet(this, _indexSyncer).getPostponeAction(() => this.syncMoves(calculatedMoves));
    } else {
      this.syncMoves(calculatedMoves);
    }
  }

  /**
   * Gets callback for hook triggered after performing change of indexes order.
   *
   * @returns {Function}
   */
  getIndexesChangeSyncMethod() {
    const SYNC_ORDER_CHANGE_METHOD_NAME = `set${toUpperCaseFirst(_classPrivateFieldGet(this, _axis))}Order`;
    return source => {
      if (_classPrivateFieldGet(this, _indexSyncer).isPerformingUndoRedo()) {
        return;
      }
      const newSequence = _classPrivateFieldGet(this, _indexMapper).getIndexesSequence();
      if (source === 'update') {
        const relativeTransformation = _classPrivateFieldGet(this, _indexesSequence).map(index => newSequence.indexOf(index));
        const sheetDimensions = _classPrivateFieldGet(this, _indexSyncer).getEngine().getSheetDimensions(_classPrivateFieldGet(this, _indexSyncer).getSheetId());
        let sizeForAxis;
        if (_classPrivateFieldGet(this, _axis) === 'row') {
          sizeForAxis = sheetDimensions.height;
        } else {
          sizeForAxis = sheetDimensions.width;
        }
        const numberOfReorganisedIndexes = relativeTransformation.length;

        // Sheet dimension can be changed by HF's engine for purpose of calculating values. It extends dependency
        // graph to calculate values outside of a defined dataset. This part of code could be removed after resolving
        // feature request from HF issue board (handsontable/hyperformula#1179).
        for (let i = numberOfReorganisedIndexes; i < sizeForAxis; i += 1) {
          relativeTransformation.push(i);
        }
        _classPrivateFieldGet(this, _indexSyncer).getEngine()[SYNC_ORDER_CHANGE_METHOD_NAME](_classPrivateFieldGet(this, _indexSyncer).getSheetId(), relativeTransformation);
      }
      _classPrivateFieldSet(this, _indexesSequence, newSequence);
    };
  }

  /**
   * Initialize the AxisSyncer.
   */
  init() {
    _classPrivateFieldSet(this, _indexesSequence, _classPrivateFieldGet(this, _indexMapper).getIndexesSequence());
  }
}
export default AxisSyncer;