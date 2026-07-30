"use strict";

exports.__esModule = true;
exports.default = alignmentItem;
var _utils = require("../utils");
var _separator = require("./separator");
var C = _interopRequireWildcard(require("../../../i18n/constants"));
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
const KEY = 'alignment';

/**
 * @returns {object}
 */
exports.KEY = KEY;
function alignmentItem() {
  return {
    key: KEY,
    name() {
      return this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_ALIGNMENT);
    },
    disabled() {
      if (this.countRows() === 0 || this.countCols() === 0) {
        return true;
      }
      return !(this.getSelectedRange() && !this.selection.isSelectedByCorner());
    },
    submenu: {
      items: [{
        key: `${KEY}:left`,
        name() {
          let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_ALIGNMENT_LEFT);
          const hasClass = (0, _utils.checkSelectionConsistency)(this.getSelectedRange(), (row, col) => {
            const className = this.getCellMeta(row, col).className;
            if (className && className.indexOf('htLeft') !== -1) {
              return true;
            }
          });
          if (hasClass) {
            label = (0, _utils.markLabelAsSelected)(label);
          }
          return label;
        },
        callback() {
          const selectedRange = this.getSelectedRange();
          const stateBefore = (0, _utils.getAlignmentClasses)(selectedRange, (row, col) => this.getCellMeta(row, col).className);
          const type = 'horizontal';
          const alignment = 'htLeft';
          this.runHooks('beforeCellAlignment', stateBefore, selectedRange, type, alignment);
          (0, _utils.align)(selectedRange, type, alignment, (row, col) => this.getCellMeta(row, col), (row, col, key, value) => this.setCellMeta(row, col, key, value));
          this.render();
        },
        disabled: false
      }, {
        key: `${KEY}:center`,
        name() {
          let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_ALIGNMENT_CENTER);
          const hasClass = (0, _utils.checkSelectionConsistency)(this.getSelectedRange(), (row, col) => {
            const className = this.getCellMeta(row, col).className;
            if (className && className.indexOf('htCenter') !== -1) {
              return true;
            }
          });
          if (hasClass) {
            label = (0, _utils.markLabelAsSelected)(label);
          }
          return label;
        },
        callback() {
          const selectedRange = this.getSelectedRange();
          const stateBefore = (0, _utils.getAlignmentClasses)(selectedRange, (row, col) => this.getCellMeta(row, col).className);
          const type = 'horizontal';
          const alignment = 'htCenter';
          this.runHooks('beforeCellAlignment', stateBefore, selectedRange, type, alignment);
          (0, _utils.align)(selectedRange, type, alignment, (row, col) => this.getCellMeta(row, col), (row, col, key, value) => this.setCellMeta(row, col, key, value));
          this.render();
        },
        disabled: false
      }, {
        key: `${KEY}:right`,
        name() {
          let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_ALIGNMENT_RIGHT);
          const hasClass = (0, _utils.checkSelectionConsistency)(this.getSelectedRange(), (row, col) => {
            const className = this.getCellMeta(row, col).className;
            if (className && className.indexOf('htRight') !== -1) {
              return true;
            }
          });
          if (hasClass) {
            label = (0, _utils.markLabelAsSelected)(label);
          }
          return label;
        },
        callback() {
          const selectedRange = this.getSelectedRange();
          const stateBefore = (0, _utils.getAlignmentClasses)(selectedRange, (row, col) => this.getCellMeta(row, col).className);
          const type = 'horizontal';
          const alignment = 'htRight';
          this.runHooks('beforeCellAlignment', stateBefore, selectedRange, type, alignment);
          (0, _utils.align)(selectedRange, type, alignment, (row, col) => this.getCellMeta(row, col), (row, col, key, value) => this.setCellMeta(row, col, key, value));
          this.render();
        },
        disabled: false
      }, {
        key: `${KEY}:justify`,
        name() {
          let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_ALIGNMENT_JUSTIFY);
          const hasClass = (0, _utils.checkSelectionConsistency)(this.getSelectedRange(), (row, col) => {
            const className = this.getCellMeta(row, col).className;
            if (className && className.indexOf('htJustify') !== -1) {
              return true;
            }
          });
          if (hasClass) {
            label = (0, _utils.markLabelAsSelected)(label);
          }
          return label;
        },
        callback() {
          const selectedRange = this.getSelectedRange();
          const stateBefore = (0, _utils.getAlignmentClasses)(selectedRange, (row, col) => this.getCellMeta(row, col).className);
          const type = 'horizontal';
          const alignment = 'htJustify';
          this.runHooks('beforeCellAlignment', stateBefore, selectedRange, type, alignment);
          (0, _utils.align)(selectedRange, type, alignment, (row, col) => this.getCellMeta(row, col), (row, col, key, value) => this.setCellMeta(row, col, key, value));
          this.render();
        },
        disabled: false
      }, {
        name: _separator.KEY
      }, {
        key: `${KEY}:top`,
        name() {
          let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_ALIGNMENT_TOP);
          const hasClass = (0, _utils.checkSelectionConsistency)(this.getSelectedRange(), (row, col) => {
            const className = this.getCellMeta(row, col).className;
            if (className && className.indexOf('htTop') !== -1) {
              return true;
            }
          });
          if (hasClass) {
            label = (0, _utils.markLabelAsSelected)(label);
          }
          return label;
        },
        callback() {
          const selectedRange = this.getSelectedRange();
          const stateBefore = (0, _utils.getAlignmentClasses)(selectedRange, (row, col) => this.getCellMeta(row, col).className);
          const type = 'vertical';
          const alignment = 'htTop';
          this.runHooks('beforeCellAlignment', stateBefore, selectedRange, type, alignment);
          (0, _utils.align)(selectedRange, type, alignment, (row, col) => this.getCellMeta(row, col), (row, col, key, value) => this.setCellMeta(row, col, key, value));
          this.render();
        },
        disabled: false
      }, {
        key: `${KEY}:middle`,
        name() {
          let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_ALIGNMENT_MIDDLE);
          const hasClass = (0, _utils.checkSelectionConsistency)(this.getSelectedRange(), (row, col) => {
            const className = this.getCellMeta(row, col).className;
            if (className && className.indexOf('htMiddle') !== -1) {
              return true;
            }
          });
          if (hasClass) {
            label = (0, _utils.markLabelAsSelected)(label);
          }
          return label;
        },
        callback() {
          const selectedRange = this.getSelectedRange();
          const stateBefore = (0, _utils.getAlignmentClasses)(selectedRange, (row, col) => this.getCellMeta(row, col).className);
          const type = 'vertical';
          const alignment = 'htMiddle';
          this.runHooks('beforeCellAlignment', stateBefore, selectedRange, type, alignment);
          (0, _utils.align)(selectedRange, type, alignment, (row, col) => this.getCellMeta(row, col), (row, col, key, value) => this.setCellMeta(row, col, key, value));
          this.render();
        },
        disabled: false
      }, {
        key: `${KEY}:bottom`,
        name() {
          let label = this.getTranslatedPhrase(C.CONTEXTMENU_ITEMS_ALIGNMENT_BOTTOM);
          const hasClass = (0, _utils.checkSelectionConsistency)(this.getSelectedRange(), (row, col) => {
            const className = this.getCellMeta(row, col).className;
            if (className && className.indexOf('htBottom') !== -1) {
              return true;
            }
          });
          if (hasClass) {
            label = (0, _utils.markLabelAsSelected)(label);
          }
          return label;
        },
        callback() {
          const selectedRange = this.getSelectedRange();
          const stateBefore = (0, _utils.getAlignmentClasses)(selectedRange, (row, col) => this.getCellMeta(row, col).className);
          const type = 'vertical';
          const alignment = 'htBottom';
          this.runHooks('beforeCellAlignment', stateBefore, selectedRange, type, alignment);
          (0, _utils.align)(selectedRange, type, alignment, (row, col) => this.getCellMeta(row, col), (row, col, key, value) => this.setCellMeta(row, col, key, value));
          this.render();
        },
        disabled: false
      }]
    }
  };
}