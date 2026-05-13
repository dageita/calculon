"use strict";

exports.__esModule = true;
exports.align = align;
exports.checkSelectionConsistency = checkSelectionConsistency;
exports.filterSeparators = filterSeparators;
exports.getAlignmentClasses = getAlignmentClasses;
exports.getValidSelection = getValidSelection;
exports.hasSubMenu = hasSubMenu;
exports.isDisabled = isDisabled;
exports.isItemHidden = isItemHidden;
exports.isSelectionDisabled = isSelectionDisabled;
exports.isSeparator = isSeparator;
exports.markLabelAsSelected = markLabelAsSelected;
exports.normalizeSelection = normalizeSelection;
exports.prepareHorizontalAlignClass = prepareHorizontalAlignClass;
exports.prepareVerticalAlignClass = prepareVerticalAlignClass;
require("core-js/modules/es.array.push.js");
var _array = require("../../helpers/array");
var _element = require("../../helpers/dom/element");
var _separator = require("./predefinedItems/separator");
/**
 * @param {CellRange[]} selRanges An array of the cell ranges.
 * @returns {object[]}
 */
function normalizeSelection(selRanges) {
  return (0, _array.arrayMap)(selRanges, range => ({
    start: range.getTopStartCorner(),
    end: range.getBottomEndCorner()
  }));
}

/**
 * @param {HTMLElement} cell The HTML cell element to check.
 * @returns {boolean}
 */
function isSeparator(cell) {
  return (0, _element.hasClass)(cell, 'htSeparator');
}

/**
 * @param {HTMLElement} cell The HTML cell element to check.
 * @returns {boolean}
 */
function hasSubMenu(cell) {
  return (0, _element.hasClass)(cell, 'htSubmenu');
}

/**
 * @param {HTMLElement} cell The HTML cell element to check.
 * @returns {boolean}
 */
function isDisabled(cell) {
  return (0, _element.hasClass)(cell, 'htDisabled');
}

/**
 * @param {HTMLElement} cell The HTML cell element to check.
 * @returns {boolean}
 */
function isSelectionDisabled(cell) {
  return (0, _element.hasClass)(cell, 'htSelectionDisabled');
}

/**
 * @param {Core} hot The Handsontable instance.
 * @returns {Array[]|null}
 */
function getValidSelection(hot) {
  const selected = hot.getSelected();
  if (!selected) {
    return null;
  }
  if (selected[0] < 0) {
    return null;
  }
  return selected;
}

/**
 * @param {string} className The full element class name to process.
 * @param {string} alignment The slignment class name to compare with.
 * @returns {string}
 */
function prepareVerticalAlignClass(className, alignment) {
  if (className.indexOf(alignment) !== -1) {
    return className;
  }
  const replacedClassName = className.replace('htTop', '').replace('htMiddle', '').replace('htBottom', '').replace('  ', '');
  return `${replacedClassName} ${alignment}`;
}

/**
 * @param {string} className The full element class name to process.
 * @param {string} alignment The slignment class name to compare with.
 * @returns {string}
 */
function prepareHorizontalAlignClass(className, alignment) {
  if (className.indexOf(alignment) !== -1) {
    return className;
  }
  const replacedClassName = className.replace('htLeft', '').replace('htCenter', '').replace('htRight', '').replace('htJustify', '').replace('  ', '');
  return `${replacedClassName} ${alignment}`;
}

/**
 * @param {CellRange[]} ranges An array of the cell ranges.
 * @param {Function} callback The callback function.
 * @returns {object}
 */
function getAlignmentClasses(ranges, callback) {
  const classes = {};
  (0, _array.arrayEach)(ranges, range => {
    range.forAll((row, col) => {
      // Alignment classes should only collected within cell ranges. We skip header coordinates.
      if (row >= 0 && col >= 0) {
        if (!classes[row]) {
          classes[row] = [];
        }
        classes[row][col] = callback(row, col);
      }
    });
  });
  return classes;
}

/**
 * @param {CellRange[]} ranges An array of the cell ranges.
 * @param {string} type The type of the alignment axis ('horizontal' or 'vertical').
 * @param {string} alignment CSS class name to add.
 * @param {Function} cellDescriptor The function which fetches the cell meta object based in passed coordinates.
 * @param {Function} propertySetter The function which contains logic for added/removed alignment.
 */
function align(ranges, type, alignment, cellDescriptor, propertySetter) {
  (0, _array.arrayEach)(ranges, range => {
    range.forAll((row, col) => {
      // Alignment classes should only collected within cell ranges. We skip header coordinates.
      if (row >= 0 && col >= 0) {
        applyAlignClassName(row, col, type, alignment, cellDescriptor, propertySetter);
      }
    });
  });
}

/**
 * @param {number} row The visual row index.
 * @param {number} col The visual column index.
 * @param {string} type The type of the alignment axis ('horizontal' or 'vertical').
 * @param {string} alignment CSS class name to add.
 * @param {Function} cellDescriptor The function which fetches the cell meta object based in passed coordinates.
 * @param {Function} propertySetter The function which contains logic for added/removed alignment.
 */
function applyAlignClassName(row, col, type, alignment, cellDescriptor, propertySetter) {
  const cellMeta = cellDescriptor(row, col);
  let className = alignment;
  if (cellMeta.className) {
    if (type === 'vertical') {
      className = prepareVerticalAlignClass(cellMeta.className, alignment);
    } else {
      className = prepareHorizontalAlignClass(cellMeta.className, alignment);
    }
  }
  propertySetter(row, col, 'className', className);
}

/**
 * @param {CellRange[]} ranges An array of the cell ranges.
 * @param {Function} comparator The comparator function.
 * @returns {boolean}
 */
function checkSelectionConsistency(ranges, comparator) {
  let result = false;
  if (Array.isArray(ranges)) {
    (0, _array.arrayEach)(ranges, range => {
      range.forAll((row, col) => {
        // Selection consistency should only check within cell ranges. We skip header coordinates.
        if (row >= 0 && col >= 0 && comparator(row, col)) {
          result = true;
          return false;
        }
      });
      return result;
    });
  }
  return result;
}

/**
 * @param {string} label The label text.
 * @returns {string}
 */
function markLabelAsSelected(label) {
  // workaround for https://github.com/handsontable/handsontable/issues/1946
  return `<span class="selected">${String.fromCharCode(10003)}</span>${label}`;
}

/**
 * @param {object} item The object which describes the context menu item properties.
 * @param {Core} instance The Handsontable instance.
 * @returns {boolean}
 */
function isItemHidden(item, instance) {
  return !item.hidden || !(typeof item.hidden === 'function' && item.hidden.call(instance));
}

/**
 * @param {object[]} items The context menu items collection.
 * @param {string} separator The string which identifies the context menu separator item.
 * @returns {object[]}
 */
function shiftSeparators(items, separator) {
  const result = items.slice(0);
  for (let i = 0; i < result.length;) {
    if (result[i].name === separator) {
      result.shift();
    } else {
      break;
    }
  }
  return result;
}

/**
 * @param {object[]} items The context menu items collection.
 * @param {string} separator The string which identifies the context menu separator item.
 * @returns {object[]}
 */
function popSeparators(items, separator) {
  let result = items.slice(0);
  result.reverse();
  result = shiftSeparators(result, separator);
  result.reverse();
  return result;
}

/**
 * Removes duplicated menu separators from the context menu items collection.
 *
 * @param {object[]} items The context menu items collection.
 * @returns {object[]}
 */
function removeDuplicatedSeparators(items) {
  const result = [];
  (0, _array.arrayEach)(items, (value, index) => {
    if (index > 0) {
      if (result[result.length - 1].name !== value.name) {
        result.push(value);
      }
    } else {
      result.push(value);
    }
  });
  return result;
}

/**
 * Removes menu separators from the context menu items collection.
 *
 * @param {object[]} items The context menu items collection.
 * @param {string} separator The string which identifies the context menu separator item.
 * @returns {object[]}
 */
function filterSeparators(items) {
  let separator = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : _separator.KEY;
  let result = items.slice(0);
  result = shiftSeparators(result, separator);
  result = popSeparators(result, separator);
  result = removeDuplicatedSeparators(result);
  return result;
}