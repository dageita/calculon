import { CELL_TYPE } from "../constants.mjs";
import VisualSelection from "../visualSelection.mjs";
/**
 * Creates the new instance of Selection responsible for highlighting currently selected cell. This type of selection
 * can present on the table only one at the time.
 *
 * @param {object} highlightParams A configuration object to create a highlight.
 * @param {Function} highlightParams.cellCornerVisible Function to determine if cell's corner should be visible.
 * @returns {Selection}
 */
function createHighlight(_ref) {
  let {
    cellCornerVisible,
    ...restOptions
  } = _ref;
  const s = new VisualSelection({
    className: 'current',
    border: {
      width: 2,
      color: '#4b89ff',
      cornerVisible: cellCornerVisible
    },
    ...restOptions,
    selectionType: CELL_TYPE
  });
  return s;
}
export default createHighlight;