import { ACTIVE_HEADER_TYPE } from "../constants.mjs";
import VisualSelection from "../visualSelection.mjs";
/**
 * @param {object} highlightParams A configuration object to create a highlight.
 * @param {string} highlightParams.activeHeaderClassName Highlighted headers' class name.
 * @returns {Selection}
 */
function createHighlight(_ref) {
  let {
    activeHeaderClassName,
    ...restOptions
  } = _ref;
  const s = new VisualSelection({
    highlightHeaderClassName: activeHeaderClassName,
    ...restOptions,
    selectionType: ACTIVE_HEADER_TYPE
  });
  return s;
}
export default createHighlight;