import staticRegister from "./../../../utils/staticRegister.mjs";
import { ACTIVE_HEADER_TYPE, AREA_TYPE, CELL_TYPE, CUSTOM_SELECTION_TYPE, FILL_TYPE, HEADER_TYPE } from "../constants.mjs";
import activeHeaderHighlight from "./activeHeader.mjs";
import areaHighlight from "./area.mjs";
import cellHighlight from "./cell.mjs";
import customSelection from "./customSelection.mjs";
import fillHighlight from "./fill.mjs";
import headerHighlight from "./header.mjs";
const {
  register,
  getItem
} = staticRegister('highlight/types');
register(ACTIVE_HEADER_TYPE, activeHeaderHighlight);
register(AREA_TYPE, areaHighlight);
register(CELL_TYPE, cellHighlight);
register(CUSTOM_SELECTION_TYPE, customSelection);
register(FILL_TYPE, fillHighlight);
register(HEADER_TYPE, headerHighlight);

/**
 * @param {string} highlightType The selection type.
 * @param {object} options The selection options.
 * @returns {Selection}
 */
function createHighlight(highlightType, options) {
  return getItem(highlightType)({
    type: highlightType,
    ...options
  });
}
export { createHighlight };