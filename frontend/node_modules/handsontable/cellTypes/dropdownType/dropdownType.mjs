import { DropdownEditor } from "../../editors/dropdownEditor/index.mjs";
import { autocompleteRenderer } from "../../renderers/autocompleteRenderer/index.mjs";
import { autocompleteValidator } from "../../validators/autocompleteValidator/index.mjs";
export const CELL_TYPE = 'dropdown';
export const DropdownCellType = {
  CELL_TYPE,
  editor: DropdownEditor,
  // displays small gray arrow on right side of the cell
  renderer: autocompleteRenderer,
  validator: autocompleteValidator
};