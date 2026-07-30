"use strict";

exports.__esModule = true;
/**
 * @typedef {'top'|'bottom'|'inline_start'|'top_inline_start_corner'|'bottom_inline_start_corner'} CLONE_TYPES_ENUM
 */
const CLONE_TOP = 'top';
exports.CLONE_TOP = CLONE_TOP;
const CLONE_BOTTOM = 'bottom';
exports.CLONE_BOTTOM = CLONE_BOTTOM;
const CLONE_INLINE_START = 'inline_start';
exports.CLONE_INLINE_START = CLONE_INLINE_START;
const CLONE_TOP_INLINE_START_CORNER = 'top_inline_start_corner';
exports.CLONE_TOP_INLINE_START_CORNER = CLONE_TOP_INLINE_START_CORNER;
const CLONE_BOTTOM_INLINE_START_CORNER = 'bottom_inline_start_corner';
exports.CLONE_BOTTOM_INLINE_START_CORNER = CLONE_BOTTOM_INLINE_START_CORNER;
const CLONE_TYPES = [CLONE_TOP, CLONE_BOTTOM, CLONE_INLINE_START, CLONE_TOP_INLINE_START_CORNER, CLONE_BOTTOM_INLINE_START_CORNER];
exports.CLONE_TYPES = CLONE_TYPES;
const CLONE_CLASS_NAMES = new Map([[CLONE_TOP, `ht_clone_${CLONE_TOP}`], [CLONE_BOTTOM, `ht_clone_${CLONE_BOTTOM}`], [CLONE_INLINE_START, `ht_clone_${CLONE_INLINE_START} ht_clone_left`], [CLONE_TOP_INLINE_START_CORNER, `ht_clone_${CLONE_TOP_INLINE_START_CORNER} ht_clone_top_left_corner`], [CLONE_BOTTOM_INLINE_START_CORNER, `ht_clone_${CLONE_BOTTOM_INLINE_START_CORNER} ht_clone_bottom_left_corner`]]);
exports.CLONE_CLASS_NAMES = CLONE_CLASS_NAMES;