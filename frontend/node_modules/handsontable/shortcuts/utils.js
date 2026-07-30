"use strict";

exports.__esModule = true;
// This file handles key-name discrepancies between browsers.
// For the list of discrepancies, go to: https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key/Key_Values.
const mappings = new Map([[' ', 'space'],
// custom mapping
['spacebar', 'space'], ['scroll', 'scrolllock'], ['del', 'delete'], ['esc', 'escape'], ['medianexttrack', 'mediatracknext'], ['mediaprevioustrack', 'mediatrackprevious'], ['volumeup', 'audiovolumeup'], ['volumedown', 'audiovolumedown'], ['volumemute', 'audiovolumemute'], ['multiply', '*'], ['add', '+'], ['divide', '/'], ['subtract', '-'], ['left', 'arrowleft'], ['right', 'arrowright'], ['up', 'arrowup'], ['down', 'arrowdown']]);

/* eslint-disable jsdoc/require-description-complete-sentence */

/**
 * Get a single, normalized string from the list of the `KeyboardEvent.key` properties.
 *
 * @param {Array<string>} keys The list of the `KeyboardEvent.key` properties
 * @returns {string}
 */
const normalizeKeys = keys => {
  return keys.map(key => {
    const lowercaseKey = key.toLowerCase();
    if (mappings.has(lowercaseKey)) {
      return mappings.get(lowercaseKey);
    }
    return lowercaseKey;
  }).sort().join('+');
};

/**
 * Get the list of the `KeyboardEvent.key` properties from a single, normalized string.
 *
 * @param {string} normalizedKeys A single, normalized string that contains the list of the `KeyboardEvent.key` properties
 * @returns {Array<string>}
 */
exports.normalizeKeys = normalizeKeys;
const getKeysList = normalizedKeys => {
  return normalizedKeys.split('+');
};

/**
 * Normalize a `KeyboardEvent.key` property, to use it for keyboard shortcuts.
 *
 * @param {string} key KeyboardEvent's key property.
 * @returns {string}
 */
exports.getKeysList = getKeysList;
const normalizeEventKey = key => {
  return key.toLowerCase();
};
exports.normalizeEventKey = normalizeEventKey;