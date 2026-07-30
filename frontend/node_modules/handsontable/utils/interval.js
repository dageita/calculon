"use strict";

exports.__esModule = true;
exports.parseDelay = parseDelay;
var _feature = require("./../helpers/feature");
/**
 * @class Interval
 */
class Interval {
  static create(func, delay) {
    return new Interval(func, delay);
  }
  constructor(func, delay) {
    /**
     * Animation frame request id.
     *
     * @type {number}
     */
    this.timer = null;
    /**
     * Function to invoke repeatedly.
     *
     * @type {Function}
     */
    this.func = func;
    /**
     * Number of milliseconds that function should wait before next call.
     */
    this.delay = parseDelay(delay);
    /**
     * Flag which indicates if interval object was stopped.
     *
     * @type {boolean}
     * @default true
     */
    this.stopped = true;
    /**
     * Interval time (in milliseconds) of the last callback call.
     *
     * @private
     * @type {number}
     */
    this._then = null;
    /**
     * Bounded function `func`.
     *
     * @private
     * @type {Function}
     */
    this._callback = () => this.__callback();
  }

  /**
   * Start loop.
   *
   * @returns {Interval}
   */
  start() {
    if (this.stopped) {
      this._then = Date.now();
      this.stopped = false;
      this.timer = (0, _feature.requestAnimationFrame)(this._callback);
    }
    return this;
  }

  /**
   * Stop looping.
   *
   * @returns {Interval}
   */
  stop() {
    if (!this.stopped) {
      this.stopped = true;
      (0, _feature.cancelAnimationFrame)(this.timer);
      this.timer = null;
    }
    return this;
  }

  /**
   * Loop callback, fired on every animation frame.
   *
   * @private
   */
  __callback() {
    this.timer = (0, _feature.requestAnimationFrame)(this._callback);
    if (this.delay) {
      const now = Date.now();
      const elapsed = now - this._then;
      if (elapsed > this.delay) {
        this._then = now - elapsed % this.delay;
        this.func();
      }
    } else {
      this.func();
    }
  }
}
var _default = Interval;
/**
 * Convert delay from string format to milliseconds.
 *
 * @param {number|string} delay The delay in FPS (frame per second) or number format.
 * @returns {number}
 */
exports.default = _default;
function parseDelay(delay) {
  let result = delay;
  if (typeof result === 'string' && /fps$/.test(result)) {
    result = 1000 / parseInt(result.replace('fps', '') || 0, 10);
  }
  return result;
}