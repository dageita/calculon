"use strict";

exports.__esModule = true;
var _viewportColumns = _interopRequireDefault(require("./calculator/viewportColumns"));
exports.ViewportColumnsCalculator = _viewportColumns.default;
var _viewportRows = _interopRequireDefault(require("./calculator/viewportRows"));
exports.ViewportRowsCalculator = _viewportRows.default;
var _coords = _interopRequireDefault(require("./cell/coords"));
exports.CellCoords = _coords.default;
var _range = _interopRequireDefault(require("./cell/range"));
exports.CellRange = _range.default;
var _core = _interopRequireDefault(require("./facade/core"));
exports.default = _core.default;
exports.Core = _core.default;
var _selection = _interopRequireDefault(require("./selection"));
exports.Selection = _selection.default;
var Renderer = _interopRequireWildcard(require("./renderer"));
exports.Renderer = Renderer;
var _orderView = require("./utils/orderView");
exports.OrderView = _orderView.OrderView;
exports.SharedOrderView = _orderView.SharedOrderView;
var _eventManager = require("../../../eventManager");
exports.getListenersCounter = _eventManager.getListenersCounter;
function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }
function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }
function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }