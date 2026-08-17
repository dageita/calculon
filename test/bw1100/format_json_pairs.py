#!/usr/bin/env python3
"""Format JSON with numeric curve points kept on one line.

For example, a two-number array in an efficiency curve becomes::

    [2.56e-07, 0.001 ],

The default target is ``systems/BW1100.json`` and is updated atomically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _number(value: int | float) -> str:
    if isinstance(value, bool):
        raise TypeError('bool is not a numeric curve value')
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('JSON does not permit NaN or infinity')
        return repr(value)
    return str(value)


def _is_numeric_pair(value: object) -> bool:
    return (isinstance(value, list) and len(value) == 2 and
            all(isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value))


def render(value: object, level: int = 0, space_before_close: bool = True) -> str:
    """Render JSON while compacting only two-number arrays.

    Other arrays retain normal pretty-printed indentation, so this never makes
    longer efficiency tables or configuration lists hard to review.
    """
    indent = '  ' * level
    child_indent = '  ' * (level + 1)
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return _number(value)
    if _is_numeric_pair(value):
        close = ' ]' if space_before_close else ']'
        return f'[{_number(value[0])}, {_number(value[1])}{close}'
    if isinstance(value, list):
        if not value:
            return '[]'
        items = [render(item, level + 1, space_before_close) for item in value]
        return '[\n' + ',\n'.join(child_indent + item for item in items) + '\n' + indent + ']'
    if isinstance(value, dict):
        if not value:
            return '{}'
        items = []
        for key, item in value.items():
            encoded_key = json.dumps(str(key), ensure_ascii=False)
            items.append(f'{encoded_key}: {render(item, level + 1, space_before_close)}')
        return '{\n' + ',\n'.join(child_indent + item for item in items) + '\n' + indent + '}'
    raise TypeError(f'Unsupported JSON value type: {type(value).__name__}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_file', nargs='?', type=Path,
                        default=ROOT / 'systems' / 'BW1100.json')
    parser.add_argument('--check', action='store_true',
                        help='exit 1 if the file is not already formatted')
    parser.add_argument('--stdout', action='store_true',
                        help='write the formatted content to stdout')
    parser.add_argument('--no-space-before-close', action='store_true',
                        help='use conventional [x, y] instead of [x, y ]')
    args = parser.parse_args()

    if args.check and args.stdout:
        parser.error('--check and --stdout cannot be used together')
    source = args.json_file.read_text(encoding='utf-8')
    data = json.loads(source)
    formatted = render(data, space_before_close=not args.no_space_before_close) + '\n'

    if args.stdout:
        print(formatted, end='')
        return
    if args.check:
        if source != formatted:
            print(f'needs formatting: {args.json_file}')
            raise SystemExit(1)
        print(f'already formatted: {args.json_file}')
        return

    temporary = args.json_file.with_suffix(args.json_file.suffix + '.tmp')
    temporary.write_text(formatted, encoding='utf-8')
    os.replace(temporary, args.json_file)
    print(f'formatted: {args.json_file}')


if __name__ == '__main__':
    main()
