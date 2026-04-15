#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check if upstream kernel commits exist in a target branch."""

import argparse
import os
import re
import subprocess
import sys
from enum import Enum, auto


class Status(Enum):
    """Commit status: EXISTS (found), REVIEW (diff differs), ABSENT (missing)."""
    EXISTS = auto()
    REVIEW = auto()
    ABSENT = auto()


_cfg = {'upstream': 'upstream/master', 'merge_base': None, 'branch': None}
_branch_log = []


CHERRYPICK_PATTERNS = [
    r'Cherry[- ]?picked from commit:?\s*{sha}',
    r'Backported from commit:?\s*{sha}',
    r'commit\s+{sha}[0-9a-f]*\s+upstream',
    r'Upstream commit\s+{sha}',
]


def git(*args):
    """Run git command, return (returncode, stdout)."""
    result = subprocess.run(['git'] + list(args), capture_output=True, text=True, check=False)
    return result.returncode, result.stdout.strip()


def git_ok(*args):
    """Run git command, return stdout (empty string on failure)."""
    rc, out = git(*args)
    return out if rc == 0 else ''


def is_ancestor(sha, ref):
    """Check if sha is an ancestor of ref."""
    rc, _ = git('merge-base', '--is-ancestor', sha, ref)
    return rc == 0


def get_body(sha):
    """Get full commit message. Returns None on failure."""
    rc, out = git('show', '--format=%B', '-s', sha)
    return out if rc == 0 else None


def short_sha(sha, length=12):
    """Get abbreviated SHA, or dashes if None."""
    return sha[:length] if sha else '-' * length


def init_branch_log():
    """Cache git log --pretty=oneline for branch."""
    ref = _cfg['branch']
    base = _cfg['merge_base']
    range_spec = f'{base}..{ref}' if base else ref

    rc, out = git('log', range_spec, '--pretty=oneline')
    if rc != 0:
        return

    for line in out.split('\n'):
        if ' ' in line:
            sha, title = line.split(' ', 1)
            _branch_log.append((sha, title))


def search_branch_log(pattern):
    """Search cached branch log for pattern.
    
    Returns (sha, title) or (None, None).
    """
    for sha, title in _branch_log:
        if pattern in title:
            return sha, title
    return None, None


def log_grep(ref, pattern):
    """Search git log with --grep, using merge_base..ref range.
    
    Returns: list of (sha, title) tuples
    """
    base = _cfg['merge_base']
    range_spec = f'{base}..{ref}' if base else ref
    rc, out = git('log', range_spec, '--pretty=oneline', '--grep', pattern, '--fixed-strings')
    if rc != 0:
        return []
    result = []
    for line in out.split('\n'):
        if line and ' ' in line:
            sha, title = line.split(' ', 1)
            result.append((sha, title))
    return result


def extract_marker_sha(body):
    """Extract SHA from any cherry-pick marker in body."""
    if not body:
        return None
    for pattern in CHERRYPICK_PATTERNS:
        regex = pattern.format(sha=r'([a-f0-9]{7,40})')
        if match := re.search(regex, body, re.I):
            return match.group(1)
    return None


def find_cherrypick(title):
    """Find cherry-pick of upstream SHA in branch.

    Returns: (local_sha, marker_sha) or (None, None)
    marker_sha is the SHA in the cherry-pick marker.
    """
    local_sha, _ = search_branch_log(title)
    if not local_sha:
        return None, None

    local_body = get_body(local_sha)
    marker_sha = extract_marker_sha(local_body)
    return local_sha, marker_sha


def check_series(upstream_body):
    """Return ' SERIES (missing...)' if series has missing members, else ''."""
    if not upstream_body:
        return ''
    match = re.search(r'Link:\s*https?://[^/]+/(?:r/)?(\d+\.\d+)-\d+-', upstream_body)
    if not match:
        return ''

    series = log_grep(_cfg['upstream'], match.group(1))
    if len(series) <= 1:
        return ''

    missing = []
    for sha, title in series:
        local_sha, _ = search_branch_log(title)
        if not local_sha:
            missing.append(sha[:12])
    return f" SERIES ({', '.join(missing)})" if missing else ''


def compare_diffs(sha1, sha2):
    """Compare diffs, ignoring whitespace differences."""
    def normalize(output):
        return [' '.join(line.split()) for line in output.split('\n')
                if re.match(r'^[+-][^+-]', line)]
    rc1, diff1 = git('show', '--format=', sha1)
    rc2, diff2 = git('show', '--format=', sha2)
    if rc1 != 0 or rc2 != 0:
        return False
    return normalize(diff1) == normalize(diff2)


def process_commit(sha, title):
    """Process commit, return (status, local_sha, series_tag, note)."""
    upstream_body = get_body(sha)

    local_sha, marker_sha = find_cherrypick(title)

    if not local_sha:
        return Status.ABSENT, None, '', ''

    diffs_match = compare_diffs(sha, local_sha)
    series_tag = check_series(upstream_body)

    if marker_sha and marker_sha.startswith(sha[:12]):
        note = ''
    elif marker_sha:
        note = 'SHA not in upstream' if not is_ancestor(marker_sha, _cfg['upstream']) else 'different SHA marker'
    else:
        note = 'no SHA marker'

    if diffs_match:
        return Status.EXISTS, local_sha, series_tag, note
    return Status.REVIEW, local_sha, series_tag, note


def format_status(status, local_sha, series_tag, note):
    """Format status string for output."""
    sha = short_sha(local_sha)
    note_str = f', {note}' if note else ''

    if status == Status.EXISTS:
        return f'EXISTS ({sha}{note_str}){series_tag}'
    if status == Status.REVIEW:
        return f'REVIEW ({sha}{note_str}){series_tag}'
    return 'ABSENT'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Check if upstream kernel commits exist in a target branch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Status meanings:
  EXISTS   Commit found in branch (diffs match)
  REVIEW   Commit found but diff differs - needs manual review
  ABSENT   Commit not in branch (needs backporting)
  SERIES   Commit is part of a series with missing members

Notes in parentheses:
  (sha)                         Properly marked cherry-pick
  (sha, no SHA marker)          Found by title, no SHA marker
  (sha, different SHA marker)   Marker references different upstream SHA
  (sha, SHA not in upstream)    Marker references SHA not in configured upstream

Examples:
  %(prog)s sha_list.txt
  %(prog)s list1.txt list2.txt -b my-branch
''')
    parser.add_argument('sha_files', nargs='*', help='Files containing SHA lists')
    parser.add_argument('-b', '--branch', help='Target branch (default: current)')
    parser.add_argument('-u', '--upstream', default='upstream/master',
                        help='Upstream ref (default: upstream/master)')
    return parser


def read_sha_files(sha_files):
    """Read SHA files and return list of lines."""
    lines = []
    for sha_file in sha_files:
        with open(sha_file, encoding='utf-8') as f:
            lines += [line for line in f if line.strip() and not line.startswith('#')]
    return lines


def setup_config(args):
    """Set up configuration. Returns (branch, error_msg) - error_msg is None on success."""
    if not os.path.isdir('.git'):
        return None, 'Error: Not in a git repository'

    _, branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    branch = args.branch or branch
    remote = args.upstream.split('/')[0] if '/' in args.upstream else args.upstream

    _, remotes = git('remote')
    if remote not in remotes.split('\n'):
        return None, f"Error: Remote '{remote}' not found"

    _cfg['branch'] = branch
    _cfg['upstream'] = args.upstream

    rc, merge_base = git('merge-base', branch, args.upstream)
    if rc != 0:
        return None, f"Error: No common ancestor between '{branch}' and '{args.upstream}'"
    _cfg['merge_base'] = merge_base

    return branch, None


def print_summary(exists, review, absent, series, total):
    """Print summary and detailed lists."""
    print('\nSummary')
    print('_______')
    for label, items in [('EXISTS', exists), ('REVIEW', review), ('ABSENT', absent), ('SERIES', series)]:
        print(f'{label:<7}: {len(items)}')
    print(f'Total  : {total}')

    if review:
        print(f'\nREVIEW ({len(review)}):')
        for sha, (local_sha, title) in review.items():
            print(f'  {sha[:12]} {title[:60]:<61} ({local_sha[:12]} differs)')

    if absent:
        print(f'\nABSENT ({len(absent)}):')
        for sha, title in absent:
            print(f'  {sha[:12]} {title}')

    if series:
        print(f'\nSERIES ({len(series)}):')
        for sha, title, missing in series:
            print(f'  {sha[:12]} {title}')
            print(f'    Missing: {missing}')


def main():
    """Main entry point."""
    parser = parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()
    if not args.sha_files:
        parser.print_help()
        return 1

    lines = read_sha_files(args.sha_files)

    branch, error = setup_config(args)
    if error:
        print(error, file=sys.stderr)
        return 1

    init_branch_log()

    exists, absent, series = [], [], []
    review = {}

    for line in lines:
        sha, title = line.split(' ', 1)
        title = title.strip()
        print(f'{sha[:12]} {title[:60]:<61}', end='', flush=True)

        if is_ancestor(sha, branch):
            print('EXISTS')
            exists.append((sha, sha))
            continue

        status, local_sha, series_tag, note = process_commit(sha, title)
        print(format_status(status, local_sha, series_tag, note))

        if status == Status.EXISTS:
            exists.append((sha, local_sha or sha))
            if series_tag:
                series.append((sha, title, series_tag.strip()))
        elif status == Status.REVIEW:
            review[sha] = (local_sha, title)
        else:
            absent.append((sha, title))

    print_summary(exists, review, absent, series, len(lines))
    return len(review) + len(absent)


if __name__ == '__main__':
    sys.exit(main())
