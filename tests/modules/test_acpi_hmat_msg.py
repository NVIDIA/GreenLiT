# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit contracts for read-only HMAT compiled-callsite observation."""

from itertools import combinations
from unittest.mock import call

from common import glt_core as GLTCore
from modules.acpi import glt_test_acpi_msg as acpi_test
from tests.test_base import GLTTestBase


PROC_CONTROL = "/proc/dynamic_debug/control"
DEBUGFS_CONTROL = "/sys/kernel/debug/dynamic_debug/control"
HMAT_SOURCE = "drivers/acpi/numa/hmat.c"
TARGETS = (
    ("hmat_parse_locality", "Locality: Flags:00 Type:Access Latency"),
    ("hmat_parse_locality", "  Initiator-Target[0-1]:100ns"),
    ("hmat_parse_cache", "Cache: Domain:0 Size:1024"),
    ("hmat_parse_proximity_domain", "Memory (0x1000 length 0x1000) Flags:0001"),
)
PREFIXES = ("Locality: Flags:", "  Initiator-Target[", "Cache: Domain:", "Memory (")
NON_TARGETS = (
    ("hmat_parse_proximity_domain", "Memory Domain missing"),
    ("hmat_parse_proximity_domain", "Processor Domain missing"),
)
CATALOG_HEADER = "# filename:lineno [module]function flags format"
FIXED_CATALOG = CATALOG_HEADER + "\n" + (
    'drivers/pci/pci.c:1234 [pci]pci_bus_read_config_byte =_ "not hmat\\n"\n'
    'drivers/acpi/numa/hmat.c:467 [hmat]hmat_parse_locality =_ "  Initiator-Target[%u-%u]:%u%s\\n"\n'
    'drivers/acpi/numa/hmat.c:454 [hmat]hmat_parse_locality =p "Locality: Flags:%02x Type:%s '
    'Initiator Domains:%u Target Domains:%u Base:%lld\\n"\n'
    'drivers/acpi/numa/hmat.c:497 [hmat]hmat_parse_cache =_ "Cache: Domain:%u Size:%llu '
    'Attrs:%08x SMBIOS Handles:%d\\n"\n'
    'drivers/acpi/numa/hmat.c:582 [hmat]hmat_parse_proximity_domain =_ "Invalid Processor Domain\\n"\n'
    'drivers/acpi/numa/hmat.c:574 [hmat]hmat_parse_proximity_domain =_ "Memory Domain missing from SRAT\\n"\n'
    'drivers/acpi/numa/hmat.c:563 [hmat]hmat_parse_proximity_domain =_ "Memory (%#llx length '
    '%#llx) Flags:%04x Processor Domain:%u Memory Domain:%u\\n"\n'
)
UNFIXED_CATALOG = CATALOG_HEADER + "\n" + (
    'drivers/acpi/numa/hmat.c:568 [hmat]hmat_parse_proximity_domain =_ "Invalid Processor Domain\\n"\n'
    'drivers/acpi/numa/hmat.c:560 [hmat]hmat_parse_proximity_domain =_ "Memory Domain missing from SRAT\\n"\n'
)
PARTIAL_CATALOG = "\n".join(row for row in FIXED_CATALOG.split("\n") if "Cache:" not in row)
REFERENCE_COMMIT = "c8e88de1b44e58cacdef41ea9aaa78fca35f1357"
LEGACY_DEBUG = "\n".join(f"kernel: acpi/hmat: {text}" for _, text in TARGETS)


def catalog_record(function, text, source=HMAT_SOURCE, flags="=_", suffix=""):
    """Build the public dynamic-debug descriptor layout, without escaping text."""
    return f'{source}:454 [acpi]{function} {flags} "{text}"{suffix}'


class TestAcpiHmatMsg(GLTTestBase):
    """Compiled registrations provide evidence independently of emitted logs."""

    def setUp(self):
        super().setUp()
        # Honest legacy seam: benign logs let expected warning/skip outcomes fail.
        log_patch = self.mock.patch.object(
            acpi_test.glt_utils, "get_kernel_log", return_value="ACPI: HMAT discovered\n",
        )
        self.get_log = log_patch.start()
        self.addCleanup(log_patch.stop)
        check_patch = self.mock.patch.object(
            acpi_test.glt_utils, "check_kernel_log", return_value=False,
        )
        self.check_log = check_patch.start()
        self.addCleanup(check_patch.stop)

    def test_complete_inventory_ignores_descriptor_flags(self):
        """C01: enabled debug output cannot invalidate compiled registrations."""
        self.get_log.return_value = LEGACY_DEBUG
        for flags in ("=_", "=p", "=pmflt", "=Z9"):
            with self.subTest(flags=flags):
                snapshot = "\n".join(
                    catalog_record(function, text, flags=flags,
                                   source="/build/linux/" + HMAT_SOURCE).replace(
                                       ":454 [acpi]", f":{900 + index} [kernel]", 1)
                    for index, (function, text) in reversed(list(enumerate(TARGETS)))
                )
                with self.mock.patch.object(
                    acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
                ) as read:
                    acpi_test.test_acpi_hmat_msg(None, None)
                read.assert_called_once_with(PROC_CONTROL)
                self.get_log.assert_not_called()
                self.check_log.assert_not_called()

    def test_missing_zero_to_three_targets_fails_metadata_contract(self):
        """D01/A03: every required pair must exist, without claiming patch ancestry."""
        for count in range(4):
            for present in combinations(range(4), count):
                with self.subTest(present=present):
                    entries = [TARGETS[index] for index in present] + list(NON_TARGETS)
                    snapshot = "\n".join(catalog_record(*entry) for entry in entries)
                    with self.mock.patch.object(
                        acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
                    ):
                        with self.assertRaises(GLTCore.GLTTestFail) as context:
                            acpi_test.test_acpi_hmat_msg(None, None)
                    message = str(context.exception)
                    for index, (function, _) in enumerate(TARGETS):
                        if index not in present:
                            self.assertIn(f"{function}(): {PREFIXES[index]!r}", message)
                    self.assertRegex(message.lower(), r"source|chang|renam")
                    self.assertIn(REFERENCE_COMMIT, message)
                    self.assertNotIn("Include kernel commit", message)
                    self.assertNotIn("to fix", message)

    def test_renamed_functions_or_formats_fail_metadata_contract(self):
        """D01/A03: renamed metadata fails the explicitly exact identity contract."""
        for change in ("function", "format"):
            for selected in ((0,), (1,), (2,), (3,), (0, 1, 2, 3)):
                with self.subTest(change=change, selected=selected):
                    entries = []
                    for index, (function, text) in enumerate(TARGETS):
                        if index in selected:
                            if change == "function":
                                function += "_renamed"
                            else:
                                text = "Renamed " + text
                        entries.append(catalog_record(function, text))
                    with self.mock.patch.object(
                        acpi_test.glt_utils, "read_sysfs_value", return_value="\n".join(entries),
                    ):
                        with self.assertRaises(GLTCore.GLTTestFail):
                            acpi_test.test_acpi_hmat_msg(None, None)

    def test_both_catalog_paths_absent_skips(self):
        """C03: unavailable instrumentation is an applicability limit."""
        with self.mock.patch.object(
            acpi_test.glt_utils, "read_sysfs_value", side_effect=FileNotFoundError("absent"),
        ) as read:
            with self.assertRaises(GLTCore.GLTTestSkip):
                acpi_test.test_acpi_hmat_msg(None, None)
        self.assertEqual(read.call_args_list, [call(PROC_CONTROL), call(DEBUGFS_CONTROL)])

    def test_absent_proc_catalog_uses_debugfs(self):
        """C03: only an absent primary interface permits the alternate path."""
        self.get_log.return_value = LEGACY_DEBUG
        snapshot = "\n".join(catalog_record(*entry) for entry in TARGETS)
        with self.mock.patch.object(
            acpi_test.glt_utils, "read_sysfs_value",
            side_effect=[FileNotFoundError("absent"), snapshot],
        ) as read:
            acpi_test.test_acpi_hmat_msg(None, None)
        self.assertEqual(read.call_args_list, [call(PROC_CONTROL), call(DEBUGFS_CONTROL)])
        self.get_log.assert_not_called()
        self.check_log.assert_not_called()

    def test_io_errors_fail_as_collector_errors_without_fallback(self):
        """C03: failed observation must not be reported as a missing patch."""
        for error in (PermissionError("denied"), OSError("synthetic I/O error")):
            for alternate in (False, True):
                with self.subTest(error=type(error).__name__, alternate=alternate):
                    effects = [FileNotFoundError("absent"), error] if alternate else [error]
                    with self.mock.patch.object(
                        acpi_test.glt_utils, "read_sysfs_value", side_effect=effects,
                    ) as read:
                        with self.assertRaises(GLTCore.GLTTestFail) as context:
                            acpi_test.test_acpi_hmat_msg(None, None)
                    message = str(context.exception)
                    self.assertIn("collector", message.lower())
                    self.assertIn(str(error), message)
                    self.assertNotIn("Include kernel commit", message)
                    attempts = [call(PROC_CONTROL)]
                    if alternate:
                        attempts.append(call(DEBUGFS_CONTROL))
                    self.assertEqual(read.call_args_list, attempts)

    def test_unicode_errors_warn_without_fallback(self):
        """C03: strict decoding loss makes the whole observation incomplete."""
        for alternate in (False, True):
            with self.subTest(alternate=alternate):
                error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
                effects = [FileNotFoundError("absent"), error] if alternate else [error]
                with self.mock.patch.object(
                    acpi_test.glt_utils, "read_sysfs_value", side_effect=effects,
                ) as read:
                    with self.assertRaises(GLTCore.GLTTestWarn):
                        acpi_test.test_acpi_hmat_msg(None, None)
                attempts = [call(PROC_CONTROL)]
                if alternate:
                    attempts.append(call(DEBUGFS_CONTROL))
                self.assertEqual(read.call_args_list, attempts)

    def test_unavailable_or_unrecognizable_snapshot_warns(self):
        """C04: absent usable records are incomplete observation, not patch proof."""
        for snapshot in ("", " \t\n ", "# filename:lineno [module]function flags format",
                         "not a descriptor", 'other.c:1 [x]fn =_ "unterminated'):
            with self.subTest(snapshot=snapshot), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
            ):
                with self.assertRaises(GLTCore.GLTTestWarn):
                    acpi_test.test_acpi_hmat_msg(None, None)

    def test_valid_non_hmat_records_skip_despite_unrelated_noise(self):
        """C04: unrelated malformed text must not supply HMAT applicability."""
        unrelated = catalog_record("other_fn", "hello", source="drivers/other.c")
        for noise in ("", "not a descriptor", 'other.c:1 [x]fn =_ "broken',
                      'not a descriptor mentioning drivers/acpi/numa/hmat.c:45'):
            with self.subTest(noise=noise), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=unrelated + "\n" + noise,
            ):
                with self.assertRaises(GLTCore.GLTTestSkip):
                    acpi_test.test_acpi_hmat_msg(None, None)

    def test_malformed_strict_hmat_rows_warn(self):
        """C04: recognized source with broken metadata makes observation incomplete."""
        rows = (
            HMAT_SOURCE + ':45 [acpi]hmat_parse_cache =_ "Cache: missing delimiter',
            HMAT_SOURCE + ':45 missing fields',
            HMAT_SOURCE + ':45 [acpi]hmat_parse_cache _ "Cache: bad flags"',
            HMAT_SOURCE + ':45 [acpi]hmat_parse_cache =_ "Cache: suffix" invalid',
        )
        for row in rows:
            with self.subTest(row=row), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=row,
            ):
                with self.assertRaises(GLTCore.GLTTestWarn):
                    acpi_test.test_acpi_hmat_msg(None, None)

    def test_complete_snapshot_with_malformed_hmat_warns_at_every_position(self):
        """D03: malformed HMAT metadata disqualifies even a complete target set."""
        self.get_log.return_value = LEGACY_DEBUG
        rows = [catalog_record(*entry) for entry in TARGETS]
        noise = (HMAT_SOURCE + ':45 [acpi]broken =_ "unterminated\n'
                 'unparseable noise\n' + catalog_record("other", "noise", source="elsewhere.c"))
        for index in range(len(rows) + 1):
            snapshot = "\n".join(rows[:index] + [noise] + rows[index:])
            with self.subTest(index=index), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
            ):
                with self.assertRaises(GLTCore.GLTTestWarn):
                    acpi_test.test_acpi_hmat_msg(None, None)
        self.get_log.assert_not_called()
        self.check_log.assert_not_called()

    def test_raw_quotes_backslashes_and_class_suffixes_are_valid(self):
        """C01/C04: catalog formats are raw greedy quoted text, not C strings."""
        self.get_log.return_value = LEGACY_DEBUG
        for suffix in ("", " class:memory", " class unknown, _id:7",
                       " class unknown, _id:-7"):
            snapshot = "\n".join(
                catalog_record(function, text + r' "raw quote" \040 \\ tail', suffix=suffix)
                for function, text in TARGETS
            )
            with self.subTest(suffix=suffix), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
            ):
                acpi_test.test_acpi_hmat_msg(None, None)
        self.get_log.assert_not_called()
        self.check_log.assert_not_called()

    def test_literal_octal_does_not_create_a_target_prefix(self):
        """C04: literal escapes cannot be decoded into positive registration proof."""
        rows = [catalog_record(*entry) for entry in TARGETS[1:]]
        rows.append(catalog_record("hmat_parse_locality", r"\114ocality: escaped spelling"))
        with self.mock.patch.object(
            acpi_test.glt_utils, "read_sysfs_value", return_value="\n".join(rows),
        ):
            with self.assertRaises(GLTCore.GLTTestFail):
                acpi_test.test_acpi_hmat_msg(None, None)

    def test_vertical_tab_in_non_hmat_format_stays_in_its_record(self):
        """C04: real vertical tabs are format content; only newlines split rows."""
        snapshot = catalog_record("other", "before\x0bafter", source="drivers/other.c")
        with self.mock.patch.object(
            acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
        ):
            with self.assertRaises(GLTCore.GLTTestSkip):
                acpi_test.test_acpi_hmat_msg(None, None)

    def test_interior_separators_preserve_complete_target_records(self):
        """C04: raw separators after target prefixes must not destroy registrations."""
        self.get_log.return_value = LEGACY_DEBUG
        for separator in ("\x0b", "\x0c", "\u2028", "\u2029", "\x85"):
            snapshot = "\n".join(
                catalog_record(function, text + separator + "tail")
                for function, text in TARGETS
            )
            with self.subTest(separator=repr(separator)), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
            ):
                acpi_test.test_acpi_hmat_msg(None, None)

    def test_invalid_source_tokens_cannot_supply_hmat_identity(self):
        """C04/C05: source identity requires the first full source token boundary."""
        spoof_rows = (
            catalog_record(*TARGETS[0], source="xdrivers/acpi/numa/hmat.c"),
            catalog_record(*TARGETS[0], source=HMAT_SOURCE + ".backup"),
            HMAT_SOURCE + ':45extra [acpi]hmat_parse_locality =_ "Locality: spoof"',
            HMAT_SOURCE + ':4.5 [acpi]hmat_parse_locality =_ "Locality: spoof"',
            'prefix ' + catalog_record(*TARGETS[0]),
        )
        unrelated = catalog_record("other", "valid", source="drivers/other.c")
        for row in spoof_rows:
            # Boundary lookalikes may still be valid non-HMAT descriptor records.
            valid_non_hmat = row.startswith(("xdrivers/", HMAT_SOURCE + ".backup"))
            for companion in ("", unrelated):
                outcome = GLTCore.GLTTestSkip if companion or valid_non_hmat else GLTCore.GLTTestWarn
                with self.subTest(row=row, companion=bool(companion)), self.mock.patch.object(
                    acpi_test.glt_utils, "read_sysfs_value", return_value=row + "\n" + companion,
                ):
                    with self.assertRaises(outcome):
                        acpi_test.test_acpi_hmat_msg(None, None)

    def test_moved_source_and_quoted_source_lookalikes_skip(self):
        """C05: identical messages outside the exact source cannot prove coverage."""
        for source in ("drivers/acpi/hmat.c", "drivers/acpi/numa/renamed.c", "drivers/other.c"):
            rows = [catalog_record(function, text, source=source) for function, text in TARGETS]
            rows.append(catalog_record("other", HMAT_SOURCE + ':45 "Locality: spoof"',
                                       source="drivers/other.c"))
            with self.subTest(source=source), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value="\n".join(rows),
            ):
                with self.assertRaises(GLTCore.GLTTestSkip):
                    acpi_test.test_acpi_hmat_msg(None, None)

    def test_duplicates_wrong_functions_and_memory_flags_cannot_replace_pairs(self):
        """C05: each distinct source/function/prefix pair needs its own evidence."""
        for omitted in range(4):
            function, text = TARGETS[omitted]
            rows = [catalog_record(*entry) for index, entry in enumerate(TARGETS) if index != omitted]
            rows.extend((catalog_record(*TARGETS[(omitted + 1) % 4]),
                         catalog_record("other_function", text),
                         catalog_record(function, "Memory Flags:0001 Processor Domain:0"),
                         catalog_record(function, "quoted " + text),
                         catalog_record(function, text, source="drivers/other.c")))
            with self.subTest(omitted=omitted), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value="\n".join(rows),
            ):
                with self.assertRaises(GLTCore.GLTTestFail):
                    acpi_test.test_acpi_hmat_msg(None, None)

    def test_revision_and_console_text_cannot_change_catalog_result(self):
        """C06: registration evidence is independent of firmware/log visibility."""
        snapshot = "\n".join(catalog_record(*entry) for entry in TARGETS)
        logs = ("", "kernel: ACPI HMAT revision 1 console_loglevel=4\n" + LEGACY_DEBUG,
                "kernel: ACPI HMAT revision 2 console_loglevel=7\n"
                "kernel: acpi/hmat: Memory Flags:0001 Processor Domain:0\n" + LEGACY_DEBUG)
        for text in logs:
            self.get_log.return_value = text
            with self.subTest(log=text), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
            ):
                acpi_test.test_acpi_hmat_msg(None, None)
        self.get_log.assert_not_called()
        self.check_log.assert_not_called()


class TestAcpiHmatReferenceVerdicts(GLTTestBase):
    """D01/D04: explicit reference conformance and applicability verdicts."""

    def test_complete_duplicates_and_unrelated_rows_preserve_pass(self):
        """D03: accepted unrelated records cannot erase a complete HMAT lifecycle."""
        rows = [catalog_record(*entry) for entry in TARGETS] * 2
        unrelated = catalog_record("other", "valid", source="elsewhere.c") + "\nnoise"
        for index in (0, len(rows) // 2, len(rows)):
            snapshot = "\n".join(rows[:index] + [unrelated] + rows[index:])
            with self.subTest(index=index), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
            ):
                acpi_test.test_acpi_hmat_msg(None, None)

    def test_fixed_catalog_passes(self):
        """A01: all four source-backed registrations satisfy the compiled contract."""
        with self.mock.patch.object(acpi_test.glt_utils, "read_sysfs_value", return_value=FIXED_CATALOG):
            acpi_test.test_acpi_hmat_msg(None, None)

    def test_raw_prefix_spacing_and_source_formats_are_required(self):
        """D04: normalization cannot fabricate the source-backed format identities."""
        variants = ((0, "Locality: changed"), (2, "Cache: changed"))
        variants += tuple((1, " " * count + TARGETS[1][1].lstrip()) for count in (0, 1, 3))
        for changed, text in variants:
            rows = [catalog_record(function, text if index == changed else original)
                    for index, (function, original) in enumerate(TARGETS)]
            with self.subTest(changed=changed, text=text), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value="\n".join(rows),
            ):
                with self.assertRaises(GLTCore.GLTTestFail):
                    acpi_test.test_acpi_hmat_msg(None, None)

    def test_skip_does_not_diagnose_disabled_kernel_configuration(self):
        """D03: source absence alone cannot diagnose a kernel configuration."""
        for effects in ([FileNotFoundError("absent"), FileNotFoundError("absent")],
                        [catalog_record("other", "valid", source="elsewhere.c")]):
            with self.subTest(effects=effects), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", side_effect=effects,
            ):
                with self.assertRaises(GLTCore.GLTTestSkip) as context:
                    acpi_test.test_acpi_hmat_msg(None, None)
            for forbidden in ("CONFIG_", "not enabled", "lacks"):
                self.assertNotIn(forbidden, str(context.exception))

    def test_unfixed_and_partial_catalogs_fail(self):
        """D01/A01: legacy registrations establish applicability, not completeness."""
        for snapshot, absent in ((UNFIXED_CATALOG, range(4)), (PARTIAL_CATALOG, (2,))):
            with self.subTest(snapshot=snapshot), self.mock.patch.object(
                acpi_test.glt_utils, "read_sysfs_value", return_value=snapshot,
            ):
                with self.assertRaises(GLTCore.GLTTestFail) as context:
                    acpi_test.test_acpi_hmat_msg(None, None)
            message = str(context.exception)
            for index, (function, _) in enumerate(TARGETS):
                pair = f"{function}(): {PREFIXES[index]!r}"
                if index in absent:
                    self.assertIn(pair, message)
                else:
                    self.assertNotIn(pair, message)
            self.assertIn(REFERENCE_COMMIT, message)
            self.assertRegex(message.lower(), r"downstream|renam|reword")
            self.assertIn("does not prove", message)
            self.assertNotIn("Include kernel commit", message)
            self.assertNotIn("to fix", message)
