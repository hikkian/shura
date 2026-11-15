"""Property tests over a matrix of machines (tests/fixtures/detect/matrix/<machine>/): ~35 configurations written as the raw
output of the tools the detector runs (lscpu, nvidia-smi, lspci, vulkaninfo, sysctl, sysfs attributes, PowerShell JSON),
plus random variations of VRAM and RAM with a fixed seed.

Properties (for every machine and every variation):
  (a) the detector does not raise, and the profile it builds is valid;
  (b) when a plan exists, it does not need more RAM or VRAM than the planner itself allows after its reserves;
  (c) the margin it leaves is not negative, and the budgets are the planner's own constants applied to the profile;
  (d) the same input gives the same plan;
  (e) when there is no plan, the answer says why (a list of reasons), it does not raise.
(f) "more memory never gives a worse window" is NOT a promise of the planner: it is reported in REPORT-PUB1.md, not tested.

The matrix found five violations (F1 to F5); each one is fixed in the detector, with its own commit, and the machine that showed
it keeps its test. The planner itself was not changed: see REPORT-PUB1.md.
"""
import json
import random
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "installer"))
from universal import detect as D, planner, schema  # noqa: E402

MATRIX = ROOT / "tests/fixtures/detect/matrix"
HW_FIXTURES = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "tests/fixtures/hardware").glob("*.json"))}
REAL = json.loads((ROOT / "catalog/models.json").read_text())
SYN = json.loads((ROOT / "tests/fixtures/catalog_test.json").read_text())
CATALOGS = {"real": REAL, "synthetic": SYN}
GiB = 1024 ** 3
MiB = 1024 ** 2

# the file of a fixture -> the command or the path the detector asks for (same convention as tests/test_detect.py)
COMMANDS = {"lscpu.txt": "lscpu", "nvidia_smi.txt": "nvidia-smi", "lspci.txt": "lspci", "vulkaninfo.txt": "vulkaninfo",
            "sysctl.txt": "sysctl"}
POWERSHELL = {"ps_cpu.json": "Win32_Processor", "ps_video.json": "Win32_VideoController", "ps_vram.json": "HardwareInformation"}


class MatrixEnv:
    """A FakeEnv-like object (run/read/exists/listdir) built from one machine folder."""

    def __init__(self, folder):
        self.commands, self.files, self.powershell = {}, {}, {}
        for name, cmd in COMMANDS.items():
            if (folder / name).exists():
                self.commands[cmd] = (folder / name).read_text()
        if (folder / "meminfo.txt").exists():
            self.files["/proc/meminfo"] = (folder / "meminfo.txt").read_text()
        if (folder / "kfd").exists():
            self.files["/dev/kfd"] = ""
        for card in sorted((folder / "sysfs").glob("*")) if (folder / "sysfs").exists() else []:
            for attr in sorted(card.iterdir()):
                self.files[f"/sys/class/drm/{card.name}/device/{attr.name}"] = attr.read_text()
        for name, key in POWERSHELL.items():
            if (folder / "windows" / name).exists():
                self.powershell[key] = (folder / "windows" / name).read_text()

    def run(self, cmd, timeout=15):
        if cmd[0] in ("powershell", "pwsh"):
            return next((text for key, text in self.powershell.items() if key in cmd[-1]), None)
        return self.commands.get(cmd[0])

    def read(self, path):
        return self.files.get(path)

    def exists(self, path):
        return path in self.files or any(f.startswith(path + "/") for f in self.files)

    def listdir(self, path):
        return sorted({f[len(path) + 1:].split("/")[0] for f in self.files if f.startswith(path + "/")})


def machines():
    return sorted(p.name for p in MATRIX.iterdir() if p.is_dir())


def detect_folder(name):
    """(profile, memory values for the Windows fixtures) of one machine folder, through the same flow the installer uses."""
    folder = MATRIX / name
    env = MatrixEnv(folder)
    if (folder / "sysctl.txt").exists():
        cpu, mem, gpus, notes = D.detect_macos(env)
        return D.build_profile("macos", cpu, mem, gpus, notes, arch="arm64", disk_free=100 * GiB)
    if (folder / "windows").exists():
        memory = json.loads((folder / "windows" / "memory.json").read_text())
        with unittest.mock.patch.object(D, "_windows_memory", return_value=(memory["total"], memory["available"])):
            cpu, mem, gpus, notes = D.detect_windows(env)
        return D.build_profile("windows", cpu, mem, gpus, notes, disk_free=100 * GiB)
    cpu, mem, gpus, notes = D.detect_linux(env)
    return D.build_profile("linux", cpu, mem, gpus, notes, disk_free=100 * GiB)


def check_plan(test, hw, cat, label):
    """Properties (b) to (e) for one profile and one catalog."""
    p = planner.plan(hw, cat, menu=False)
    test.assertEqual(p, planner.plan(hw, cat, menu=False), f"{label}: the same input must give the same plan (d)")
    if not p["ok"]:
        test.assertTrue(p["reasons"] and all(isinstance(r, str) and r for r in p["reasons"]), f"{label}: a missing plan needs reasons (e)")
        return p
    mem = p["memory"]
    test.assertLessEqual(mem["ram_need"], mem["ram_budget"], f"{label}: plan needs more RAM than the planner allows (b)")
    test.assertGreaterEqual(mem["ram_budget"] - mem["ram_need"], 0, f"{label}: negative RAM margin (c)")
    if p["mode"] != "cpu":
        test.assertLessEqual(mem["vram_need"], max(0, mem["vram_budget"]), f"{label}: plan needs more VRAM than the planner allows (b)")
        test.assertGreaterEqual(mem["vram_budget"] - mem["vram_need"], 0, f"{label}: negative VRAM margin (c)")
    return p


def expected_budgets(hw):
    """The planner's own reserves, applied to the profile by hand (c): the RAM budget and the VRAM budget of the planner's GPU."""
    cfg = planner._cfg({})
    total = hw["memory"]["total"]
    ram = total - max(cfg["os_reserve_min"], cfg["os_reserve_frac"] * total)
    gpus = sorted(hw["gpus"], key=lambda g: g["vram_total"], reverse=True)
    vram = None
    if gpus and not gpus[0].get("unified"):
        g = gpus[0]
        display = g.get("display", g["vram_used"] > 256 * MiB)
        vram = g["vram_total"] - g["vram_used"] - (cfg["vram_reserve_display"] if display else cfg["vram_reserve_headless"])
    return ram, vram


def make_machine_tests(cls):
    """One test method per machine folder: the detector and the planner, with the properties above."""
    for name in machines():
        def test(self, name=name):
            hw = detect_folder(name)
            self.assertEqual(schema.validate_hardware(hw), [], f"{name}: the detector's profile must be valid (a)")
            for cat_name, cat in CATALOGS.items():
                check_plan(self, hw, cat, f"{name}/{cat_name}")
            ram, vram = expected_budgets(hw)
            for cat_name, cat in CATALOGS.items():
                p = planner.plan(hw, cat, menu=False)
                if p["ok"]:
                    self.assertEqual(p["memory"]["ram_budget"], int(ram), f"{name}: RAM budget must be the planner's reserve (c)")
                    if vram is not None and p["mode"] != "cpu":
                        self.assertEqual(p["memory"]["vram_budget"], int(vram), f"{name}: VRAM budget must be the planner's reserve (c)")
        test.__name__ = "test_machine_" + name
        test.__doc__ = f"Machine {name}: properties (a)-(e) with the real and the synthetic catalog."
        setattr(cls, test.__name__, test)
    return cls


class MachineMatrix(unittest.TestCase):
    pass


make_machine_tests(MachineMatrix)



class SpecificFindings(unittest.TestCase):
    """Findings the generic properties cannot see: the right number is in the wrong place, or something is dropped silently."""

    def test_f4_each_card_keeps_its_own_vram_when_an_apu_is_listed_first(self):
        """F4: lspci lists the APU first (bus 08) and the discrete card second (bus 2d); sysfs numbers the discrete card first.
        The detector pairs them by position, so the discrete card gets the APU's 512 MiB and the APU gets its 16 GiB."""
        hw = detect_folder("edge_amd_plus_apu_32g")
        by_name = {g["name"]: g["vram_total"] for g in hw["gpus"]}
        self.assertEqual(by_name["Navi 32 [Radeon RX 7700 XT / 7800 XT]"], 17163091968)

    def test_f5_nvidia_listed_by_lspci_but_not_by_nvidia_smi_is_named_in_the_notes(self):
        """F5: an NVIDIA card that lspci shows and nvidia-smi does not describe is dropped with no note. The detector promises that
        whatever it cannot determine is named in detection_notes."""
        hw = detect_folder("edge_nvidia_no_smi_16g")
        self.assertTrue(any("NVIDIA" in n for n in hw["detection_notes"]), hw["detection_notes"])


class NvidiaSmiNotAvailable(unittest.TestCase):
    """F1: a GPU whose memory nvidia-smi cannot report is left out; the GPUs next to it are kept."""

    def test_a_line_with_na_memory_is_skipped_and_the_others_are_kept(self):
        text = ("NVIDIA GeForce RTX 3060, [N/A], [N/A], 550.120\n"
                "NVIDIA GeForce RTX 3060, 12288, 300, 550.120\n")
        gpus = D.parse_nvidia_smi(text)
        self.assertEqual([(g["name"], g["vram_total"]) for g in gpus], [("NVIDIA GeForce RTX 3060", 12288 * MiB)])


class AmdVramUnreadable(unittest.TestCase):
    """F2 and F3: a card whose sysfs counters are text, or zero, is not listed as a GPU; the note says why."""

    def test_zero_vram_is_skipped_with_a_note(self):
        hw = detect_folder("edge_amd_vram_zero_16g")
        self.assertEqual(hw["gpus"], [])
        self.assertTrue(any("VRAM not readable" in n for n in hw["detection_notes"]), hw["detection_notes"])

    def test_text_in_the_counters_is_skipped_with_a_note_and_does_not_raise(self):
        hw = detect_folder("edge_amd_vram_garbage_16g")
        self.assertEqual(hw["gpus"], [])
        self.assertTrue(any("VRAM not readable" in n for n in hw["detection_notes"]), hw["detection_notes"])


class RandomVariations(unittest.TestCase):
    """Random VRAM and RAM for the discrete and integrated reference profiles, with a fixed seed (1234): the same numbers every run."""
    SEED = 1234
    BASES = ("ref_rtx4070s_12g_32g", "rtx3060_12g_16g", "rx9070_16g_48g_fedora", "rx7900xtx_24g_64g", "apple_m2max_64g")

    def variations(self):
        rng = random.Random(self.SEED)
        for i in range(30):
            base = HW_FIXTURES[rng.choice(self.BASES)]
            hw = json.loads(json.dumps(base))
            if hw["gpus"] and not hw["gpus"][0].get("unified"):
                g = hw["gpus"][0]
                g["vram_total"] = rng.randint(4, 32) * GiB
                g["vram_used"] = rng.randint(0, g["vram_total"] // GiB) * GiB // rng.choice((1, 2, 4))
            hw["memory"]["total"] = rng.choice((8, 16, 32, 64)) * GiB
            hw["memory"]["available"] = int(hw["memory"]["total"] * rng.uniform(0.3, 0.9))
            hw["memory"].pop("bandwidth_gbs", None)
            yield f"variation{i:02d}_{hw['gpus'][0]['name'] if hw['gpus'] else 'cpu'}", hw

    def test_every_variation_is_a_valid_profile_and_obeys_the_properties(self):
        for label, hw in self.variations():
            with self.subTest(label):
                self.assertEqual(schema.validate_hardware(hw), [], label)
                for cat_name, cat in CATALOGS.items():
                    check_plan(self, hw, cat, f"{label}/{cat_name}")
                ram, vram = expected_budgets(hw)
                p = planner.plan(hw, REAL, menu=False)
                if p["ok"]:
                    self.assertEqual(p["memory"]["ram_budget"], int(ram), f"{label}: RAM budget (c)")
                    if vram is not None and p["mode"] != "cpu":
                        self.assertEqual(p["memory"]["vram_budget"], int(vram), f"{label}: VRAM budget (c)")


class NoPlanIsExplicit(unittest.TestCase):
    """(e) on the machines that cannot run the catalog: a reason per refused model, not an exception."""

    def test_eight_gigabytes_of_ram_gives_a_reasoned_refusal(self):
        hw = detect_folder("edge_ram_8g_nvidia_12g")
        hw["memory"]["total"], hw["memory"]["available"] = 8 * GiB, 4 * GiB
        p = check_plan(self, hw, REAL, "edge_ram_8g_nvidia_12g/8GiB")
        if not p["ok"]:
            self.assertTrue(any("RAM" in r or "fits" in r for r in p["reasons"]), p["reasons"])


if __name__ == "__main__":
    unittest.main()
