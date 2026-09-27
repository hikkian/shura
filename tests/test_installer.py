"""Planner and GGUF-parser tests. Offline: layouts come from tests/fixtures/layouts.json (real Tiel-Coder
headers, sizes only). Run: python3 -m unittest discover -s tests"""
import json
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "installer"))
import shura_setup as S  # noqa: E402

GB, MB = S.GB, S.MB
LAYOUTS = json.loads((Path(__file__).parent / "fixtures/layouts.json").read_text())


def hw(ram_gb, vram_gb, desktop_vram_mb=700):
    return {"gpus": [{"vram_total": vram_gb * GB, "vram_used": desktop_vram_mb * MB, "cuda_arch": "89"}],
            "ram_total": ram_gb * GB, "physical_cores": 6}


def choose(h, desktop_ram_gb=6, forced=None):
    return S.choose(h, lambda q: LAYOUTS.get(q), desktop_ram_gb * GB, forced)


class ChooseQuant(unittest.TestCase):
    def test_reference_machine_gets_benchmarked_quant(self):
        self.assertEqual(choose(hw(32, 12))[0], "IQ4_XS")

    def test_16gb_ram_with_12gb_gpu_steps_down(self):
        self.assertEqual(choose(hw(16, 12))[0], "IQ3_XXS")

    def test_16gb_ram_with_16gb_gpu_keeps_full_quality(self):
        self.assertEqual(choose(hw(16, 16))[0], "IQ4_XS")

    def test_big_gpu_upgrades_quality(self):
        self.assertEqual(choose(hw(32, 24))[0], "Q4_K_XL")

    def test_too_little_ram_is_refused_with_reason(self):
        quant, placement, why = choose(hw(8, 8))
        self.assertIsNone(quant)
        self.assertIsNone(placement)
        self.assertIn("not enough memory", why)

    def test_no_gpu(self):
        quant, _, why = choose({"gpus": [], "ram_total": 64 * GB, "physical_cores": 8})
        self.assertIsNone(quant)
        self.assertIn("NVIDIA", why)

    def test_forced_quant(self):
        self.assertEqual(choose(hw(32, 12), forced="Q3_K_XL")[0], "Q3_K_XL")


class PlacementInvariants(unittest.TestCase):
    def test_never_exceeds_memory_budgets(self):
        for ram in (16, 24, 32, 64):
            for vram in (8, 10, 12, 16, 24):
                h = hw(ram, vram)
                quant, p, _ = choose(h)
                if not p:
                    continue
                gpu = h["gpus"][0]
                with self.subTest(ram=ram, vram=vram, quant=quant):
                    self.assertLessEqual(p["vram_llama"], gpu["vram_total"] - gpu["vram_used"] - S.VRAM_RESERVE + 1)
                    self.assertLessEqual(p["ram_llama"], ram * GB - 6 * GB + 1)
                    self.assertLessEqual(p["slots"], S.MAX_SLOTS)
                    self.assertGreaterEqual(p["vision_ncmoe"], p["ncmoe"])

    def test_vision_mode_frees_room_for_projector(self):
        layout = LAYOUTS["IQ4_XS"]
        p = S.place(layout, hw(32, 12)["gpus"][0], 32 * GB, 6 * GB)
        slot_cost = sum(layout["experts"][:p["ncmoe"]]) / 256
        freed = (p["slots"] - p["vision_slots"]) * slot_cost + sum(layout["experts"][p["ncmoe"]:p["vision_ncmoe"]])
        self.assertGreaterEqual(freed, S.VISION_EXTRA)

    def test_more_cpu_layers_means_more_cache_slots(self):
        layout, g = LAYOUTS["IQ4_XS"], hw(32, 12)["gpus"][0]
        a = S.place(layout, g, 32 * GB, 6 * GB)
        b = S.place(layout, g, 32 * GB, 6 * GB, ncmoe=a["ncmoe"] + 1)
        self.assertGreater(b["slots"], a["slots"])

    def test_threads_capped_at_physical_cores_and_8(self):
        self.assertEqual(S.threads_for({"physical_cores": 6}), 6)
        self.assertEqual(S.threads_for({"physical_cores": 16}), 8)


class GgufHeader(unittest.TestCase):
    @staticmethod
    def build(tensors, kv):
        """Minimal GGUF v3: kv = {key: (type, value)}, tensors = [(name, size_bytes)] laid out back to back."""
        def s(x):
            return struct.pack("<Q", len(x)) + x.encode()
        out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
        for k, (t, v) in kv.items():
            out += s(k) + struct.pack("<I", t) + (s(v) if t == 8 else struct.pack("<I", v))
        off = 0
        for name, size in tensors:
            out += s(name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<IQ", 0, off)
            off += size
        header_len = len(out)
        return out, (header_len + 31) // 32 * 32 + off

    def test_expert_bytes_per_layer_from_offsets(self):
        data, file_size = self.build(
            [("token_embd.weight", 1000), ("blk.0.ffn_up_exps.weight", 300), ("blk.0.ffn_down_exps.weight", 200),
             ("blk.0.attn_q.weight", 50), ("blk.1.ffn_gate_exps.weight", 400), ("output.weight", 70)],
            {"general.architecture": (8, "qwen35moe"), "qwen35moe.block_count": (4, 2)})
        layout = S.parse_gguf_header(data, file_size)
        self.assertEqual(layout["n_layers"], 2)
        self.assertEqual(layout["experts"], [500, 400])
        self.assertEqual(layout["other"], 1000 + 50 + 70)

    def test_truncated_header_raises_eof(self):
        data, file_size = self.build([("blk.0.ffn_up_exps.weight", 10)],
                                     {"general.architecture": (8, "qwen35moe"), "qwen35moe.block_count": (4, 1)})
        with self.assertRaises(EOFError):
            S.parse_gguf_header(data[:20], file_size)


if __name__ == "__main__":
    unittest.main()
