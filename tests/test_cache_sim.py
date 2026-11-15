from pathlib import Path
import importlib.util
import struct
import sys
import tempfile
import unittest

module_path = Path(__file__).resolve().parents[1] / "bench" / "cache-sim.py"
spec = importlib.util.spec_from_file_location("cache_sim", module_path)
cache_sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cache_sim
spec.loader.exec_module(cache_sim)


def position(cpu_route, gpu_route):
    return (tuple(cpu_route), tuple(gpu_route))


class CacheReplayTests(unittest.TestCase):
    def test_expert_weight_size_uses_ggml_block_formula(self):
        tiel_gate = cache_sim.quantized_bytes_per_expert((2048, 512, 256), 256, 21)
        tiel_down = cache_sim.quantized_bytes_per_expert((512, 2048, 256), 256, 23)
        occamy_each = cache_sim.quantized_bytes_per_expert((512, 2048, 256), 256, 23)
        self.assertEqual(tiel_gate, 450560)
        self.assertEqual(tiel_down, 557056)
        self.assertEqual(2*tiel_gate+tiel_down, 1458176)
        self.assertEqual(3*occamy_each, 1671168)

    def test_csv_parser_splits_token_index_resets_without_overlap(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            path = Path(tmp) / "trace.csv"
            rows = []
            for base in (0, 0):
                for token in range(2):
                    for layer in range(2):
                        rows.append(",".join(map(str, (token, layer, *([layer+token]*8)))))
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            sequences = cache_sim.read_trace_csv(path, expected_layers=(0, 1))
            self.assertEqual([len(s["positions"]) for s in sequences], [2, 2])
            self.assertEqual(sequences[0]["positions"], sequences[1]["positions"])

    def test_negative_prefill_rows_are_filtered_like_runtime_profile_loader(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            path=Path(tmp)/"negative.csv"
            rows=[]
            for layer in range(2):
                rows.append(",".join(map(str,(-1,layer,*([layer]*8)))))
            for layer in range(2):
                rows.append(",".join(map(str,(0,layer,*([layer]*8)))))
            path.write_text("\n".join(rows)+"\n",encoding="utf-8")
            generated=cache_sim.read_trace_csv(path,expected_layers=(0,1))
            all_rows=cache_sim.read_trace_csv(path,expected_layers=(0,1),include_negative=True)
            self.assertEqual(sum(len(s["positions"]) for s in generated),1)
            self.assertEqual(sum(len(s["positions"]) for s in all_rows),2)

    def test_static_top_n_and_per_layer_hits(self):
        train = [{"positions": [position(range(8), [0]*8), position(range(8), [1]*8)]}]
        counts = cache_sim.profile_counts(train, gpu_layers=(1,))
        cache = cache_sim.top_cache(counts, 1, gpu_layers=(1,))
        measured = cache_sim.evaluate_static(cache, train, {0:100,1:10}, cpu_layers=(0,), gpu_layers=(1,))
        self.assertEqual(measured["hit_rate"], 0.5)
        self.assertEqual(measured["layers"][1]["hit_rate"], 0.5)
        self.assertEqual(measured["ram_mb_per_token"], (8*100+8*10*(1-.5))/1e6)

    def test_lru_replacement_cap_is_per_layer_per_token_and_counts_pcie(self):
        train = [{"positions": [position([0]*8, [0]*8) for _ in range(4)]}]
        check = [{"positions": [position([0]*8, [1]*8) for _ in range(3)]}]
        out = cache_sim.simulate_adaptive(train, [], check, {0:100,1:10}, capacity=1,
            policy="lru", replacements_per_layer=1, cpu_layers=(0,), gpu_layers=(1,))
        self.assertEqual(out["replacements"], 1)
        self.assertEqual(out["layers"][1]["replacements"], 1)
        self.assertEqual(out["hits"], 16)
        self.assertEqual(out["pcie_mb_per_token"], 8*10/3/1e6)
        self.assertEqual(out["ram_mb_per_token"], (3*8*100+8*10)/3/1e6)
        self.assertEqual(out["max_replacements_per_token"], 1)
        self.assertEqual(out["p99_replacements_per_token"], 1)

    def test_hybrid_keeps_profile_half_pinned_and_caps_dynamic_replacements(self):
        train = [{"positions": [position([0]*8, [0]*8), position([0]*8, [1]*8)]}]
        check = [{"positions": [position([0]*8, [2]*8), position([0]*8, [0]*4+[2]*4)]}]
        out = cache_sim.simulate_adaptive(train, [], check, {0:100,1:10}, capacity=2,
            policy="lru", replacements_per_layer=1, cpu_layers=(0,), gpu_layers=(1,),
            static_pinned_slots=1)
        self.assertEqual(out["static_pinned_slots"], 1)
        self.assertEqual(out["replacements"], 1)
        self.assertEqual(out["layers"][1]["hits"], 8)
        self.assertEqual(out["max_replacements_per_token"], 1)

    def test_gguf_header_reader_does_not_require_tensor_payload(self):
        def string(s):
            data=s.encode()
            return struct.pack("<Q",len(data))+data
        def meta(key, kind, value):
            return string(key)+struct.pack("<I",kind)+value
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            path=Path(tmp)/"tiny.gguf"
            entries=[
                meta("general.architecture",8,string("qwen3moe")),
                meta("qwen3moe.expert_count",4,struct.pack("<I",2)),
                meta("qwen3moe.expert_used_count",4,struct.pack("<I",1)),
                meta("qwen3moe.block_count",4,struct.pack("<I",1)),
            ]
            tensor_infos=[]
            for name,offset in (("blk.0.ffn_up_exps.weight",0),("blk.0.ffn_gate_exps.weight",1024),("blk.0.ffn_down_exps.weight",2048)):
                tensor_infos.append(string(name)+struct.pack("<I",2)+struct.pack("<QQ",256,2)+struct.pack("<IQ",23,offset))
            path.write_bytes(b"GGUF"+struct.pack("<IQQ",3,3,4)+b"".join(entries)+b"".join(tensor_infos))
            parsed=cache_sim.read_gguf_experts(path)
            self.assertEqual(parsed["expert_count"],2)
            self.assertEqual(parsed["experts_per_token"],1)
            self.assertEqual(parsed["expert_bytes"][0],3*136)


if __name__ == "__main__":
    unittest.main()
