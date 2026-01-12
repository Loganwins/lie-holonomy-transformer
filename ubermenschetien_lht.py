#!/usr/bin/env python3
"""
UBERMENSCHETIEN HEAVEN ENGINE + LIE-HOLONOMY TRANSFORMER
---------------------------------------------------------
Full integration: Soviet-Nietzschean AI scaffold with geometric reasoning.

Features:
- Hermes-3 merged model (local, 4-bit quantized)
- Memory: JSONL logbook + optional vector embeddings
- Tools: shell, python sandbox, local search
- Soviet cybernetics: Tsetlin Automata (tool reinforcement)
- Nietzschean reflection: maxim generation, critique cycles
- LHT: Lie-Holonomy geometric consistency analysis
- Final Übermensch Report
"""

import os, sys, json, time, shutil, subprocess, traceback, random, math, statistics, re
from datetime import datetime
from typing import List, Dict, Any, Optional

# === PATHS ===
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SCRIPT_DIR = os.path.join(ROOT, "scripts")
RUN_DIR = os.path.join(ROOT, "runs")

for path in [DATA_DIR, SCRIPT_DIR, RUN_DIR]:
    os.makedirs(path, exist_ok=True)

# === MODEL PATHS ===
MODEL_ROOT = "/media/programmer/nvme2/ubermesnchetien4/models"
MODEL_PATH = f"{MODEL_ROOT}/current-merged-v4"
TOKENIZER_PATH = f"{MODEL_ROOT}/tokenizer_ref"

# === OPTIONAL: Voice synthesis ===
VOICE_OK = False
try:
    import pyttsx3
    TTS = pyttsx3.init()
    VOICE_OK = True
except:
    pass

# === OPTIONAL: Vector memory ===
VECTOR_OK = False
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = os.environ.get("UBERMENCHETIEN_EMBED_MODEL", "all-MiniLM-L6-v2")
    _client = chromadb.Client()
    _collection = _client.get_or_create_collection("ubermenschetien_memory")
    _embedder = SentenceTransformer(EMBED_MODEL)
    VECTOR_OK = True
except:
    pass

# === LHT: Lie-Holonomy Transformer ===
LHT_OK = False
try:
    from lht import LieHolonomyTransformer, LHTConfig, WaypointDetector
    LHT_OK = True
except:
    pass

# === CONFIGURATION ===
class Config:
    system = ("Übermenschetien Heaven Engine: Machiavellian mastermind, disciplined builder, "
              "Nietzschean Übermensch with Soviet cybernetic rigor. Embody iron pragmatism, "
              "high-agency maximalist outcomes. Lie-Holonomy geometric reasoning enabled.")
    temperature = 1.01
    top_p = 0.92
    repetition_penalty = 1.05
    max_new_tokens = 500
    use_voice = False
    use_vector_memory = VECTOR_OK
    use_lht = LHT_OK
    autonomy = False
    reflect_every = 3

    @staticmethod
    def toggle(name: str):
        if not hasattr(Config, name):
            return f"[config] no such flag: {name}"
        val = getattr(Config, name)
        if isinstance(val, bool):
            setattr(Config, name, not val)
            return f"[config] {name} → {getattr(Config, name)}"
        return f"[config] {name} not boolean; current={val}"

# === STATE & MEMORY ===
class Store:
    state_path = f"{RUN_DIR}/state.json"
    mem_path = f"{RUN_DIR}/memory.jsonl"
    goals_path = f"{RUN_DIR}/goals.json"
    plans_path = f"{RUN_DIR}/plans.jsonl"

    state = {
        "self": "I am Ubermenschetien Heaven Engine — I seek self-overcoming through disciplined creation.",
        "turn": 0,
        "lht_scores": []
    }
    goals: List[str] = []

    @classmethod
    def load(cls):
        if os.path.exists(cls.state_path):
            cls.state = json.load(open(cls.state_path))
        if os.path.exists(cls.goals_path):
            cls.goals = json.load(open(cls.goals_path))

    @classmethod
    def save(cls):
        json.dump(cls.state, open(cls.state_path, "w"), indent=2)
        json.dump(cls.goals, open(cls.goals_path, "w"), indent=2)

    @classmethod
    def log_mem(cls, kind: str, payload: Any):
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "data": payload
        }
        with open(cls.mem_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        if Config.use_vector_memory and VECTOR_OK:
            text = f"{kind}: {json.dumps(payload, ensure_ascii=False)}"
            vec = _embedder.encode([text])[0].tolist()
            _collection.add(
                documents=[text],
                embeddings=[vec],
                ids=[f"{kind}-{Store.state['turn']}-{random.randint(0,1_000_000)}"]
            )

# === LLM LOADING ===
def load_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    print(f"[llm] Loading {MODEL_PATH}...")
    
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map={"": "cuda:0"},
        torch_dtype=torch.float16,
        local_files_only=True
    )
    print("[llm] Loaded on GPU (4-bit)")
    return tok, model

# === LLM GENERATION ===
def generate(tok, model, user: str, temperature=None, top_p=None, 
             repetition_penalty=None, max_new_tokens=None) -> str:
    import torch
    temperature = temperature or Config.temperature
    top_p = top_p or Config.top_p
    repetition_penalty = repetition_penalty or Config.repetition_penalty
    max_new_tokens = max_new_tokens or Config.max_new_tokens
    
    prompt = (f"<|im_start|>system\n{Config.system}\n"
              f"<|im_start|>user\n{user}\n<|im_start|>assistant\n")
    
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant\n", 1)[-1].strip()
    
    # Voice output
    if Config.use_voice and VOICE_OK:
        try:
            TTS.say(text[:500])
            TTS.runAndWait()
        except:
            pass
    
    return text

# === TOOLS (with Soviet Tsetlin automaton scoring) ===
ALLOWED_SHELL = {"ls", "cat", "wc", "head", "tail", "nvidia-smi", "df", "du", "grep", "rg", "python3", "python", "find"}

def tool_shell(cmd: str) -> str:
    try:
        exe = cmd.strip().split()[0]
        if exe not in ALLOWED_SHELL:
            return f"[shell] blocked: {exe}"
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return p.stdout.decode("utf-8", errors="ignore")[:8000]
    except Exception as e:
        return f"[shell] error: {e}"

def tool_py(code: str) -> str:
    try:
        g = {
            "__builtins__": {"range": range, "len": len, "min": min, "max": max, 
                           "sum": sum, "print": print, "str": str, "int": int, 
                           "float": float, "list": list, "dict": dict},
            "math": math, "json": json, "re": re, "statistics": statistics, "random": random
        }
        l = {}
        exec(code, g, l)
        return f"[py] ok\n{l.get('out', '')}"
    except Exception:
        return f"[py] error:\n{traceback.format_exc()[-2000:]}"

def tool_search_local(query: str, path: str = ROOT) -> str:
    rg = shutil.which("rg")
    if rg:
        cmd = f'rg -n --no-heading --hidden -S "{query}" {path}'
    else:
        cmd = f'grep -RIn --exclude-dir=.git --exclude-dir=__pycache__ -e "{query}" {path}'
    return tool_shell(cmd)

TOOLS = {"shell": tool_shell, "python": tool_py, "search": tool_search_local}
TOOL_SCORES = {k: 0 for k in TOOLS}  # Soviet Tsetlin automaton scores

def update_tool_score(tool: str, success: bool):
    if tool not in TOOL_SCORES:
        return
    TOOL_SCORES[tool] += (1 if success else -1)
    TOOL_SCORES[tool] = max(-5, min(20, TOOL_SCORES[tool]))

def tool_router(question: str, tok, model) -> str:
    sketch = generate(tok, model,
        f"Choose a tool for:\n{question}\nReply ONLY with JSON: {{'tool':'shell|python|search|none','arg':'...'}}")
    try:
        j = json.loads(sketch.splitlines()[-1].replace("'", '"'))
    except:
        return "[tool:none]"
    
    tool, arg = j.get("tool", "none"), j.get("arg", "")
    if tool in TOOLS:
        res = TOOLS[tool](arg)[:4000]
        update_tool_score(tool, True)
        Store.log_mem("tool", {"tool": tool, "arg": arg, "res_head": res[:500]})
        return f"[tool:{tool}] {res}"
    
    update_tool_score(tool, False)
    return "[tool:none]"

# === LHT ANALYSIS ===
def lht_analyze(text: str, tok) -> str:
    if not LHT_OK or not Config.use_lht:
        return "[lht] disabled"
    
    import torch
    config = LHTConfig(
        vocab_size=32000,
        d_model=256,
        d_fiber=32,
        n_heads=4,
        n_layers=2,
        lie_algebra_rank=4,
        max_seq_len=512
    )
    lht = LieHolonomyTransformer(config)
    tokens = tok(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        out = lht(input_ids=tokens["input_ids"].clamp(0, 31999).clamp(0, 31999), return_geometric_losses=True)
    
    hol = out.get("holonomy_loss", torch.tensor(0.0))
    curv = out.get("curvature_loss", torch.tensor(0.0))
    
    if isinstance(hol, torch.Tensor):
        hol = hol.item() if torch.isfinite(hol) else 999.0
    if isinstance(curv, torch.Tensor):
        curv = curv.item() if torch.isfinite(curv) else 999.0
    
    consistency = 1.0 / (1.0 + min(hol, 100))
    Store.state["lht_scores"].append(consistency)
    
    verdict = "✓ CONSISTENT" if consistency > 0.5 else "⚠ CHECK REASONING"
    
    return f"""
[LHT Geometric Analysis]
  Holonomy:    {hol:.4f} (lower = more consistent)
  Curvature:   {curv:.4f} (lower = simpler reasoning)
  Consistency: {consistency:.1%}
  Verdict:     {verdict}
"""

# === PLANNING / REFLECTION ===
def persona_directive() -> str:
    return "Übermenschetien Heaven Engine: Soviet cybernetic Nietzschean clarity, pragmatic maxims, geometric truth."

def plan_for(goal: str, tok, model) -> str:
    user = (f"{persona_directive()}\nGoal: {goal}\n"
            f"Deliver:\n- 5 concrete steps\n- Constraints & risks\n- Nightly audit criteria\n- Nietzschean maxim")
    response = generate(tok, model, user)
    
    # LHT analysis
    if Config.use_lht:
        response += lht_analyze(response, tok)
    
    return response

def reflect_on(last_output: str, tok, model) -> str:
    user = f"{persona_directive()}\nCritique and improve:\n{last_output}\nReturn refined plan with sharper steps."
    response = generate(tok, model, user)
    
    if Config.use_lht:
        response += lht_analyze(response, tok)
    
    return response

# === FINAL REPORT ===
def final_report():
    print("\n" + "=" * 60)
    print("FINAL ÜBERMENSCH REPORT")
    print("=" * 60)
    print(f"Turns completed: {Store.state['turn']}")
    print(f"Goals tracked: {len(Store.goals)}")
    print(f"\nTool scores (Tsetlin automata):")
    print(json.dumps(TOOL_SCORES, indent=2))
    
    if os.path.exists(Store.mem_path):
        lines = open(Store.mem_path).read().splitlines()
        print(f"\nMemory entries: {len(lines)}")
    
    # LHT metrics
    if Store.state.get("lht_scores"):
        scores = Store.state["lht_scores"]
        print(f"\n[LHT Reasoning Metrics]")
        print(f"  Analyses run:    {len(scores)}")
        print(f"  Avg consistency: {sum(scores)/len(scores):.1%}")
        print(f"  Min consistency: {min(scores):.1%}")
        print(f"  Max consistency: {max(scores):.1%}")
    
    print(f"\nVector memory: {'ON' if Config.use_vector_memory else 'OFF'}")
    print(f"LHT reasoning: {'ON' if Config.use_lht else 'OFF'}")
    print(f"Voice output:  {'ON' if Config.use_voice else 'OFF'}")
    
    print("\n" + "-" * 60)
    print("Nietzschean maxim: Become who you are — iterate beyond all limits.")
    print("Geometric truth: Consistency is holonomy-freedom.")
    print("=" * 60)

# === HELP ===
HELP = """
╔══════════════════════════════════════════════════════════════╗
║         ÜBERMENSCHETIEN HEAVEN ENGINE + LHT                  ║
╠══════════════════════════════════════════════════════════════╣
║  GOALS                                                       ║
║    goals          List all goals                             ║
║    add: <text>    Add a new goal                             ║
║    del: <idx>     Delete goal by index                       ║
║    plan: <idx>    Generate plan for goal (with LHT)          ║
║                                                              ║
║  REASONING                                                   ║
║    reflect        Refine last plan                           ║
║    lht: <text>    Analyze text for geometric consistency     ║
║                                                              ║
║  TOOLS                                                       ║
║    tool: <query>  Auto-select and use tool                   ║
║    shell: <cmd>   Run shell command directly                 ║
║    py: <code>     Run Python code directly                   ║
║    search: <q>    Search local files                         ║
║                                                              ║
║  CONFIG                                                      ║
║    toggle <flag>  Toggle: use_voice, use_vector_memory,      ║
║                          use_lht, autonomy                   ║
║    status         Show current state                         ║
║                                                              ║
║  OTHER                                                       ║
║    help           Show this help                             ║
║    quit           Exit with final report                     ║
╚══════════════════════════════════════════════════════════════╝
"""

# === MAIN LOOP ===
def main():
    print("🟥🟨🟥 Übermenschetien Heaven Engine + Lie-Holonomy Transformer")
    print(f"    LHT Reasoning: {'ON' if LHT_OK else 'OFF'}")
    print(f"    Vector Memory: {'ON' if VECTOR_OK else 'OFF'}")
    print(f"    Voice Output:  {'ON' if VOICE_OK else 'OFF'}")
    print("    Type 'help' for commands.\n")

    Store.load()
    tok, model = load_llm()
    last_plan = ""

    while True:
        try:
            u = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not u:
            continue
        if u == "help":
            print(HELP)
            continue
        if u == "quit":
            break

        # Goals
        if u == "goals":
            print("[goals]")
            if not Store.goals:
                print("  (none)")
            for i, g in enumerate(Store.goals):
                print(f"  [{i}] {g}")
            continue

        if u.startswith("add:"):
            Store.goals.append(u[4:].strip())
            Store.save()
            print("[goals] added")
            continue

        if u.startswith("del:"):
            try:
                Store.goals.pop(int(u[4:].strip()))
                Store.save()
                print("[goals] deleted")
            except:
                print("[goals] bad index")
            continue

        if u.startswith("plan:"):
            try:
                goal = Store.goals[int(u[5:].strip())]
            except:
                print("[plan] bad index")
                continue
            out = plan_for(goal, tok, model)
            last_plan = out
            Store.log_mem("plan", {"goal": goal, "plan": out})
            print(out)
            continue

        # Reflection
        if u == "reflect":
            if not last_plan:
                print("[reflect] no plan to refine")
                continue
            improved = reflect_on(last_plan, tok, model)
            last_plan = improved
            Store.log_mem("reflect", {"plan": improved})
            print(improved)
            continue

        # LHT direct analysis
        if u.startswith("lht:"):
            print(lht_analyze(u[4:].strip(), tok))
            continue

        # Tools
        if u.startswith("tool:"):
            print(tool_router(u[5:].strip(), tok, model))
            continue

        if u.startswith("shell:"):
            print(tool_shell(u[6:].strip()))
            continue

        if u.startswith("py:"):
            print(tool_py(u[3:].strip()))
            continue

        if u.startswith("search:"):
            print(tool_search_local(u[7:].strip()))
            continue

        # Config
        if u.startswith("toggle"):
            parts = u.split(maxsplit=1)
            if len(parts) > 1:
                print(Config.toggle(parts[1]))
            else:
                print("[toggle] specify flag: use_voice, use_vector_memory, use_lht, autonomy")
            continue

        if u == "status":
            status = {
                "turn": Store.state["turn"],
                "goals": len(Store.goals),
                "autonomy": Config.autonomy,
                "use_vector_memory": Config.use_vector_memory,
                "use_lht": Config.use_lht,
                "use_voice": Config.use_voice,
                "tool_scores": TOOL_SCORES,
                "model": "current-merged-v4 + LoRA"
            }
            print(json.dumps(status, indent=2))
            continue

        # Default: free conversation
        out = generate(tok, model, f"{persona_directive()}\nUser request: {u}\nProvide procedure + Nietzschean maxim.")
        Store.log_mem("reply", {"in": u, "out": out})
        print(out)
        
        # Periodic LHT check
        if Config.use_lht and Store.state["turn"] % 3 == 0:
            print(lht_analyze(out, tok))
        
        Store.state["turn"] += 1
        Store.save()

    final_report()

if __name__ == "__main__":
    main()
