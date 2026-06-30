"""Pipeline patches — WORKING"""

import os
import sys
import importlib.util

sys.dont_write_bytecode = True

def apply_all_patches():
    patch_dir = os.path.dirname(os.path.abspath(__file__))
    patches = ["patch_final_v2", "patch_eb_physics", "patch_snr_fix"]
    
    for name in patches:
        try:
            path = os.path.join(patch_dir, name + ".py")
            if not os.path.exists(path):
                continue
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            if hasattr(module, "apply"):
                module.apply()
        except Exception as e:
            print(f"  ⚠️  Patch {name}: {e}")
    
    print("\n" + "="*60)
    print("  ALL PATCHES APPLIED ✅")
    print("="*60 + "\n")
