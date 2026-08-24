"""
Script điều phối chạy tất cả các bước của Lab Day 22:
  1. src/01_langsmith_rag_pipeline.py
  2. src/02_prompt_hub_ab_routing.py
  3. src/03_ragas_evaluation.py
  4. src/04_guardrails_validator.py
"""

import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def run_step(step_name: str, script_path: str):
    print("\n" + "=" * 65)
    print(f"🚀 Đang chạy {step_name}: {script_path}")
    print("=" * 65)
    result = subprocess.run([sys.executable, str(ROOT_DIR / script_path)], cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"❌ {step_name} thất bại với mã lỗi {result.returncode}")
        return False
    print(f"✅ {step_name} hoàn thành thành công!")
    return True


def main():
    print("=================================================================")
    print("           Day 22 Lab: LangSmith & Evaluation Pipeline           ")
    print("=================================================================")

    steps = [
        ("Bước 1: RAG Pipeline với LangSmith", "src/01_langsmith_rag_pipeline.py"),
        ("Bước 2: Prompt Hub & A/B Routing",   "src/02_prompt_hub_ab_routing.py"),
        ("Bước 3: RAGAS Evaluation",           "src/03_ragas_evaluation.py"),
        ("Bước 4: Guardrails AI Validators",   "src/04_guardrails_validator.py"),
    ]

    for name, script in steps:
        success = run_step(name, script)
        if not success:
            sys.exit(1)

    print("\n" + "=" * 65)
    print("🎉 TẤT CẢ CÁC BƯỚC ĐÃ HOÀN THÀNH XUẤT SẮC!")
    print("=" * 65)


if __name__ == "__main__":
    main()
