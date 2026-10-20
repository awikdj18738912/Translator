"""Independent multilingual transcription, translation and refinement workbench."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def create_workbench(*, translation_model: Path, refiner_model: Path,
                     gpu: int, asr_url: str, data_dir: Path, token_previews: bool = True,
                     refinement_gate: str = "regex", translation_strategy: str = "adaptive",
                     refiner_dtype: str = "bfloat16"):
    # Import our local snapshot, never add the Codex-web checkout to sys.path.
    import torch
    from fastapi.staticfiles import StaticFiles
    from lmt_runtime import LmtRuntime
    from system.web_app import create_app
    from system.zh_itn_bridge import DEFAULT_LIBRARY as ZH_ITN_LIBRARY
    from system.translation_web import install_translation_routes

    torch.set_num_threads(4)
    data_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(
        refiner_model, f"cuda:{gpu}", asr_url, None, 256,
        data_dir / "refinement.jsonl", entity_db=data_dir / "entities.db",
        entity_fuzzy_mode="off", refinement_gate_mode=refinement_gate,
        entity_audio_repair=False, entity_asr_context=False, entity_ctc_mode="off",
        refiner_dtype=refiner_dtype,
        zh_itn_library=ZH_ITN_LIBRARY,
    )
    print("Loading LMT translation model...", flush=True)
    runtime = LmtRuntime(translation_model, gpu)

    def translate(source: str, language: str) -> str:
        rows, _ = runtime.translate([source], language, beams=3, max_new_tokens=512)
        if not rows[0]["eos"]:
            raise RuntimeError("翻译超过生成上限，请缩短输入片段。")
        return rows[0]["hypothesis"]

    def stream_translate(source,language,on_text):
        rows,_=runtime.translate_stream(source,language,on_text)
        if not rows[0]['eos']:
            raise RuntimeError('翻译超过生成上限，请缩短输入片段。')
        return rows[0]['hypothesis']

    translate("Please confirm the delivery date.", "en")
    install_translation_routes(app, translate, asr_url=asr_url,
                               output_dir=data_dir / "sessions",
                               stream_translate=stream_translate if token_previews else None,
                               translation_strategy=translation_strategy)
    app.mount("/static", StaticFiles(directory=ROOT / "system" / "web"), name="static")
    app.state.translation_runtime = runtime
    app.state.translation_gate_mode = refinement_gate
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-model", type=Path,
                        default=Path("/home/aim0/data/models/ASR/LMT-60-0.6B"))
    parser.add_argument("--refiner-model", type=Path,
                        default=Path("/home/aim0/data/models/ASR/AgenticASR-Refiner"))
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--refiner-dtype", choices=("auto", "float32", "float16", "bfloat16"),
                        default="bfloat16", help="Refiner weight precision; float32 restores the original loading precision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--asr-url", default="http://127.0.0.1:8766")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--refinement-gate",choices=("regex","off"),default="regex",
                        help="regex: existing text rules, no ASR confidence; off: refine every complete group")
    parser.add_argument("--translation-output", choices=("tokens", "windows"),default="tokens",
                        help="tokens: provisional prefixes and beam-3 confirmation per window; windows: original complete snapshots")
    parser.add_argument('--translation-strategy',choices=('adaptive','single'),default='adaptive')
    args = parser.parse_args()
    app = create_workbench(translation_model=args.translation_model,
                           refiner_model=args.refiner_model, gpu=args.gpu,
                           asr_url=args.asr_url, data_dir=args.data_dir,
                           token_previews=args.translation_output == "tokens",
                           refinement_gate=args.refinement_gate,translation_strategy=args.translation_strategy,
                           refiner_dtype=args.refiner_dtype)
    print(f"Translation workbench: http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
