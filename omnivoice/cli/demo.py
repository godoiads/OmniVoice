#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
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
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Voice profiles (saved cloned voices)
# ---------------------------------------------------------------------------
_VOICES_DIR = Path(__file__).resolve().parent.parent.parent / "voices"
_VOICES_DIR.mkdir(exist_ok=True)


def _list_voices():
    """Return list of saved voice profile names."""
    return ["Nenhum"] + sorted(
        p.stem for p in _VOICES_DIR.glob("*.pt")
    )


def _save_voice(name, prompt_data):
    """Save a voice clone prompt to disk."""
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in name).strip()
    if not safe_name:
        return "Nome invalido."
    path = _VOICES_DIR / f"{safe_name}.pt"
    torch.save(prompt_data, path)
    return f"Voz '{safe_name}' salva com sucesso."


def _load_voice(name):
    """Load a saved voice clone prompt."""
    if not name or name == "Nenhum":
        return None
    path = _VOICES_DIR / f"{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    return None


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Genero": ["Male / Masculino", "Female / Feminino"],
    "Idade": [
        "Child / Crianca",
        "Teenager / Adolescente",
        "Young Adult / Jovem adulto",
        "Middle-aged / Meia-idade",
        "Elderly / Idoso",
    ],
    "Tom de voz": [
        "Very Low Pitch / Muito grave",
        "Low Pitch / Grave",
        "Moderate Pitch / Moderado",
        "High Pitch / Agudo",
        "Very High Pitch / Muito agudo",
    ],
    "Estilo": ["Whisper / Sussurro"],
    "Sotaque em ingles": [
        "American Accent / Americano",
        "Australian Accent / Australiano",
        "British Accent / Britanico",
        "Chinese Accent / Chines",
        "Canadian Accent / Canadense",
        "Indian Accent / Indiano",
        "Korean Accent / Coreano",
        "Portuguese Accent / Portugues",
        "Russian Accent / Russo",
        "Japanese Accent / Japones",
    ],
    "Dialeto chines": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "Sotaque em ingles": "Funciona apenas para fala em ingles.",
    "Dialeto chines": "Funciona apenas para fala em chines.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Digite o texto para sintetizar."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Envie um audio de referencia."
            # Auto-trim: usar apenas os primeiros 8 segundos
            import torchaudio as _ta
            _wav, _sr = _ta.load(ref_audio)
            _max_samples = _sr * 8
            if _wav.shape[1] > _max_samples:
                _trimmed_path = ref_audio + ".trim.wav"
                _ta.save(_trimmed_path, _wav[:, :_max_samples], _sr)
                ref_audio = _trimmed_path
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Pronto."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    ).set(
        body_background_fill="#0f0f0f",
        body_background_fill_dark="#0f0f0f",
        background_fill_primary="#1a1a1a",
        background_fill_primary_dark="#1a1a1a",
        background_fill_secondary="#222222",
        background_fill_secondary_dark="#222222",
        block_background_fill="#1a1a1a",
        block_background_fill_dark="#1a1a1a",
        block_border_color="#333333",
        block_border_color_dark="#333333",
        block_label_background_fill="#222222",
        block_label_background_fill_dark="#222222",
        block_title_text_color="#e0e0e0",
        block_title_text_color_dark="#e0e0e0",
        body_text_color="#d4d4d4",
        body_text_color_dark="#d4d4d4",
        body_text_color_subdued="#888888",
        body_text_color_subdued_dark="#888888",
        border_color_accent="#444444",
        border_color_accent_dark="#444444",
        border_color_primary="#333333",
        border_color_primary_dark="#333333",
        button_primary_background_fill="#D7F205",
        button_primary_background_fill_dark="#D7F205",
        button_primary_text_color="#0f0f0f",
        button_primary_text_color_dark="#0f0f0f",
        input_background_fill="#222222",
        input_background_fill_dark="#222222",
        input_border_color="#444444",
        input_border_color_dark="#444444",
        input_placeholder_color="#666666",
        input_placeholder_color_dark="#666666",
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important; color: #D7F205 !important;}
    .gradio-container .prose {font-size: 1.1em !important; color: #d4d4d4 !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    footer {display: none !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Idioma (opcional)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Mantenha como Auto para detectar automaticamente.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Configuracoes avancadas (opcional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Velocidade",
                info="1.0 = normal. >1 mais rapido, <1 mais lento. Ignorado se Duracao estiver definida.",
            )
            du = gr.Number(
                value=None,
                label="Duracao (segundos)",
                info=(
                    "Deixe vazio para usar velocidade."
                    " Defina uma duracao fixa para sobrescrever a velocidade."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Passos de inferencia",
                info="Padrao: 32. Menor = mais rapido, maior = melhor qualidade.",
            )
            dn = gr.Checkbox(
                label="Remover ruido",
                value=True,
                info="Padrao: ativado. Desmarque para desativar.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Escala de orientacao (CFG)",
                info="Padrao: 2.0.",
            )
            pp = gr.Checkbox(
                label="Pre-processar audio",
                value=True,
                info="Remove silencios e ajusta o audio de referencia.",
            )
            po = gr.Checkbox(
                label="Pos-processar saida",
                value=True,
                info="Remove silencios longos do audio gerado.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice") as demo:
        gr.Markdown(
            """
# OmniVoice

Modelo text-to-speech com suporte a **600+ idiomas**:

- **Clonar Voz** — Clone qualquer voz a partir de um audio de referencia
- **Design de Voz** — Crie vozes personalizadas com atributos

Desenvolvido com [OmniVoice](https://github.com/k2-fsa/OmniVoice)
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Clonar Voz"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Texto para sintetizar",
                            lines=4,
                            placeholder="Digite o texto que deseja transformar em fala...",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.95em;font-weight:600;color:#D7F205;'>"
                            "Vozes salvas</span>"
                        )
                        vc_saved = gr.Dropdown(
                            label="Usar voz salva",
                            choices=_list_voices(),
                            value="Nenhum",
                            info="Selecione uma voz salva ou envie um audio abaixo.",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "— OU envie um novo audio de referencia —"
                            "</span>"
                        )
                        vc_ref_audio = gr.Audio(
                            label="Audio de referencia",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recomendado: audio de 3 a 10 segundos."
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label="Texto do audio de referencia (opcional)",
                            lines=2,
                            placeholder="Transcricao do audio de referencia. Deixe vazio"
                            " para transcrever automaticamente via Whisper.",
                        )
                        vc_lang = _lang_dropdown("Idioma (opcional)")
                        with gr.Accordion("Instrucao adicional (opcional)", open=False):
                            vc_instruct = gr.Textbox(label="Instrucao", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Gerar", variant="primary")
                        with gr.Accordion("Salvar voz clonada", open=False):
                            vc_save_name = gr.Textbox(
                                label="Nome da voz",
                                placeholder="Ex: Brian WestSide",
                            )
                            vc_save_btn = gr.Button("Salvar voz")
                            vc_save_status = gr.Textbox(
                                label="", lines=1, interactive=False
                            )
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Audio gerado",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status", lines=2)

                # Store last voice prompt for saving
                _last_voice_prompt = {"data": None}

                def _clone_fn(
                    text, lang, saved_voice, ref_aud, ref_text, instruct,
                    ns, gs, dn, sp, du, pp, po,
                ):
                    # Use saved voice if selected
                    saved = _load_voice(saved_voice)
                    if saved is not None:
                        _last_voice_prompt["data"] = saved
                        gen_config = OmniVoiceGenerationConfig(
                            num_step=int(ns or 32),
                            guidance_scale=float(gs) if gs is not None else 2.0,
                            denoise=bool(dn) if dn is not None else True,
                            preprocess_prompt=bool(pp),
                            postprocess_output=bool(po),
                        )
                        lang_val = lang if (lang and lang != "Auto") else None
                        kw = dict(
                            text=text.strip(),
                            language=lang_val,
                            generation_config=gen_config,
                            voice_clone_prompt=saved,
                        )
                        if sp is not None and float(sp) != 1.0:
                            kw["speed"] = float(sp)
                        if du is not None and float(du) > 0:
                            kw["duration"] = float(du)
                        if instruct and instruct.strip():
                            kw["instruct"] = instruct.strip()
                        try:
                            audio = model.generate(**kw)
                        except Exception as e:
                            return None, f"Erro: {type(e).__name__}: {e}"
                        waveform = (audio[0] * 32767).astype(np.int16)
                        return (sampling_rate, waveform), f"Pronto. (voz salva: {saved_voice})"

                    # Otherwise use ref audio (original flow)
                    result = _gen(
                        text, lang, ref_aud, instruct,
                        ns, gs, dn, sp, du, pp, po,
                        mode="clone", ref_text=ref_text or None,
                    )
                    # Capture the voice prompt for saving
                    if ref_aud:
                        try:
                            import torchaudio as _ta
                            _wav, _sr = _ta.load(ref_aud)
                            _max_samples = _sr * 8
                            _audio_path = ref_aud
                            if _wav.shape[1] > _max_samples:
                                _trimmed = ref_aud + ".trim.wav"
                                _ta.save(_trimmed, _wav[:, :_max_samples], _sr)
                                _audio_path = _trimmed
                            _last_voice_prompt["data"] = model.create_voice_clone_prompt(
                                ref_audio=_audio_path,
                                ref_text=ref_text or None,
                            )
                        except Exception:
                            pass
                    return result

                def _save_fn(name):
                    if not name or not name.strip():
                        return "Digite um nome para a voz.", gr.update()
                    if _last_voice_prompt["data"] is None:
                        return "Gere um audio primeiro para salvar a voz.", gr.update()
                    msg = _save_voice(name, _last_voice_prompt["data"])
                    return msg, gr.update(choices=_list_voices(), value="Nenhum")

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_saved,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

                vc_save_btn.click(
                    _save_fn,
                    inputs=[vc_save_name],
                    outputs=[vc_save_status, vc_saved],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Design de Voz"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Texto para sintetizar",
                            lines=4,
                            placeholder="Digite o texto que deseja transformar em fala...",
                        )
                        vd_lang = _lang_dropdown("Idioma (opcional)")

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Gerar", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Audio gerado",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
