"""Tamil TAPO math demo for a Hugging Face ZeroGPU Space."""
import os
import re
import html

import spaces  # Import before torch to initialize ZeroGPU CUDA emulation.
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL_ID = os.getenv("MODEL_ID", "GL3MON/Qwen3.5-2B-Tamil-TAPO")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
XCOT_MODEL_ID = os.getenv("XCOT_MODEL_ID", "jg-eno/Qwen-3.5-sft-ta-ml")
XCOT_BASE_ID = "Qwen/Qwen3.5-2B"
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))
SYSTEM_PROMPT = (
    "Solve the mathematical word problem. First translate the complete question "
    "faithfully into English inside <english_translation>...</english_translation>. "
    "Preserve all quantities, units, and relationships. Then reason in English "
    "and give the final answer in \\boxed{...}. Do not put reasoning inside the translation tags."
)
XCOT_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem carefully. Explain the reasoning in English, "
    "even when the question is in another language. "
    "End with #### followed by the final numeric answer."
)
TAPO_MODE = "TAPO · Tamil reasoning"
XCOT_MODE = "xCOT-style SFT · multilingual"
LANGUAGES = ["Tamil", "Malayalam", "English", "Romanized Tamil", "Romanized Malayalam"]


def load_models():
    options = {"revision": MODEL_REVISION, "token": os.getenv("HF_TOKEN") or None}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
    tapo_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, **options
    ).eval().to("cuda")
    xcot_tokenizer = AutoTokenizer.from_pretrained(XCOT_MODEL_ID, token=os.getenv("HF_TOKEN") or None)
    xcot_base = Qwen3_5ForConditionalGeneration.from_pretrained(
        XCOT_BASE_ID, dtype=torch.bfloat16, token=os.getenv("HF_TOKEN") or None,
    )
    xcot_model = PeftModel.from_pretrained(
        xcot_base, XCOT_MODEL_ID, token=os.getenv("HF_TOKEN") or None,
        torch_device="cpu",
    ).eval().to("cuda")
    return tokenizer, tapo_model, xcot_tokenizer, xcot_model


tokenizer, tapo_model, xcot_tokenizer, xcot_model = load_models()


@spaces.GPU(duration=120)
def solve(question, max_new_tokens, mode=TAPO_MODE, language="Tamil"):
    if not question.strip():
        raise gr.Error("Enter a mathematical word problem.")
    if not 128 <= int(max_new_tokens) <= 2048:
        raise gr.Error("Maximum output tokens must be between 128 and 2048.")
    if mode == TAPO_MODE:
        active_tokenizer, model = tokenizer, tapo_model
        inputs = active_tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": question.strip()}],
            tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        )
    elif mode == XCOT_MODE:
        if language not in LANGUAGES:
            raise gr.Error("Select a supported question language.")
        active_tokenizer, model = xcot_tokenizer, xcot_model
        # The adapter tokenizer carries the same chat template used by the
        # processor in training_scripts/sft/xcot_ta_ml_gsm8k_train.py.
        prompt = xcot_tokenizer.apply_chat_template(
            [{"role": "system", "content": XCOT_SYSTEM_PROMPT},
             {"role": "user", "content": f"Question language: {language}\nProblem: {question.strip()}"}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = active_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    else:
        raise gr.Error("Select a supported inference model.")
    if inputs["input_ids"].shape[1] > MAX_INPUT_TOKENS:
        raise gr.Error(f"Please shorten the problem to fit {MAX_INPUT_TOKENS} input tokens.")
    inputs = inputs.to(model.device)
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=int(max_new_tokens),
                pad_token_id=active_tokenizer.pad_token_id or active_tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise gr.Error("GPU memory exhausted. Shorten the input or reduce the output limit.") from None
    return active_tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


EXAMPLES = [
    ("Tamil · Reading a book", "ஜூலி 120 பக்க புத்தகத்தைப் படித்து வருகிறார். நேற்று, அவளால் 12 பக்கங்களைப் படிக்க முடிந்தது, இன்று, அவர் நேற்று விட இரண்டு மடங்கு பக்கங்களைப் படித்தார். மீதமுள்ள பக்கங்களில் பாதியை நாளை படிக்க விரும்பினால், அவள் எத்தனை பக்கங்களைப் படிக்க வேண்டும்?"),
    ("Tamil · Selling clips", "ஏப்ரல் மாதத்தில் நடாலியா தனது 48 நண்பர்களுக்கு கிளிப்களை விற்றார், பின்னர் அவர் மே மாதத்தில் பாதி கிளிப்களை விற்றார். ஏப்ரல் மற்றும் மே மாதங்களில் நடாலியா முற்றிலும் எத்தனை கிளிப்களை விற்றது?"),
    ("Malayalam · Babysitting", "വെങ് ബേബിസിറ്റിംഗിന് മണിക്കൂറിൽ $12 സമ്പാദിക്കുന്നു. ഇന്നലെ അവൾ 50 മിനിറ്റ് ബേബിസിറ്റിംഗ് ചെയ്തു. അവൾക്ക് എത്ര രൂപ ലഭിച്ചു?"),
]
EMPTY_RESPONSE = "*Your solution will appear here.*"


def present_response(raw):
    # Escape model text before adding our own headings; keep LaTeX intact.
    translation = re.search(r"<english_translation>(.*?)</english_translation>", raw, re.DOTALL)
    if translation:
        reasoning = raw[translation.end():].strip()
        return ("### English translation\n\n" + html.escape(translation.group(1).strip())
                + "\n\n---\n\n### Worked solution\n\n" + html.escape(reasoning))
    if "####" in raw:
        reasoning, answer = raw.rsplit("####", 1)
        return ("### Worked solution\n\n" + html.escape(reasoning.strip())
                + "\n\n---\n\n### Final answer\n\n**" + html.escape(answer.strip()) + "**")
    return html.escape(raw)


CSS = """
.gradio-container {max-width: 1140px !important; margin: auto;}
#hero {padding: 34px 4px 22px;}
#hero h1 {font-size: clamp(28px, 4vw, 42px); letter-spacing: -1.3px; line-height: 1.15;}
#hero p {max-width: 700px; font-size: 16px; line-height: 1.65; opacity: .8;}
.model-link {display: inline-flex; align-items: center; gap: 8px; width: fit-content;
  margin-top: 4px; padding: 10px 14px; border: 1px solid var(--border-color-primary);
  border-radius: 10px; color: #0d9488; font-size: 14px; font-weight: 600;
  text-decoration: none;}
.model-link:hover {border-color: #0d9488; text-decoration: underline;}
#eyebrow {font-size: 11px; letter-spacing: 2px; font-weight: 700; color: #0d9488;}
#input-panel, #output-panel {border-radius: 18px; padding: 24px; border: 1px solid var(--border-color-primary);}
#question textarea {font-size: 16px; line-height: 1.85;}
#solve {min-height: 48px; border-radius: 10px; font-weight: 600;}
#solution {min-height: 330px; font-size: 15px; line-height: 1.85;}
#solution h3 {font-size: 17px; margin-top: 12px;}
.example-button {text-align: left; border-radius: 10px; min-height: 42px; font-size: 13px;}
#footer {padding: 16px 4px; font-size: 12px; opacity: .65;}
@media (max-width: 640px) {#input-panel, #output-panel {padding: 16px;} #hero {padding-top: 18px;}}
"""
THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate")

with gr.Blocks(title="IndicReason · Tamil TAPO") as demo:
    with gr.Column(elem_id="hero"):
        gr.HTML('<div id="eyebrow">INDICREASON / TAMIL TAPO</div>')
        gr.Markdown("# IndicReason - TAPO Training\n"
                    "Explore mathematical reasoning across Indic languages. Choose TAPO or "
                    "the multilingual SFT model, then enter a word problem.")
        gr.HTML('<a class="model-link" href="https://huggingface.co/GL3MON/Qwen3.5-2B-Tamil-TAPO" '
                'target="_blank" rel="noopener noreferrer">TAPO model weights ↗</a> '
                '<a class="model-link" href="https://huggingface.co/jg-eno/Qwen-3.5-sft-ta-ml" '
                'target="_blank" rel="noopener noreferrer">xCOT-style SFT adapter ↗</a>')
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320, elem_id="input-panel"):
            gr.Markdown("### Your question\nSelect a model, then write a problem or choose an example.")
            mode = gr.Radio([TAPO_MODE, XCOT_MODE], value=TAPO_MODE, label="Inference model")
            language = gr.Dropdown(LANGUAGES, value="Tamil", label="Question language (multilingual SFT)")
            question = gr.Textbox(label="Word problem", show_label=False, lines=7,
                                  placeholder="Type or paste your question here…", elem_id="question")
            gr.Markdown("**Try an example**")
            example_buttons = []
            for label, problem in EXAMPLES:
                button = gr.Button(label, size="sm", elem_classes=["example-button"])
                example_buttons.append((button, problem))
            with gr.Accordion("Generation settings", open=False):
                limit = gr.Slider(128, 2048, value=1536, step=128, label="Maximum response tokens")
            with gr.Row():
                submit = gr.Button("Generate solution", variant="primary", scale=3, elem_id="solve")
                clear = gr.Button("Clear", scale=1)
        with gr.Column(scale=1, min_width=320, elem_id="output-panel"):
            gr.Markdown("### The solution\nEnglish translation · Reasoning · Final answer")
            solution = gr.Markdown(EMPTY_RESPONSE, elem_id="solution",
                                   latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                     {"left": r"\[", "right": r"\]", "display": True},
                                                     {"left": r"\(", "right": r"\)", "display": False}])
            with gr.Accordion("View original model response", open=False):
                answer = gr.Textbox(label="Original response", lines=10, interactive=False,
                                    buttons=["copy"])
    gr.Markdown("TAPO is trained for Tamil; the multilingual adapter is an xCOT-style SFT baseline. "
                "Responses may contain errors; check the translation and arithmetic.", elem_id="footer")
    for button, problem in example_buttons:
        example_language = "Malayalam" if button == example_buttons[-1][0] else "Tamil"
        button.click(lambda text=problem, lang=example_language: (text, lang),
                     inputs=None, outputs=[question, language], api_name=False)
    clear.click(lambda: ("", "", EMPTY_RESPONSE), outputs=[question, answer, solution], api_name=False)
    submit.click(solve, [question, limit, mode, language], answer, api_name="solve", concurrency_limit=1).then(
        present_response, answer, solution, api_name=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
        theme=THEME, css=CSS,
    )
