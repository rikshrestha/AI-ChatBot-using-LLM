import os
import gradio as gr
from huggingface_hub import InferenceClient

# -----------------------
# Load Token and Model
# -----------------------
HF_TOKEN = os.environ.get("HF_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_API_TOKEN is missing.")

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")

client = InferenceClient(token=HF_TOKEN, model=MODEL_ID)


# -----------------------
# Ask model helper
# -----------------------
def ask_model(messages):
    try:
        response = client.chat_completion(
            messages,
            max_tokens=300,
            stream=False
        )

        # HF often returns response.choices[0].message["content"]
        if hasattr(response, "choices"):
            msg = response.choices[0].message
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
        return "I could not parse the model output."

    except Exception as e:
        return f"Model Error: {str(e)}"


# -----------------------
# Main Chat Logic
# -----------------------
def respond(user_message, history):
    """
    history = list of dict messages:
    [
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
    ]
    """

    if history is None:
        history = []

    # Add user msg
    history.append({"role": "user", "content": user_message})

    # Call model
    bot_reply = ask_model(history)

    # Add bot msg
    history.append({"role": "assistant", "content": bot_reply})

    return history, ""  # return updated history & clear textbox


def clear_chat():
    return [], ""


# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🔥 Mistral-7B-Instruct Model ChatBot")

    chatbot = gr.Chatbot()  # IMPORTANT!
    msg = gr.Textbox(placeholder="Ask me anything…", show_label=False)
    clear_btn = gr.Button("Clear")

    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()
