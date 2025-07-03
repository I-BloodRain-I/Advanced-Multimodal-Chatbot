from typing import List, Literal, TypedDict, Generator
import logging
import time

import gradio as gr

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Type Definitions --------------------
class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str

# -------------------- Core Functions --------------------

def mock_token_stream(*args, **kwargs) -> Generator[str, None, None]:
    """
    Simulates token streaming by yielding tokens with delays.
    """
    for token in ["This", " is", " a", " streaming", " reply", " generated", " by", 
                  " an", " advanced", " language", " model", " using", " deep", 
                  " learning", " techniques"]:
        time.sleep(0.1)
        yield token

def format_prompt(message: str, history: List[Message]) -> str:
    """"
    Constructs a prompt string from the message and history.
    """
    prompt_parts = []
    for msg in history or []:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            prompt_parts.append(f"User: {content}")
        else:
            prompt_parts.append(f"Assistant: {content}")
    prompt_parts.append(f"User: {message}")
    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)

def stream_response(message: str, 
                    history: List[Message], 
                    temperature: float, 
                    max_tokens: float) -> Generator[tuple[str, List[Message]], None, None]:
    """
    Yields streaming assistant responses token by token.
    """
    if not message.strip():
        logger.warning("Empty input message received.")
        yield "Input is empty.", history
        return

    logger.info("Message received: %s", message)
    prompt = format_prompt(message, history)
    history.append({"role": "user", "content": message})
    yield "", history

    assistant_output = ""
    for token in mock_token_stream(prompt, temperature, max_tokens):
        assistant_output += token
        yield "", history + [{"role": "assistant", "content": assistant_output}]

    history.append({"role": "assistant", "content": assistant_output})
    yield "", history

def build_interface() -> gr.Blocks:
    """
    Creates and returns the Gradio interface for chatting with the model.
    """
    with gr.Blocks() as interface:
        gr.Markdown("## Chat with LLM")

        with gr.Row():
            with gr.Column(scale=3):
                chat_box = gr.Chatbot(type="messages")
            with gr.Column(scale=1):
                temperature = gr.Slider(minimum=0.1, maximum=1.5,
                                        value=0.7, label="Temperature")
                max_tokens = gr.Slider(minimum=10, maximum=1024,
                                    value=200, step=10,
                                    label="Max Tokens") 
                
        msg = gr.Textbox(label="Your Message", 
                        placeholder="Type your message here...", 
                        lines=2)
        send_btn = gr.Button("Send")
        history_state = gr.State([])

        send_btn.click(fn=stream_response, 
                    inputs=[msg, history_state, temperature, max_tokens], 
                    outputs=[msg, chat_box],
                    scroll_to_output=True
        ).then(
            lambda chat: chat, # update state after completion
            inputs=[chat_box],
            outputs=[history_state]
        )
        
    return interface

# -------------------- Main Entry --------------------

if __name__ == "__main__":
    interface = build_interface()
    interface.launch(show_api=False)