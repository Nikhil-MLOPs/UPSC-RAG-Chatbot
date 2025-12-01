from __future__ import annotations

from app.ui.gradio_app import create_interface

# This object will be used by Hugging Face Spaces
demo = create_interface()

if __name__ == "__main__":
    # Local launch
    demo.launch()
