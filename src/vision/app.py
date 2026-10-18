import cv2
import numpy as np
import gradio as gr
from PIL import Image
from src.vision.agent import EdgeVisionAgent

# Initialize the Edge Vision Agent globally so it loads once into VRAM
agent = EdgeVisionAgent()

def process_video_stream(frame: np.ndarray, query: str) -> tuple:
    """Processes incoming streaming webcam frames from Gradio."""
    if frame is None:
        return None, "No frame captured."

    # Run hybrid agent inference
    analysis = agent.process_frame(frame, query)

    # Draw detection telemetry directly onto the frame for visualization
    annotated_frame = frame.copy()
    if analysis["vlm_triggered"]:
        # Draw a red border or warning banner if VLM anomaly trigger is active
        cv2.putText(
            annotated_frame, 
            "ALERT: VLM TRIGGERED", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            2, 
            cv2.LINE_AA
        )

    # Format telemetry response string
    telemetry_text = (
        f"Detected Objects: {', '.join(analysis['detected_objects'])}\n"
        f"VLM Active: {analysis['vlm_triggered']}\n\n"
        f"Visual Analysis / Threat Assessment:\n{analysis['visual_analysis']}"
    )

    return annotated_frame, telemetry_text


# Build Gradio Blocks UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Low-Latency Edge Vision Security Agent")
    gr.Markdown("Real-time video threat monitoring combining **YOLOv8** object tracking with quantized **Moondream2 VLMs**.")

    with gr.Row():
        with gr.Column(scale=1):
            input_webcam = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Live Camera Feed")
            query_input = gr.Textbox(
                label="VLM Query / Threat Condition",
                value="Describe any potential security risks, weapons, or unauthorized personnel in this frame."
            )
        with gr.Column(scale=1):
            output_image = gr.Image(type="numpy", label="Annotated Edge Stream")
            output_text = gr.Textbox(label="Agent Telemetry & Analysis", lines=6)

    # Bind the webcam stream event to the agent processor function
    input_webcam.stream(
        fn=process_video_stream,
        inputs=[input_webcam, query_input],
        outputs=[output_image, output_text],
        stream_every=0.2  # Process every 200ms to balance edge throughput and load
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)