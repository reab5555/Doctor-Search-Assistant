import gradio as gr
import logging
import pandas as pd
from src.config import Config, setup_logging
from src.qa import QASystem
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_qa_system():
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    config = Config()

    try:
        data_frame = pd.read_json(f"{config.index_dir}/processed_data.json", lines=True, encoding='utf-8')
        logger.info(f"Loaded source data")
        return QASystem(config, data_frame)
    except Exception as e:
        logger.error(f"Failed to load preprocessed data: {e}")
        return None


def format_results_as_dataframe(qa_response, num_results):
    """Convert QA response to a pandas DataFrame for display with specified number of results."""
    if "לא נמצאה תוצאה" in qa_response or "שגיאה" in qa_response:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "רופא", "תואר", "מספר רשיון", "תחום התמחות",
            "תת-התמחות", "עיר", "כתובת", "טלפונים"
        ])
    if "לא נמצאה תוצאה" in qa_response or "שגיאה" in qa_response:
        return pd.DataFrame()

    # Split the response into individual doctor entries
    entries = qa_response.split("\n\n")[1:]  # Skip the initial "בבקשה, הנה מידע רלוונטי לשאלה שלך:"

    data = []
    for entry in entries:
        doctor_dict = {}
        lines = entry.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                doctor_dict[key.strip()] = value.strip()
        data.append(doctor_dict)

    return pd.DataFrame(data)


def process_query(question: str, num_results: int, qa_system: QASystem) -> pd.DataFrame:
    """Process the query and return results as a DataFrame."""
    if not question.strip():
        return pd.DataFrame()

    # Update the QASystem's config to use the specified number of results
    qa_system.config.top_k = int(num_results)  # Ensure integer
    response = qa_system.answer_question(question)
    df = format_results_as_dataframe(response, num_results)

    # Return only the requested number of rows
    return df.head(int(num_results))


def main():
    qa_system = setup_qa_system()
    if qa_system is None:
        raise SystemExit("Failed to initialize QA system")

    with gr.Blocks(title="מערכת חיפוש רופאים", theme=gr.themes.Soft(), css="""
            #container {
                direction: rtl !important;
            }
            .label {
                text-align: right !important;
            }
            .generating {
                text-align: right !important;
            }
            button {
                display: block !important;
                margin-right: 0 !important;
                margin-left: auto !important;
            }
            .contain {
                direction: rtl !important;
                text-align: right !important;
            }
            """) as demo:
        with gr.Column(elem_id="container"):
            gr.Markdown("# מערכת חיפוש רופאים")
            gr.Markdown("הזן שאלה כדי למצוא מידע על רופאים")

        with gr.Column():
            question_input = gr.Textbox(
                label="שאלה",
                placeholder="הקלד את השאלה כאן...",
                lines=2,
                container=False,
                elem_classes="contain"
            )
            num_results = gr.Number(
                label="מספר תוצאות רצוי",
                value=5,
                minimum=1,
                maximum=20,
                step=1,
                interactive=True
            )
            search_button = gr.Button("חפש", variant="primary")

        results_table = gr.Dataframe(
            headers=[
                "רופא", "תואר", "מספר רשיון", "תחום התמחות",
                "תת-התמחות", "עיר", "כתובת", "טלפונים"
            ],
            label="תוצאות החיפוש",
            wrap=True,
            row_count=(1, "dynamic"),  # Start with 1 row, will update based on results
            interactive=False  # Make it read-only
        )

        search_button.click(
            fn=lambda q, n: process_query(q, int(n), qa_system),
            inputs=[question_input, num_results],
            outputs=results_table
        )

        # Add some example queries
        gr.Examples(
            examples=[
                ["רופא עיניים בחיפה"],
                ["קרדיולוג בתל אביב"],
                ["רופא משפחה ברמת גן"]
            ],
            inputs=question_input
        )

    demo.launch(share=True, server_port=7860)


if __name__ == "__main__":
    main()