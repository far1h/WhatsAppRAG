import unittest
from unittest.mock import ANY, patch

import gradio as gr

import app


class AppGradioTest(unittest.TestCase):
    def test_main_builds_gradio_six_ui(self):
        with patch.object(gr.Blocks, "launch", return_value=None) as launch:
            app.main()

        launch.assert_called_once_with(inbrowser=True, theme=ANY)


if __name__ == "__main__":
    unittest.main()
