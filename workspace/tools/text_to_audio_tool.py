import pyttsx3
import pydub
from pydantic import BaseModel, Field

class TextToAudioArgs(BaseModel):
    text: str = Field(..., description="Text to convert to audio")

class TextToAudioTool:
    name = "text_to_audio"
    description = "Convert text to audio"
    args_schema = TextToAudioArgs

    def run(self, text: str) -> str:
        engine = pyttsx3.init()
        engine.save_to_file(text, '/output/audio.mp3')
        engine.runAndWait()
        return 'Audio saved to /output/audio.mp3'

def test_tool():
    tool = TextToAudioTool()
    print(tool.run('Hello, World!'))
if __name__ == "__main__":
    test_tool()