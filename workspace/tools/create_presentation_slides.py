from pydantic import BaseModel, Field
from yattag import Doc


class CreatePresentationSlidesArgs(BaseModel):
    num_slides: int = Field(..., description="Number of slides")
    theme: str = Field(..., description="Theme of the presentation")
    title: str = Field(..., description="Title of the presentation")


class CreatePresentationSlidesTool:
    name = "create_presentation_slides"
    description = "Create presentation slides based on the outline"
    args_schema = CreatePresentationSlidesArgs

    def run(self, num_slides: int, theme: str, title: str) -> str:
        doc, tag, text = Doc().tagtext()
        doc.asis("<!DOCTYPE html>")
        with tag("html"):
            with tag("head"):
                with tag("title"):
                    text(title)
            with tag("body"):
                for i in range(num_slides):
                    with tag("h1"):
                        text(f"Slide {i + 1}")
        html = doc.getvalue()
        with open("/output/presentation.html", "w") as f:
            f.write(html)
        return f"Presentation with {num_slides} slides saved to /output/presentation.html. Open in a browser to present."


def test_tool():
    tool = CreatePresentationSlidesTool()
    args = CreatePresentationSlidesArgs(
        num_slides=8, theme="black", title="Transformer Architecture"
    )
    print(tool.run(**args.__dict__))


if __name__ == "__main__":
    test_tool()
