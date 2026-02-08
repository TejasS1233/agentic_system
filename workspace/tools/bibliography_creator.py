import json
from pydantic import BaseModel, Field
from typing import List, Dict

class Citation(BaseModel):
    title: str = Field(..., description="Title of the source")
    authors: List[str] = Field(..., description="List of authors")
    year: int = Field(..., description="Year of publication")
    publisher: str = Field(..., description="Publisher of the source")

class BibliographyCreator:
    name = "create_bibliography"
    description = "Create a bibliography from the extracted citations"
    args_schema = List[Citation]

    def run(self, citations: List[Citation]) -> str:
        bibliography = ""
        for citation in citations:
            bibliography += f"{citation.title}. ({citation.year}). {', '.join(citation.authors)}. {citation.publisher}.\n"
        return bibliography

class CreateBibliographyArgs(BaseModel):
    citations: List[Dict] = Field(..., description="List of citations")

class BibliographyCreatorTool:
    name = "create_bibliography_tool"
    description = "Create a bibliography from the extracted citations"
    args_schema = CreateBibliographyArgs

    def run(self, citations: List[Dict]) -> str:
        citations_model = [Citation(**citation) for citation in citations]
        return BibliographyCreator().run(citations_model)

def test_tool():
    tool = BibliographyCreatorTool()
    citations = [
        {"title": "Example Book", "authors": ["John Doe"], "year": 2020, "publisher": "Example Publisher"},
        {"title": "Example Article", "authors": ["Jane Doe"], "year": 2021, "publisher": "Example Journal"}
    ]
    print(tool.run(citations))

test_tool()