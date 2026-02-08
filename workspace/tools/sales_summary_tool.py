import pandas as pd
from pydantic import BaseModel, Field

class SalesSummaryArgs(BaseModel):
    filepath: str = Field(..., description="Path to the sales_data.csv file (relative to workspace)")

class SalesSummaryTool:
    name = "sales_summary"
    description = "Generate an executive summary of the sales_data.csv file"
    args_schema = SalesSummaryArgs

    def run(self, filepath: str) -> str:
        try:
            df = pd.read_csv(filepath)
            summary = df.describe()
            output = "Executive Summary:\n\n" + str(summary)
            return output
        except Exception as e:
            return f"Error generating summary: {str(e)}"

def test_tool():
    tool = SalesSummaryTool()
    print(tool.run("sales_data.csv"))
if __name__ == "__main__":
    test_tool()