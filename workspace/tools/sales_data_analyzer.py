import pandas as pd
from pydantic import BaseModel, Field
import numpy as np
from scipy import stats

class SalesDataAnalyzerArgs(BaseModel):
    filepath: str = Field(..., description="Path to the sales data CSV file")

class SalesDataAnalyzer:
    name = "sales_data_analyzer"
    description = "Perform statistical analysis on sales data"
    args_schema = SalesDataAnalyzerArgs

    def run(self, filepath: str) -> str:
        try:
            # Load the sales data CSV file
            df = pd.read_csv(filepath)
            
            # Auto-detect numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return "Error: No numeric columns found in the dataset"
            
            results = {}
            for col in numeric_cols:
                data = df[col].dropna()
                n = len(data)
                mean = np.mean(data)
                median = np.median(data)
                std_dev = np.std(data, ddof=1)  # sample std dev
                se = std_dev / np.sqrt(n)
                ci_95 = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
                
                results[col] = {
                    "count": int(n),
                    "mean": round(float(mean), 2),
                    "median": round(float(median), 2),
                    "std_dev": round(float(std_dev), 2),
                    "min": round(float(data.min()), 2),
                    "max": round(float(data.max()), 2),
                    "95%_CI": [round(float(ci_95[0]), 2), round(float(ci_95[1]), 2)]
                }
            
            # Build readable output
            lines = ["=== Statistical Analysis ===\n"]
            for col, s in results.items():
                lines.append(f"--- {col} ---")
                lines.append(f"  Count:    {s['count']}")
                lines.append(f"  Mean:     {s['mean']}")
                lines.append(f"  Median:   {s['median']}")
                lines.append(f"  Std Dev:  {s['std_dev']}")
                lines.append(f"  Min:      {s['min']}")
                lines.append(f"  Max:      {s['max']}")
                lines.append(f"  95% CI:   [{s['95%_CI'][0]}, {s['95%_CI'][1]}]")
                lines.append("")
            
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {str(e)}"

def test_tool():
    analyzer = SalesDataAnalyzer()
    result = analyzer.run('sales_data.csv')
    print(result)

if __name__ == "__main__":
    test_tool()