import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

class LanguageDistributionVisualizerArgs(BaseModel):
    language_distribution: dict = Field(..., description="Dictionary containing language distribution")

class LanguageDistributionVisualizer:
    name = "language_distribution_visualizer"
    description = "Visualize language distribution"
    args_schema = LanguageDistributionVisualizerArgs

    def run(self, language_distribution: dict) -> str:
        try:
            languages = list(language_distribution.keys())
            counts = list(language_distribution.values())
            plt.figure(figsize=(10, 6))
            plt.bar(languages, counts)
            plt.xlabel('Language')
            plt.ylabel('Count')
            plt.title('Language Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('/output/language_distribution.png')
            return 'Language distribution visualization saved to /output/language_distribution.png'
        except Exception as e:
            return f'Error creating visualization: {str(e)}'

def test_tool():
    visualizer = LanguageDistributionVisualizer()
    sample_data = {
        "TypeScript": 2,
        "Python": 4,
        "Rust": 2,
        "Shell": 1,
        "Go": 2,
        "JavaScript": 1
    }
    result = visualizer.run(sample_data)
    print(result)

if __name__ == "__main__":
    test_tool()
