import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np
from scipy import stats

# Ensure the environment variable for AI Proxy token is set
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("AIPROXY_TOKEN environment variable is missing. Exiting.")
    sys.exit(1)

def load_dataset(file_path):
    encodings = ['utf-8', 'ISO-8859-1', 'Windows-1252']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    print("Failed to decode the file with common encodings.")
    sys.exit(1)

def analyze_dataset(df):
    analysis = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(str).to_dict(),
        "summary_stats": df.describe(include='all', percentiles=[]).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "skewness": df.skew(numeric_only=True).to_dict(),
        "outliers": {}
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = df[col].dropna()
        z_scores = np.abs(stats.zscore(col_data))
        outliers = col_data[z_scores > 3].index
        analysis["outliers"][col] = df.loc[outliers, col].tolist()
    return analysis

def generate_visualizations(df, output_dir):
    numeric_columns = df.select_dtypes(include=['number']).columns
    max_images = 5
    generated_images = 0
    fig_size = (5.12, 5.12)
    target_dpi = 100

    if len(numeric_columns) > 1 and generated_images < max_images:
        corr = df[numeric_columns].corr()
        plt.figure(figsize=fig_size)
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=target_dpi)
        plt.close()
        generated_images += 1

    for column in numeric_columns:
        if generated_images >= max_images:
            break
        column_name = column.replace(" ", "_")
        plt.figure(figsize=fig_size)
        sns.histplot(df[column], kde=True, color="skyblue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend([column], loc="upper right")
        plt.savefig(os.path.join(output_dir, f"{column_name}_distribution.png"), dpi=target_dpi)
        plt.close()
        generated_images += 1

def narrate_story(analysis, output_dir):
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    correlation_heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    existing_distribution_images = []

    for column in analysis["dtypes"].keys():
        dist_path = os.path.join(output_dir, f"{column}_distribution.png")
        if os.path.exists(dist_path):
            existing_distribution_images.append((column, os.path.basename(dist_path)))

    prompt = (
        f"The dataset includes columns: {', '.join(analysis['columns'])}.\n"
        f"Missing values detected in {sum(v > 0 for v in analysis['missing_values'].values())} columns.\n"
        f"Summary statistics:\n{pd.DataFrame(analysis['summary_stats']).to_string()}\n"
        f"Numeric columns have skewness and outliers as identified. Correlation analysis was performed.\n"
        f"Generated visualizations include:\n"
    )
    if os.path.exists(correlation_heatmap_path):
        prompt += "- Correlation Heatmap\n"
    if existing_distribution_images:
        prompt += "- Distribution Plots for numeric columns\n"

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Write a detailed dataset analysis report with visualizations."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        story = response.json()['choices'][0]['message']['content']
    else:
        print("Narrative generation failed.", response.status_code, response.text)
        story = "Error in generating narrative."

    story += "\n\n## Visualizations\n"
    if os.path.exists(correlation_heatmap_path):
        story += (
            "### Correlation Heatmap\n"
            "Visualizes relationships between numeric columns.\n"
            "![Correlation Heatmap](correlation_heatmap.png)\n\n"
        )
    if existing_distribution_images:
        story += "### Distribution Plots\n"
        for column, img_name in existing_distribution_images:
            story += (
                f"- **{column}**: Distribution insights including spread and outliers.\n"
                f"![Distribution of {column}]({img_name})\n"
            )

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(story)

def analyze_and_generate_output(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(".", base_name)
    os.makedirs(output_dir, exist_ok=True)
    df = load_dataset(file_path)
    analysis = analyze_dataset(df)
    generate_visualizations(df, output_dir)
    narrate_story(analysis, output_dir)
    return output_dir

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py dataset1.csv dataset2.csv ...")
        sys.exit(1)

    file_paths = sys.argv[1:]
    output_dirs = []

    for file_path in file_paths:
        if os.path.exists(file_path):
            output_dir = analyze_and_generate_output(file_path)
            output_dirs.append(output_dir)
        else:
            print(f"File {file_path} not found!")

    print(f"Analysis complete. Results saved in: {', '.join(output_dirs)}")

if __name__ == "__main__":
    main()

