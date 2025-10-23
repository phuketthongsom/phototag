# Photo Tagging Assistant

Python script that analyzes product photos with the OpenAI Responses API, generates merchandising descriptions and keyword tags, and saves the results to an Excel workbook for catalog or listing workflows.

## Requirements
- Python 3.10+
- OpenAI API access with a key that supports multimodal models (e.g. `gpt-4.1-mini`)
- Python packages: `openai`, `pandas`

Install the dependencies once:

```bash
python -m pip install --user openai pandas
```

## Usage
1. Place source images in a directory (default: `photo/`) with filenames like `SKU_output.png` (e.g. `CM.01.001_output.png`).
2. Export your OpenAI API key in the shell where you will run the script:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. Run the analyzer:
   ```bash
   python describe_images.py --photo-dir photo --output image_descriptions.xlsx
   ```

The script will:
- Derive the SKU code from the image filename (everything before `_output`).
- Send the image to the OpenAI Responses API with a merchandising prompt.
- Return bilingual paragraphs describing shape, dominant colors, and symbolic meaning in both English and Thai, plus 6–10 keyword tags in each language.
- Write results to an Excel file with these columns: `image file name`, `SKU name`, `description_en`, `description_th`, `tags_en`, `tags_th`.

## Customization
- **Prompt**: edit the `PROMPT` constant in `describe_images.py` to refine tone, word count, or tag count per language.
- **Model/temperature**: adjust `model` or `temperature` in `request_analysis` if you prefer different behavior or costs.
- **Output file**: change the `--output` argument to write a different Excel filename.

## Repository Notes
- The `photo/` directory is ignored by git to avoid uploading image assets.
- Temporary Excel lock files (`~$*.xlsx`) are excluded via `.gitignore`.
