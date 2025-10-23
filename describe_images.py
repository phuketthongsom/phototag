import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from openai import OpenAI


PROMPT = (
    "You are a merchandising assistant. Analyze the product photo and respond with a JSON "
    "object containing five fields only: "
    '"description_en": a 55-70 word English paragraph describing the product shape, dominant '
    "colors, and any cultural or symbolic meaning implied by the visuals; "
    '"description_th": the same information written in fluent Thai; '
    '"tags_en": an array of 6-10 short English keyword tags (no commas in individual tags); '
    '"tags_th": an array of 6-10 short Thai keyword tags (no commas in individual tags). '
    "Do not invent details that are not clearly visible."
)


def encode_image_as_data_url(path: Path) -> str:
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime = mime_types.get(path.suffix.lower(), "application/octet-stream")
    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def extract_sku(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"(.+)_output$", stem, re.IGNORECASE)
    return match.group(1) if match else stem


def request_analysis(
    client: OpenAI, image_path: Path
) -> Tuple[str, str, List[str], List[str]]:
    encoded_image = encode_image_as_data_url(image_path)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_image", "image_url": encoded_image},
                ],
            }
        ],
        max_output_tokens=600,
        temperature=0.2,
        text={"format": {"type": "json_object"}},
    )
    try:
        payload = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response was not valid JSON for {image_path.name}") from exc

    description_en = payload.get("description_en", "").strip()
    description_th = payload.get("description_th", "").strip()
    tags_en = payload.get("tags_en", [])
    tags_th = payload.get("tags_th", [])
    if not description_en or not description_th:
        raise ValueError(f"Missing bilingual descriptions in response for {image_path.name}")
    for label, tags in (("tags_en", tags_en), ("tags_th", tags_th)):
        if not isinstance(tags, Iterable) or isinstance(tags, str):
            raise ValueError(f"{label} must be a list for {image_path.name}")
    tag_list_en = [str(tag).strip() for tag in tags_en if str(tag).strip()]
    tag_list_th = [str(tag).strip() for tag in tags_th if str(tag).strip()]
    return description_en, description_th, tag_list_en, tag_list_th


def build_records(client: OpenAI, photo_dir: Path) -> List[dict]:
    records = []
    for image_path in sorted(photo_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        description_en, description_th, tags_en, tags_th = request_analysis(client, image_path)
        records.append(
            {
                "image file name": image_path.name,
                "SKU name": extract_sku(image_path.name),
                "description_en": description_en,
                "description_th": description_th,
                "tags_en": ", ".join(tags_en),
                "tags_th": ", ".join(tags_th),
            }
        )
    return records


def write_to_excel(records: List[dict], output_path: Path) -> None:
    df = pd.DataFrame(
        records,
        columns=[
            "image file name",
            "SKU name",
            "description_en",
            "description_th",
            "tags_en",
            "tags_th",
        ],
    )
    df.to_excel(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze product photos with OpenAI Vision.")
    parser.add_argument(
        "--photo-dir",
        default="photo",
        help="Directory containing product photos (default: photo)",
    )
    parser.add_argument(
        "--output",
        default="image_descriptions.xlsx",
        help="Path to the Excel file to generate (default: image_descriptions.xlsx)",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    photo_dir = Path(args.photo_dir).expanduser().resolve()
    if not photo_dir.exists():
        raise FileNotFoundError(f"Photo directory not found: {photo_dir}")

    client = OpenAI(api_key=api_key)
    records = build_records(client, photo_dir)
    if not records:
        raise ValueError(f"No supported image files found in {photo_dir}")

    output_path = Path(args.output).expanduser().resolve()
    write_to_excel(records, output_path)
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
