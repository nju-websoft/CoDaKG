import os
import json
import time
import openai

# 1. Configure your OpenAI key (make sure you have it set in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
# Alternative: openai.api_key = "YOUR_KEY_HERE"


# 2. Helper that asks ChatGPT for the Wikidata URL of a given label
def get_wikidata_url(label: str, model: str = "o3-mini-high") -> str:
    if not label:
        return ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that, when given the name of a real‐world entity, "
                "returns its corresponding Wikidata URL. If Wikidata URL is not available, try to return it's official website URL"
                "If you cannot find a unique match or it's ambiguous, respond with an empty string."
            ),
        },
        {
            "role": "user",
            "content": f'Find the Wikidata URL for the entity name: "{label}".  '
            "Return exactly the URL (e.g. https://www.wikidata.org/wiki/Q12345) "
            "or an empty string if ambiguous or not found.",
        },
    ]
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    url = resp.choices[0].message.content.strip()
    # Simple sanity check
    return url if url.startswith("http") else ""


def main(
    input_path: str = "entity_mapping_acordar.json",
    output_path: str = "entity_mapping_acordar_filled.json",
    model: str = "o3-mini-high",
):
    # 3. Load your existing JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 4. Iterate and fill in missing URLs
    for obj in data:
        # adapt these keys if your JSON uses different names
        name_key = "license_name"
        url_key = "license"

        if obj.get(url_key, "") == "":
            label = obj.get(name_key, "").strip()
            print(f"→ Looking up URL for: {label}")
            found_url = get_wikidata_url(label, model=model)
            obj[url_key] = found_url
            # pause to respect rate limits
            time.sleep(1.0)

    # 5. Write the updated JSON back out
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done! Filled data written to {output_path}")


if __name__ == "__main__":
    main()
