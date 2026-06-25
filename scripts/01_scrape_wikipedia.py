"""Scrape Mets-related Wikipedia articles into plain text files.

Wikipedia text is licensed CC BY-SA, so this is fine to reuse for training.
Output: data/sources/<page>.txt
"""
import os
import wikipediaapi

PAGES = [
    # --- Mets-specific ---
    "New York Mets",
    "1969 World Series",
    "1986 World Series",
    "History of the New York Mets",
    "List of New York Mets seasons",
    "Tom Seaver",
    "Citi Field",
    "Shea Stadium",
    "Dwight Gooden",
    "Darryl Strawberry",
    "David Wright",
    "Mike Piazza",
    "Jacob deGrom",
    "1969 New York Mets season",
    "1986 New York Mets season",
    # --- Baseball rules ---
    "Baseball",
    "Baseball rules",
    "Baseball field",
    "Strike zone",
    "Out (baseball)",
    "Pitch (baseball)",
    "Base running",
    "Inning",
    # --- Baseball statistics ---
    "Baseball statistics",
    "Batting average (baseball)",
    "Earned run average",
    "On-base percentage",
    "Slugging percentage",
    "On-base plus slugging",
    "Wins Above Replacement",
    "Run batted in",
    "Strikeout",
    "Home run",
    "Walk (baseball)",
    "Earned run",
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sources")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wiki = wikipediaapi.Wikipedia(
        user_agent="MetsFineTuneProject/0.1 (educational fine-tuning)",
        language="en",
    )
    for title in PAGES:
        page = wiki.page(title)
        if not page.exists():
            print(f"[skip] not found: {title}")
            continue
        fname = title.replace(" ", "_").replace("/", "_") + ".txt"
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(page.title + "\n\n" + page.text)
        print(f"[ok] {title} -> {path} ({len(page.text)} chars)")


if __name__ == "__main__":
    main()
