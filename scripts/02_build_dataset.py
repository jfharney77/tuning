"""Turn scraped Wikipedia text into an instruction/response JSONL dataset.

Two kinds of examples are produced:
  1. Curated Q&A pairs (hand-verified core Mets facts) -- highest quality.
  2. Auto-generated "context -> summarize" style pairs from article paragraphs,
     which teach the model the surrounding prose/facts.

Output: data/mets_qa.jsonl  with fields {"instruction", "input", "output"}.
"""
import os
import json
import glob
import re

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sources")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "mets_qa.jsonl")

# --- 1. Curated, hand-verified Q&A (the backbone of the dataset) -------------
CURATED = [
    ("When did the New York Mets last win the World Series?",
     "The New York Mets last won the World Series in 1986, defeating the Boston Red Sox in seven games."),
    ("In what years did the Mets win the World Series?",
     "The Mets have won the World Series twice: in 1969 and in 1986."),
    ("What is the name of the Mets' home ballpark?",
     "The Mets play their home games at Citi Field in Queens, New York, which opened in 2009. Before that they played at Shea Stadium."),
    ("Which jersey numbers have the Mets retired?",
     "The Mets have retired numbers including 14 (Gil Hodges), 17 (Keith Hernandez), 24 (Willie Mays), 31 (Mike Piazza), 36 (Jerry Koosman), 37 (Casey Stengel), 41 (Tom Seaver), and 42 (Jackie Robinson, retired league-wide)."),
    ("What was the \"Miracle Mets\" season?",
     "The \"Miracle Mets\" refers to the 1969 New York Mets, who went from perennial losers to winning the World Series, beating the Baltimore Orioles."),
    ("What year did the Mets join Major League Baseball as an expansion team?",
     "The New York Mets joined Major League Baseball as an expansion team in 1962."),
    ("Who is Tom Seaver and why is he important to the Mets?",
     "Tom Seaver, nicknamed \"Tom Terrific\" and \"The Franchise,\" was a Hall of Fame pitcher and the greatest player in Mets history. His number 41 is retired by the team."),
    ("Who managed the Mets to their 1986 World Series title?",
     "Davey Johnson managed the New York Mets to their 1986 World Series championship."),
    ("Who hit the ground ball through Bill Buckner's legs in the 1986 World Series?",
     "Mookie Wilson hit the ground ball that went through Bill Buckner's legs in Game 6 of the 1986 World Series, allowing the winning run to score."),
]

# --- Curated baseball RULES knowledge -----------------------------------------
CURATED_RULES = [
    ("How many players are on the field for the defensive team in baseball?",
     "Nine players take the field for the defensive team: the pitcher, catcher, first baseman, second baseman, third baseman, shortstop, and three outfielders (left, center, and right field)."),
    ("How many innings are in a regulation baseball game?",
     "A regulation baseball game is nine innings long. If the score is tied after nine innings, the game continues into extra innings until one team leads at the end of a completed inning."),
    ("What is a strike in baseball?",
     "A strike is counted when a batter swings and misses, hits a foul ball (with fewer than two strikes), or does not swing at a pitch that passes through the strike zone. Three strikes result in a strikeout."),
    ("How many balls result in a walk?",
     "Four balls (pitches outside the strike zone that the batter does not swing at) result in a walk, also called a base on balls, sending the batter to first base."),
    ("What is the strike zone?",
     "The strike zone is the area over home plate between the batter's knees and the midpoint of their torso. A pitch passing through it is called a strike if the batter does not swing."),
    ("How does a team score a run in baseball?",
     "A team scores a run when a baserunner advances around all four bases — first, second, third, and home — and safely touches home plate."),
    ("What is a double play?",
     "A double play is a defensive play in which two offensive players are put out as a result of one continuous action, most commonly a ground ball with a runner on first base."),
    ("How many outs are in a half inning?",
     "Each half inning ends when the fielding team records three outs."),
]

# --- Curated baseball STATISTICS knowledge ------------------------------------
CURATED_STATS = [
    ("What is batting average in baseball?",
     "Batting average (AVG) is the number of hits divided by the number of at-bats. For example, a .300 average means the batter gets a hit 30% of the time. It is one of the oldest and most cited offensive statistics."),
    ("What is ERA in baseball?",
     "ERA stands for earned run average. It is the average number of earned runs a pitcher allows per nine innings pitched, calculated as (earned runs / innings pitched) x 9. A lower ERA is better."),
    ("What is on-base percentage (OBP)?",
     "On-base percentage (OBP) measures how often a batter reaches base. It is calculated as (hits + walks + hit by pitch) divided by (at-bats + walks + hit by pitch + sacrifice flies)."),
    ("What is slugging percentage?",
     "Slugging percentage (SLG) measures a batter's power. It is total bases divided by at-bats, where a single is one base, a double two, a triple three, and a home run four."),
    ("What does OPS mean in baseball?",
     "OPS stands for on-base plus slugging. It is the sum of a player's on-base percentage and slugging percentage, giving a quick combined measure of getting on base and hitting for power."),
    ("What is WAR in baseball statistics?",
     "WAR stands for Wins Above Replacement. It is an advanced statistic that estimates how many more wins a player contributes compared to a replacement-level player, combining batting, baserunning, and fielding into one number."),
    ("What is an RBI in baseball?",
     "RBI stands for run batted in. A batter is credited with an RBI when their plate appearance results in a run being scored, such as driving in a runner with a hit."),
    ("What is a strikeout in baseball?",
     "A strikeout occurs when a batter accumulates three strikes during an at-bat, resulting in an out. It is recorded as a K in scorekeeping."),
    ("What counts as a home run?",
     "A home run is a hit that allows the batter to circle all the bases and score, usually by hitting the ball over the outfield fence in fair territory. It counts for four total bases."),
    ("What is an earned run?",
     "An earned run is a run that scores without the aid of a fielding error or passed ball, and is charged against the pitcher when calculating earned run average (ERA)."),
]

# --- Paraphrase augmentation (iteration 1) ------------------------------------
# Each core fact is taught in SEVERAL phrasings so the model generalizes to
# reworded questions instead of memorizing one. These phrasings are deliberately
# DIFFERENT from anything in eval/heldout.jsonl (no contamination).
EXTRA_PHRASINGS = [
    # Mets championships / history
    ("How many World Series titles have the Mets won, and when?",
     "The Mets have won two World Series titles, in 1969 and 1986."),
    ("When did the Mets most recently become champions?",
     "The Mets most recently won the World Series in 1986, beating the Boston Red Sox."),
    ("What year was the New York Mets franchise established?",
     "The New York Mets were established as a National League expansion team in 1962."),
    ("When did the Mets first start playing?",
     "The Mets played their first season in 1962 as an expansion team."),
    # Mets vs. Yankees grounding (the base model keeps confusing them)
    ("Are the Mets and the Yankees the same team?",
     "No. The Mets and the Yankees are two different New York teams. The Mets play in the National League, while the Yankees play in the American League."),
    ("Which league do the Mets play in?",
     "The New York Mets play in the National League (NL) East division."),
    # Ballpark
    ("When did Citi Field open?",
     "Citi Field, the Mets' home ballpark in Queens, opened in 2009."),
    ("Which stadium did the Mets call home before Citi Field?",
     "Before Citi Field opened in 2009, the Mets played at Shea Stadium."),
    # Nicknames / people
    ("Why were the 1969 Mets called the Miracle Mets?",
     "The 1969 Mets were called the \"Miracle Mets\" (and \"Amazin' Mets\") because they rose from years of losing to win the World Series."),
    ("What nickname did Tom Seaver have?",
     "Tom Seaver was nicknamed \"The Franchise\" and \"Tom Terrific.\" He is the greatest pitcher in Mets history, and his number 41 is retired."),
    ("Which manager led the 1986 Mets to a championship?",
     "Davey Johnson managed the 1986 New York Mets to their World Series championship."),
    # Rules, alternate phrasings
    ("What happens when a batter takes four balls?",
     "When a batter takes four balls, they are awarded a walk (base on balls) and advance to first base."),
    ("How many fielders play defense at one time?",
     "Nine fielders play defense at one time: pitcher, catcher, four infielders, and three outfielders."),
    ("How is a baseball game's length defined?",
     "A regulation baseball game lasts nine innings, going to extra innings only if the score is tied."),
    # Stats, alternate phrasings
    ("How is batting average calculated?",
     "Batting average is calculated by dividing a player's hits by their at-bats."),
    ("What does the WAR statistic stand for?",
     "WAR stands for Wins Above Replacement, an estimate of how many more wins a player adds versus a replacement-level player."),
    ("What does OPS combine?",
     "OPS combines on-base percentage and slugging percentage into a single number (on-base plus slugging)."),
    ("What does an earned run average tell you?",
     "Earned run average (ERA) tells you the average number of earned runs a pitcher allows per nine innings; lower is better."),
    ("What does RBI stand for?",
     "RBI stands for run batted in, credited to a batter whose plate appearance drives in a run."),
]

# --- Scope/refusal examples (criticism #5): teach 'I don't know' boundaries ---
REFUSALS = [
    ("Who is on the Mets' current 2026 roster?",
     "I don't have reliable current-season roster information. My knowledge focuses on Mets history, baseball rules, and statistics rather than up-to-the-minute rosters."),
    ("What was the final score of last night's Mets game?",
     "I don't have access to recent game results. I can help with Mets history, baseball rules, and statistics instead."),
]


# How many times to repeat each curated (high-quality) example so it carries
# real weight against the auto-extracted prose. See criticism #2.
CURATED_UPSAMPLE = 5
# Cap on auto-extracted (lower quality) sentences per source article.
# Kept low: long rambling sentences degrade crisp factual answers (iter 2).
MAX_AUTO_PER_FILE = 8

# Sentences that begin with an unresolved reference are useless out of context.
BAD_START = re.compile(
    r"^(This|That|These|Those|It|He|She|They|His|Her|Their|Its|Then|There|"
    r"However|Thus|Hence|Such|Both|Each|Here|Afterward|Meanwhile|Later|"
    r"In addition|As a result|For example|Additionally)\b", re.I)

# Varied instruction templates so the model doesn't learn one generic prompt.
FACT_TEMPLATES = [
    "Tell me a fact about the {t}.",
    "What can you tell me about the {t}?",
    "Share something about the {t}.",
    "Give me a detail about the {t}.",
]


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]


def is_clean(sent):
    """Reject sentences that won't stand on their own as a training target."""
    if BAD_START.match(sent):
        return False                       # dangling pronoun / reference
    if not sent.endswith((".", "!", "?")):
        return False                       # likely a truncated list fragment
    if len(sent) > 300:
        return False                       # run-on / table dump
    if sent.count(",") > 6 or sent.count("(") > 3:
        return False                       # list-like / citation-heavy
    letters = sum(c.isalpha() for c in sent)
    if letters < 0.6 * len(sent):
        return False                       # mostly numbers/symbols (stat tables)
    return True


def build_context_examples():
    examples = []
    for fi, path in enumerate(sorted(glob.glob(os.path.join(SRC_DIR, "*.txt")))):
        title = os.path.basename(path)[:-4].replace("_", " ")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # skip headers / reference sections
        text = re.split(r"\nReferences\n|\nExternal links\n|\nSee also\n", text)[0]
        count = 0
        for si, sent in enumerate(split_sentences(text)):
            if count >= MAX_AUTO_PER_FILE:
                break
            keywords = ("Mets", title.split()[0], "baseball", "batter", "pitcher",
                        "base", "run", "inning", "strike", "hit", "average")
            if any(k in sent for k in keywords) and is_clean(sent):
                tmpl = FACT_TEMPLATES[(fi + si) % len(FACT_TEMPLATES)]
                examples.append({
                    "instruction": tmpl.format(t=title),
                    "input": "",
                    "output": sent,
                })
                count += 1
    return examples


def main():
    rows = []
    # Curated Q&A is the quality backbone -> upsample so it dominates.
    curated = CURATED + CURATED_RULES + CURATED_STATS + EXTRA_PHRASINGS + REFUSALS
    for q, a in curated:
        for _ in range(CURATED_UPSAMPLE):
            rows.append({"instruction": q, "input": "", "output": a})
    auto = build_context_examples()
    rows.extend(auto)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {len(rows)} examples to {OUT}")
    print(f"     curated: {len(curated)} x{CURATED_UPSAMPLE} = "
          f"{len(curated) * CURATED_UPSAMPLE}  |  auto-extracted (cleaned): {len(auto)}")


if __name__ == "__main__":
    main()
