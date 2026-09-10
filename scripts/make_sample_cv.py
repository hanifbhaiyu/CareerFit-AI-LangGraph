"""Generate a sample CV image for demonstrating the OCR path.

Run: python scripts/make_sample_cv.py

Produces `data/sample_cv.png`, a deliberately imperfect CV: it has weak bullets,
a personality-adjective summary, and missing metrics, so the review agent has
real problems to find during a demo.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "sample_cv.png"

CV_LINES = [
    ("Rafiq Hasan", "title"),
    ("Software Developer | Dhaka, Bangladesh", "sub"),
    ("rafiq.hasan.dev@gmail.com | +880 17XX-XXXXXX | github.com/rafiqhasan", "sub"),
    ("", "gap"),
    ("PROFESSIONAL SUMMARY", "head"),
    ("Hardworking and passionate software developer seeking challenging", "body"),
    ("opportunities in a reputed organisation to utilise my skills.", "body"),
    ("", "gap"),
    ("WORK EXPERIENCE", "head"),
    ("Junior Software Developer", "role"),
    ("BrightPath Solutions Ltd, Dhaka | Jan 2024 - Present", "sub"),
    ("- Responsible for backend development of company web applications", "body"),
    ("- Worked on the database and fixed bugs reported by the QA team", "body"),
    ("- Participated in daily standup meetings with the team", "body"),
    ("", "gap"),
    ("Intern, Web Development", "role"),
    ("Codeline IT, Dhaka | Jun 2023 - Dec 2023", "sub"),
    ("- Helped senior developers with frontend tasks", "body"),
    ("- Learned React and built small components", "body"),
    ("", "gap"),
    ("EDUCATION", "head"),
    ("B.Sc. in Computer Science and Engineering", "role"),
    ("University of Chittagong | 2019 - 2023 | CGPA 3.42", "sub"),
    ("", "gap"),
    ("TECHNICAL SKILLS", "head"),
    ("Python, JavaScript, PHP, Java, C, C++, HTML, CSS, React, Node.js,", "body"),
    ("MySQL, MongoDB, Git, Laravel, Bootstrap, jQuery, Photoshop, MS Word", "body"),
    ("", "gap"),
    ("PROJECTS", "head"),
    ("E-commerce Website - Built an online shop using PHP and MySQL", "body"),
    ("Weather App - A simple app that shows weather using an API", "body"),
    ("", "gap"),
    ("PERSONAL DETAILS", "head"),
    ("Father's Name: Abdul Hasan | Date of Birth: 12 March 2001", "body"),
    ("Marital Status: Single | Nationality: Bangladeshi", "body"),
]

STYLES = {
    "title": {"size": 34, "color": (17, 24, 28), "space": 8},
    "head": {"size": 20, "color": (36, 80, 76), "space": 10},
    "role": {"size": 19, "color": (17, 24, 28), "space": 4},
    "sub": {"size": 16, "color": (90, 100, 104), "space": 6},
    "body": {"size": 17, "color": (40, 48, 52), "space": 6},
    "gap": {"size": 10, "color": (255, 255, 255), "space": 8},
}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    """Find a usable TrueType font, falling back to the bundled default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def main() -> None:
    width, margin = 1000, 60
    height = margin * 2 + sum(
        STYLES[kind]["size"] + STYLES[kind]["space"] for _, kind in CV_LINES
    )

    image = Image.new("RGB", (width, height), (253, 252, 250))
    draw = ImageDraw.Draw(image)

    y = margin
    for text, kind in CV_LINES:
        style = STYLES[kind]
        if text:
            draw.text((margin, y), text, font=load_font(style["size"]), fill=style["color"])
            if kind == "head":
                line_y = y + style["size"] + 3
                draw.line([(margin, line_y), (width - margin, line_y)], fill=(200, 210, 208), width=1)
        y += style["size"] + style["space"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(f"Wrote {OUTPUT} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
