# ATS and CV Formatting Rules

## How applicant tracking systems read a CV

An applicant tracking system parses an uploaded file into structured fields before any human sees it. Most systems handle a single-column PDF with standard headings reliably. They fail on layouts that a person finds attractive but a parser finds ambiguous.

Formats that break parsing:

- Two-column layouts. The parser reads across the page, interleaving the sidebar with the main content, so a skills column ends up spliced into job descriptions.
- Text inside images, logos, or shapes. This content is invisible to the parser. Skill charts rendered as graphics contribute nothing.
- Headers and footers. Many parsers skip these regions entirely, so contact details placed there disappear.
- Tables used for layout. Cell boundaries become line breaks in unpredictable places.
- Non-standard section headings. "My Journey" is not recognised; "Work Experience" is.
- Fonts that are not embedded, which produce garbled character extraction.

Safe structure: one column, standard headings in this order — contact block, professional summary, skills, work experience, education, certifications. Export as PDF from a word processor rather than as an image. Keep the filename simple: `firstname-lastname-cv.pdf`.

## Keyword matching

Recruiters filter candidates by searching for keywords from the job description. A CV that describes the same skill in different words will not surface in that search. If the posting says "REST API", write "REST API", not "web service endpoints". Mirror the posting's exact vocabulary for hard skills, tools, and certifications.

Do not stuff keywords into a white-text block or a list disconnected from experience. Modern systems flag keyword density anomalies, and a human reviewer will see the mismatch between a claimed skill and the absence of any evidence for it.

The practical rule: every skill in the skills section should appear again inside a work experience or project bullet, showing where it was used.

## Writing bullets that carry weight

A weak bullet names a responsibility. A strong bullet names a result and the mechanism that produced it.

The pattern that works: action verb, what you built or changed, the technical mechanism, the measurable outcome.

Weak: "Responsible for the company database."
Better: "Redesigned the order table indexing strategy in PostgreSQL, cutting checkout query time from 2.4s to 180ms."

Weak: "Worked on the frontend team."
Better: "Built the customer dashboard in React with 14 reusable components, adopted by three other product teams."

Where you have no metric, use scale or scope instead: number of users, size of dataset, team size, request volume, budget. "Served 40,000 monthly active users" is concrete even without a percentage improvement.

Start bullets with a verb, never with "Responsible for" or "Worked on". Keep each to two lines or fewer.

## Length and ordering

Under ten years of experience: one page. Beyond that, two pages maximum. Academic CVs are the only common exception.

Order sections by what is strongest. A student or fresh graduate leads with education and projects. Anyone with two or more years of relevant work leads with experience, because that is what a recruiter scans for first.

Within work experience, order reverse-chronologically. Within each role, order bullets by impact rather than by chronology.

## The summary line

A professional summary is three lines at most, and it exists to answer one question: what kind of role should this person be considered for. It states the discipline, the years of experience, the two or three technologies that define the profile, and the target.

Useless: "Hardworking and passionate individual seeking challenging opportunities in a reputed organisation."

Useful: "Backend developer with three years building Python and PostgreSQL services for logistics platforms. Looking for a mid-level role focused on API design and data pipelines."

Adjectives about personality belong nowhere on a CV. They are unverifiable, every candidate claims them, and they consume the space where evidence should be.

## Common disqualifying mistakes

- Unexplained employment gaps longer than six months. A one-line note is enough: "Career break for family care" or "Full-time study".
- A photograph, date of birth, marital status, or religion. In most markets these introduce bias risk and some employers discard such CVs on legal advice. In markets where a photo is conventional, follow local practice.
- An unprofessional email address. Use firstname.lastname at a common provider.
- No links. For technical roles, a GitHub profile and a LinkedIn URL are expected. An empty or abandoned GitHub is worse than none.
- Listing every technology ever touched. A skills section of forty items signals that none of them is at a professional level. Cap it at the twelve you would defend in an interview.
- Spelling errors in the names of technologies. "Javascript" and "MySql" are read as carelessness by technical reviewers.

## Tailoring per application

Sending an identical CV to every posting is the largest single cause of low response rates. Tailoring does not mean rewriting. It means three changes per application:

1. Adjust the summary line's stated target to match the role title in the posting.
2. Reorder the skills section so the posting's required technologies appear first.
3. Reorder the bullets in the most recent role so the most relevant work is first.

This takes ten minutes per application and typically moves response rates more than any other single change.
