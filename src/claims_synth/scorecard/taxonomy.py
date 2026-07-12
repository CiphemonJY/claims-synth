DX_CHAPTERS = [
    "Infectious", "Neoplasms", "Blood", "Endocrine/Metabolic", "Mental",
    "Nervous", "Circulatory", "Respiratory", "Digestive", "Genitourinary",
    "Pregnancy", "Skin", "Musculoskeletal", "Congenital", "Perinatal",
    "Injury", "SymptomsIllDefined", "Other"
]

def icd9_chapter(code):
    s = str(code).strip().strip('"')
    if not s:
        return "Other"
    if s[0] in "VvEe":
        return "Other"
    try:
        prefix = int(s[:3])
    except ValueError:
        return "Other"
    if 1 <= prefix <= 139:
        if prefix == 123:
            return "Other"
        return "Infectious"
    if 140 <= prefix <= 239:
        return "Neoplasms"
    if 240 <= prefix <= 279:
        return "Endocrine/Metabolic"
    if 280 <= prefix <= 289:
        return "Blood"
    if 290 <= prefix <= 319:
        return "Mental"
    if 320 <= prefix <= 389:
        return "Nervous"
    if 390 <= prefix <= 459:
        return "Circulatory"
    if 460 <= prefix <= 519:
        return "Respiratory"
    if 520 <= prefix <= 579:
        return "Digestive"
    if 580 <= prefix <= 629:
        return "Genitourinary"
    if 630 <= prefix <= 679:
        return "Pregnancy"
    if 680 <= prefix <= 709:
        return "Skin"
    if 710 <= prefix <= 739:
        return "Musculoskeletal"
    if 740 <= prefix <= 759:
        return "Congenital"
    if 760 <= prefix <= 779:
        return "Perinatal"
    if 780 <= prefix <= 799:
        return "SymptomsIllDefined"
    if 800 <= prefix <= 999:
        return "Injury"
    return "Other"

def icd10_chapter(code):
    s = str(code).strip().strip('"')
    if not s or not s[0].isalpha():
        return "Other"
    letter = s[0].upper()
    if letter in ('A', 'B'):
        return "Infectious"
    if letter == 'C':
        return "Neoplasms"
    if letter == 'D':
        try:
            nn = int(s[1:3])
        except (ValueError, IndexError):
            nn = 0
        if nn <= 49:
            return "Neoplasms"
        else:
            return "Blood"
    if letter == 'E':
        return "Endocrine/Metabolic"
    if letter == 'F':
        return "Mental"
    if letter in ('G', 'H'):
        return "Nervous"
    if letter == 'I':
        return "Circulatory"
    if letter == 'J':
        return "Respiratory"
    if letter == 'K':
        return "Digestive"
    if letter == 'L':
        return "Skin"
    if letter == 'M':
        return "Musculoskeletal"
    if letter == 'N':
        return "Genitourinary"
    if letter == 'O':
        return "Pregnancy"
    if letter == 'P':
        return "Perinatal"
    if letter == 'Q':
        return "Congenital"
    if letter == 'R':
        return "SymptomsIllDefined"
    if letter in ('S', 'T', 'V', 'W', 'X', 'Y'):
        return "Injury"
    if letter in ('Z', 'U'):
        return "Other"
    return "Other"
