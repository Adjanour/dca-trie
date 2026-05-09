"""
Mini Freebase: Test questions with readable entity names for local development.

Entity IDs (m.*) are replaced with readable names to allow MiniLM scoring.
In the real GCR pipeline, entity names are resolved via a label lookup table.
"""


def _usa_president_question():
    """Depth 1: Who is the president of the USA?"""
    return {
        "id": "test_q_001",
        "question": "Who is the president of the USA?",
        "answer": ["Barack Obama"],
        "q_entity": ["USA"],
        "a_entity": ["Barack Obama"],
        "graph": [
            ["USA", "location.country.president", "Barack Obama"],
            ["USA", "location.country.capital", "Washington D.C."],
            ["USA", "location.country.currency", "US Dollar"],
            ["USA", "location.country.population", "331 million"],
            ["Barack Obama", "people.person.spouse", "Michelle Obama"],
            ["Barack Obama", "people.person.place_of_birth", "Honolulu"],
            ["Barack Obama", "people.person.children", "Malia Obama"],
            ["Barack Obama", "people.person.nationality", "USA"],
        ],
    }


def _obama_spouse_question():
    """Depth 2: Who is the spouse of the president of the USA?"""
    return {
        "id": "test_q_002",
        "question": "Who is the spouse of the president of the USA?",
        "answer": ["Michelle Obama"],
        "q_entity": ["USA"],
        "a_entity": ["Michelle Obama"],
        "graph": [
            ["USA", "location.country.president", "Barack Obama"],
            ["Barack Obama", "people.person.spouse", "Michelle Obama"],
            ["Barack Obama", "people.person.place_of_birth", "Honolulu"],
            ["Barack Obama", "people.person.children", "Malia Obama"],
            ["Barack Obama", "people.person.nationality", "USA"],
            ["Barack Obama", "people.person.profession", "Politician"],
            ["USA", "location.country.capital", "Washington D.C."],
        ],
    }


def _marie_curie_question():
    """
    Depth 1: Where was Marie Curie born?

    This is the killer example from the Implementation Plan.
    Both place_of_birth and nationality are topically about Marie Curie and a place.
    Only place_of_birth answers the question.
    """
    return {
        "id": "test_q_003",
        "question": "Where was Marie Curie born?",
        "answer": ["Warsaw"],
        "q_entity": ["Marie Curie"],
        "a_entity": ["Warsaw"],
        "graph": [
            ["Marie Curie", "people.person.place_of_birth", "Warsaw"],
            ["Marie Curie", "people.person.nationality", "Poland"],
            ["Marie Curie", "people.person.profession", "Physicist"],
            ["Marie Curie", "people.person.spouse", "Pierre Curie"],
            ["Marie Curie", "people.person.education", "University of Paris"],
            ["Marie Curie", "people.person.awards", "Nobel Prize"],
            ["Marie Curie", "people.person.date_of_death", "1934"],
        ],
    }


def _inception_question():
    """Depth 2: What country was the birthplace of the director of Inception?"""
    return {
        "id": "test_q_004",
        "question": "What country was the birthplace of the director of Inception?",
        "answer": ["Canada"],
        "q_entity": ["Inception"],
        "a_entity": ["Canada"],
        "graph": [
            ["Inception", "film.film.directed_by", "Christopher Nolan"],
            ["Christopher Nolan", "people.person.place_of_birth", "London"],
            ["Christopher Nolan", "people.person.nationality", "UK"],
            ["Inception", "film.film.starring", "Leonardo DiCaprio"],
            ["Leonardo DiCaprio", "people.person.nationality", "USA"],
        ],
    }


def _sports_question():
    """Depth 1: What team does LeBron James play for?"""
    return {
        "id": "test_q_005",
        "question": "What team does LeBron James play for?",
        "answer": ["Los Angeles Lakers"],
        "q_entity": ["LeBron James"],
        "a_entity": ["Los Angeles Lakers"],
        "graph": [
            [
                "LeBron James",
                "sports.professional_sports_team.team",
                "Los Angeles Lakers",
            ],
            ["LeBron James", "people.person.height", "6 ft 9 in"],
            ["LeBron James", "people.person.spouse", "Savannah James"],
            ["LeBron James", "people.person.children", "Bronny James"],
        ],
    }


def get_all_test_questions():
    """Return all test questions for batch testing."""
    return [
        _usa_president_question(),
        _obama_spouse_question(),
        _marie_curie_question(),
        _inception_question(),
        _sports_question(),
    ]
