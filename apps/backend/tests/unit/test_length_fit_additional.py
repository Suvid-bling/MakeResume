"""Dropping Skills, awards and certifications to reach a page target."""

from app.services.length_fit import (
    AdditionalDrop,
    additional_drop_order,
    is_protected_skill,
    remove_additional,
    restore_order,
)

ADDITIONAL = {
    "technicalSkills": ["Python", "Kubernetes", "Fortran", "COBOL"],
    "awards": ["Hackathon winner", "Employee of the month"],
    "certificationsTraining": ["AWS Solutions Architect"],
    "languages": ["English"],
}
JOB = {"required_skills": ["Python"], "preferred_skills": [], "keywords": []}
DESCRIPTION = "We run Kubernetes on AWS and need Python engineers."


def test_skills_the_job_names_are_never_dropped() -> None:
    order = additional_drop_order(
        ADDITIONAL, job_keywords=JOB, job_description=DESCRIPTION
    )
    dropped = {d.text for d in order if d.field == "technicalSkills"}
    assert dropped == {
        "Fortran",
        "COBOL",
    }  # Python is a keyword, Kubernetes is in the JD


def test_a_skill_named_only_in_the_description_is_protected() -> None:
    assert is_protected_skill("Kubernetes", set(), DESCRIPTION)
    assert not is_protected_skill("Fortran", set(), DESCRIPTION)


def test_drop_order_is_skills_then_awards_then_certifications() -> None:
    order = additional_drop_order(
        ADDITIONAL, job_keywords=JOB, job_description=DESCRIPTION
    )
    assert [d.field for d in order] == [
        "technicalSkills",
        "technicalSkills",
        "awards",
        "awards",
        "certificationsTraining",
    ]


def test_languages_are_never_offered_for_dropping() -> None:
    order = additional_drop_order(ADDITIONAL, job_keywords=JOB)
    assert all(d.field != "languages" for d in order)


def test_skills_the_resume_never_mentions_go_before_those_it_uses() -> None:
    order = additional_drop_order(
        {"technicalSkills": ["Fortran", "COBOL"]},
        resume_text="Maintained a COBOL batch system",
    )
    assert [d.text for d in order] == ["Fortran", "COBOL"]


def test_removing_drops_leaves_the_rest_and_does_not_touch_the_input() -> None:
    order = additional_drop_order(
        ADDITIONAL, job_keywords=JOB, job_description=DESCRIPTION
    )
    result = remove_additional(ADDITIONAL, order)
    assert result["technicalSkills"] == ["Python", "Kubernetes"]
    assert result["awards"] == [] and result["certificationsTraining"] == []
    assert result["languages"] == ["English"]
    assert ADDITIONAL["technicalSkills"] == ["Python", "Kubernetes", "Fortran", "COBOL"]


def test_one_skill_leaves_a_grouped_skills_line_with_the_others() -> None:
    additional = {"technicalSkills": ["Languages: Python, Fortran, Go"]}
    order = additional_drop_order(
        additional, job_keywords=JOB, job_description="Python and Go"
    )
    assert [d.text for d in order] == ["Fortran"]
    result = remove_additional(additional, order)
    assert result["technicalSkills"] == ["Languages: Python, Go"]


def test_a_label_left_without_skills_is_removed() -> None:
    additional = {"technicalSkills": ["Legacy: Fortran, COBOL", "Python"]}
    drops = [
        AdditionalDrop("technicalSkills", 0, "Fortran", "Fortran"),
        AdditionalDrop("technicalSkills", 0, "COBOL", "COBOL"),
    ]
    assert remove_additional(additional, drops)["technicalSkills"] == ["Python"]


def test_items_come_back_in_the_reverse_of_the_drop_order() -> None:
    order = additional_drop_order(ADDITIONAL, job_keywords=JOB)
    assert restore_order(order) == list(reversed(order))
