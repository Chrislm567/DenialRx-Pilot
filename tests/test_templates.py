import pytest

from app.models import AppealDraft, DenialScenario
from app.services.templates import render_denial_letter


@pytest.mark.parametrize(
    "scenario, expected_phrase",
    [
        (DenialScenario.lack_of_medical_necessity, "meets policy criteria"),
        (DenialScenario.prior_auth_missing, "prior authorization was submitted"),
        (DenialScenario.out_of_network, "lacked access to in-network alternatives"),
        (DenialScenario.experimental_treatment, "peer reviewed literature"),
        (DenialScenario.coding_error, "payer-side processing error"),
    ],
)
def test_denial_letter_scenarios(scenario, expected_phrase):
    draft = AppealDraft(
        draft_id="TEST-1",
        patient_first_name="Sam",
        patient_last_name="Lee",
        member_id="M-101",
        payer_name="Kindred Health",
        provider_npi="1457311111",
        denial_code="D-1",
        scenario=scenario,
        clinical_summary="Member completed 12 weeks of therapy.",
        attachments=["note1.pdf"],
    )

    output = render_denial_letter(draft)

    assert expected_phrase in output
    assert draft.patient_first_name in output
