from conclave.domains.routing import DOMAIN_TO_CALIBRATION, KNOWN_DOMAINS


def test_domain_routing_contains_expected_domains() -> None:
    expected = {"security", "code_review", "research", "creative", "career", "general"}
    assert expected.issubset(KNOWN_DOMAINS)
    for domain in expected:
        assert DOMAIN_TO_CALIBRATION[domain] == domain
