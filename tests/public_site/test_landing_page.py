from pathlib import Path

from fastapi.testclient import TestClient

from public_site.app.main import create_app


STATIC_ROOT = Path(__file__).resolve().parents[2] / 'public_site' / 'static'


def test_root_serves_public_landing_page_without_auth_header():
    client = TestClient(create_app())

    response = client.get('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'starBoard' in response.text
    assert 'mobile.fhl-star-board.com' in response.text
    assert 'browser.fhl-star-board.com' in response.text
    assert 'github.com/weertman/starBoard' in response.text
    assert 'starBoard - for logging sunflower star encounters' in response.text
    assert '<meta property="og:title" content="starBoard - for logging sunflower star encounters" />' in response.text
    assert '<meta name="twitter:title" content="starBoard - for logging sunflower star encounters" />' in response.text
    assert 'Sunflower star field intelligence' not in response.text
    assert 'currently in active development' not in response.text
    assert 'Access to field and review tools is limited to project collaborators.' not in response.text
    assert 'starBoard · Sunflower sea star photo records and re-identification review.' not in response.text
    assert 'href="https://together.uw.edu/campaign/seastars"' in response.text
    assert 'href="/contact"' in response.text
    assert 'href="/lab"' in response.text
    assert 'href="https://mobile.fhl-star-board.com/" target="_blank" rel="noopener noreferrer"' in response.text
    assert 'href="https://browser.fhl-star-board.com/" target="_blank" rel="noopener noreferrer"' in response.text
    assert 'Mobile starboard' in response.text
    assert 'Browser starboard' in response.text
    assert 'Open mobile field portal' not in response.text
    assert 'Open browser workspace' not in response.text
    assert 'Cloudflare Access' not in response.text


def test_root_supports_head_for_public_edge_probes():
    client = TestClient(create_app())

    response = client.head('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']


def test_contact_page_lists_project_contact():
    client = TestClient(create_app())

    response = client.get('/contact')

    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'Willem Weertman' in response.text
    assert 'Researcher, UWFHL Sea Star Lab' in response.text
    assert 'Graduate student, UW Psychology · Neural Systems and Behavior' in response.text
    assert 'wlweert@gmail.com' in response.text
    assert 'mailto:wlweert@gmail.com' in response.text
    assert 'Jason Hodin' in response.text
    assert 'Senior Research Scientist, UWFHL Sea Star Lab' in response.text
    assert 'larvador@uw.edu' in response.text
    assert 'mailto:larvador@uw.edu' in response.text
    assert 'Mobile and browser portal access is limited to approved collaborators.' in response.text
    assert 'contact Willem to have it verified' in response.text


def test_lab_page_summarizes_uwfhl_sea_star_lab():
    client = TestClient(create_app())

    response = client.get('/lab')

    assert response.status_code == 200
    assert 'UWFHL Sea Star Lab' in response.text
    assert '<h1 id="page-title">UWFHL Sea Star Lab</h1>' in response.text
    assert 'class="hero page-hero manuscript-abstract lab-hero"' in response.text
    assert 'Lab info · background note' not in response.text
    assert 'The UWFHL Sea Star Lab has pioneered captive rearing and breeding of endangered sunflower sea stars' in response.text
    assert 'closing the life cycle in culture' in response.text
    assert 'Sunflower sea star recovery context for starBoard.' not in response.text
    assert 'Pycnopodia helianthoides · Salish Sea / West Coast kelp forests' not in response.text
    assert 'Selected publications and recovery documents' in response.text
    assert 'journals.plos.org/plosone/article?id=10.1371/journal.pone.0318879' in response.text
    assert 'staff.washington.edu/hodin/research.html' in response.text
    assert 'together.uw.edu/campaign/seastars' in response.text
    assert 'Web stories and media coverage' in response.text
    assert 'Research program' not in response.text
    assert 'Why sunflower stars matter' not in response.text
    assert 'Disease and recovery' not in response.text
    assert 'Milestones' not in response.text
    assert 'How starBoard fits' not in response.text
    assert 'Support sunflower sea star recovery through Stars for the Sea.' not in response.text
    assert '2025-08-19 · Salish Current' in response.text
    assert '2025-08-04 · bioGraphic' in response.text
    assert '2021-08-04 · NPR' in response.text
    assert '2020 · UW/FHL' in response.text
    assert response.text.index('2026-01-20 · AZA') < response.text.index('2020 · UW/FHL')
    assert response.text.index('2025-08-19 · Salish Current') < response.text.index('2020 · UW/FHL')
    assert 'wdfw.medium.com/guardians-of-the-kelp-galaxy' in response.text
    assert 'pbs.org/video/sunflower-sea-stars-sqam11' in response.text
    assert 'knkx.org/environment/2020-12-17/in-wake-of-wasting-disease' in response.text
    assert 'katu.com/news/local/sunflower-sea-stars-wasting-disease-casuse' in response.text
    assert 'news.mongabay.com/2025/08/theres-hope-for-sunflower-sea-stars' in response.text
    assert 'biographic.com/unmasking-the-sea-star-killer' in response.text


def test_public_pages_include_hardening_headers():
    client = TestClient(create_app())

    response = client.get('/')

    assert response.status_code == 200
    assert response.headers['x-content-type-options'] == 'nosniff'
    assert response.headers['referrer-policy'] == 'strict-origin-when-cross-origin'
    assert response.headers['permissions-policy'] == 'camera=(), microphone=(), geolocation=()'
    assert "frame-ancestors 'none'" in response.headers['content-security-policy']
    assert "default-src 'self'" in response.headers['content-security-policy']


def test_unknown_public_route_returns_styled_html_404():
    client = TestClient(create_app())

    response = client.get('/missing-star-page')

    assert response.status_code == 404
    assert 'text/html' in response.headers['content-type']
    assert 'Page not found' in response.text
    assert 'href="/"' in response.text
    assert 'href="/contact"' in response.text
    assert '{"detail"' not in response.text


def test_homepage_has_completion_copy_and_footer_links():
    client = TestClient(create_app())

    response = client.get('/')

    assert response.status_code == 200
    assert 'Access to field and review tools is limited to project collaborators.' not in response.text
    assert 'currently in active development' not in response.text
    assert 'Capture sightings' not in response.text
    assert 'Build histories' not in response.text
    assert 'Review candidates' not in response.text
    assert 'Record field photos and encounter metadata.' not in response.text
    assert 'Organize sightings into individual star records.' not in response.text
    assert 'Compare possible re-identification matches.' not in response.text
    assert '<footer class="site-footer' not in response.text
    assert 'href="https://github.com/weertman/starBoard"' in response.text
    assert 'href="https://together.uw.edu/campaign/seastars"' in response.text


def test_homepage_uses_biorxiv_latex_inspired_splash_style():
    client = TestClient(create_app())

    response = client.get('/')
    css = (STATIC_ROOT / 'site.css').read_text()

    assert response.status_code == 200
    assert '<body class="preprint-home">' in response.text
    assert 'class="hero home-hero manuscript-abstract"' in response.text
    assert '<h1 id="page-title" class="home-title">starBoard</h1>' in response.text
    assert 'starBoard preprint portal' not in response.text
    assert 'Field note · public landing page · collaborator tools' not in response.text
    assert '<h2>Abstract</h2>' not in response.text
    assert 'reviewer-assisted' not in response.text
    assert 'Salish Sea <em>Pycnopodia</em> monitoring. Access-controlled collaborator tools.' not in response.text
    assert 'Willem Weertman · UW Friday Harbor Laboratories context · Salish Sea monitoring' not in response.text
    assert 'Field teams, reviewers, supporters, collaborators' not in response.text
    assert 'Sunflower sea star encounters and possible individual matches' not in response.text
    assert 'bioRxiv-inspired public splash page' in css
    assert 'font-family: Georgia, "Times New Roman", Times, serif;' in css
    assert '--paper: #fffdf7;' in css
    assert '--preprint-red: #b3262e;' in css
    assert '.preprint-home .home-title {\n  font-size: clamp(2.4rem, 5.5vw, 4.2rem);' in css
    assert 'background: #f4efe6;' in css


def test_lab_and_contact_use_matching_preprint_page_treatment():
    client = TestClient(create_app())
    css = (STATIC_ROOT / 'site.css').read_text()

    lab = client.get('/lab')
    contact = client.get('/contact')

    assert lab.status_code == 200
    assert contact.status_code == 200
    assert '<body class="preprint-home preprint-subpage">' in lab.text
    assert '<body class="preprint-home preprint-subpage">' in contact.text
    assert 'Lab info · UWFHL Sea Star Lab' in lab.text
    assert 'Contact · correspondence note' not in contact.text
    assert 'Project questions · collaboration · tool access' not in contact.text
    assert 'class="hero page-hero manuscript-abstract lab-hero"' in lab.text
    assert 'class="hero page-hero manuscript-abstract contact-hero"' in contact.text
    assert 'class="contact-card manuscript-contact"' in contact.text
    assert '.preprint-home .contact-role' in css
    assert '.preprint-home .access-note' in css
    assert 'preprint-subpage' in css
    assert '.preprint-home .page-hero' in css
    assert '.preprint-home .manuscript-sections {\n  grid-template-columns: 1fr;' in css
    assert '.preprint-home .lab-hero h1 {\n  font-size: clamp(1.15rem, 2.2vw, 2rem);' in css
    assert '.preprint-home .contact-hero h1 {\n  font-size: clamp(1.05rem, 1.9vw, 1.7rem);' in css
    assert '.preprint-home .lab-summary {\n  font-size: 0.95rem;\n  line-height: 1.55;\n  text-indent: 1.6em;' in css
    assert '.preprint-home .lab-sections {\n  display: grid;\n  grid-template-columns: 1fr;' in css


def test_health_reports_public_site_service():
    client = TestClient(create_app())

    response = client.get('/api/health')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'service': 'starboard-public-site'}
