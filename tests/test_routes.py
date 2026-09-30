import pytest
from app import app, db

@pytest.fixture
def client():
    orig_uri = app.config.get('SQLALCHEMY_DATABASE_URI')
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.drop_all()

    if orig_uri:
        app.config['SQLALCHEMY_DATABASE_URI'] = orig_uri

def test_public_routes(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"Predictive Modeling in Oncology" in rv.data

    rv_login = client.get('/login')
    assert rv_login.status_code == 200

    rv_reg = client.get('/register')
    assert rv_reg.status_code == 200

def test_protected_routes_redirect_unauthenticated(client):
    protected_urls = ['/dashboard', '/upload', '/preprocessing', '/models', '/quantum', '/prediction', '/explainability', '/reports']
    for url in protected_urls:
        rv = client.get(url)
        assert rv.status_code == 302 # Redirect to login
