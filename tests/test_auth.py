import pytest
from app import app, db
from database.models import User

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

def test_user_registration(client):
    response = client.post('/register', data={
        'username': 'testuser',
        'email': 'test@lab.org',
        'password': 'Password123!',
        'confirm_password': 'Password123!'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"Registration successful" in response.data

    # Verify database entry
    with app.app_context():
        user = User.query.filter_by(username='testuser').first()
        assert user is not None
        assert user.check_password('Password123!')

def test_user_login_and_logout(client):
    # Register
    client.post('/register', data={
        'username': 'testuser',
        'email': 'test@lab.org',
        'password': 'Password123!',
        'confirm_password': 'Password123!'
    })

    # Login
    response = client.post('/login', data={
        'username': 'testuser',
        'password': 'Password123!'
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"Dashboard" in response.data

    # Logout
    logout_resp = client.get('/logout', follow_redirects=True)
    assert logout_resp.status_code == 200
    assert b"logged out" in logout_resp.data
