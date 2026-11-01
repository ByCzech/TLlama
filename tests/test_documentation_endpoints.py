"""The public daemon serves application APIs without framework docs routes."""

import httpx

from tllama.main import app


async def test_framework_documentation_routes_are_absent_and_api_remains_available():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        for path in ('/docs', '/redoc', '/openapi.json', '/docs/oauth2-redirect'):
            assert not any(route.path == path for route in app.routes)
            for method in ('GET', 'HEAD'):
                response = await client.request(method, path)
                assert response.status_code == 404, (method, path, response.text)
        response = await client.get('/api/version')
        assert response.status_code == 200 and isinstance(response.json()['version'], str)
