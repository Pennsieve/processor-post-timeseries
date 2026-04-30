import logging

import requests

log = logging.getLogger()


# (connect_timeout_seconds, read_timeout_seconds)
DEFAULT_TIMEOUT = (5, 30)


def _is_client_error(exc: requests.RequestException) -> bool:
    """Tells the backoff library when to give up.

    Give up on 4xx HTTP responses (the request itself is wrong; a retry
    won't change the answer). Keep retrying everything else: 5xx, plus
    connection-level errors with no response at all (timeouts, refused
    connections, DNS failures).
    """
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return 400 <= exc.response.status_code < 500
    return False


# encapsulates a shared API session and token refresh
class SessionManager:
    def __init__(self, auth_provider):
        self._auth_provider = auth_provider

    @property
    def session_token(self):
        return self._auth_provider.get_session_token()

    def refresh_session(self):
        self._auth_provider.refresh()


class BaseClient:
    def __init__(self, session_manager):
        self.session_manager = session_manager

    def retry_with_refresh(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in (401, 403):
                    log.warning("refreshing session")
                    self.session_manager.refresh_session()
                    return func(self, *args, **kwargs)
                raise

        return wrapper
