import logging

import requests

log = logging.getLogger()


# encapsulates a shared API session and token refresh
class SessionManager:
    def __init__(self, session_token, authentication_client=None, refresh_token=None):
        self.__session_token = session_token
        self.authentication_client = authentication_client
        self.refresh_token = refresh_token

    @property
    def session_token(self):
        return self.__session_token

    def refresh_session(self):
        if self.authentication_client is None or self.refresh_token is None:
            log.warning("session refresh is not available: no authentication client or refresh token configured")
            return

        log.info("refreshing session token using refresh token")
        self.__session_token = self.authentication_client.refresh(self.refresh_token)


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
