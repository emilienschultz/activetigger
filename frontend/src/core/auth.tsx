import { FC, PropsWithChildren, createContext, useCallback, useContext, useState } from 'react';

import { LoginParams, UserModel } from '../types';
import { HttpError } from './HTTPError';
import { login, logout, me } from './api';
import { useNotifications } from './notifications';

// Information about the current authenticated user
export type AuthenticatedUser = UserModel & {
  access_token?: string;
};
// the global auth context which provide the authenticated user and the login internal logic
export type AuthContext = {
  authenticatedUser?: AuthenticatedUser;
  login: (params: LoginParams) => Promise<void>;
  logout: () => Promise<boolean>;
};

// create a react context to centralize and share auth state and mechanism with the whole application
const authContext = createContext<AuthContext>({
  authenticatedUser: undefined,
  login: async (_: LoginParams) => {},
  logout: async () => false,
});

// internal hook which must not be used outside the context
const _useAuth = (): AuthContext => {
  // we use localstorage to persist session
  const storedAuth = localStorage.getItem('activeTigger.auth');
  // TODO check for session deprecation

  // internal state to store the current authenticated user
  const [authenticatedUser, setAuthenticatedUser] = useState<AuthenticatedUser | undefined>(
    // by default we load the local storage version
    storedAuth ? JSON.parse(storedAuth) : undefined,
  );

  // notifications
  const { notify } = useNotifications();

  /**
   * This method wraps the login API call into our authenticated user state management
   * It does the call and make sure to update our internal state accordingly
   * by providing this method we encapsulate the setter and make sure to control all updates
   */
  const _login = useCallback(
    async (params: LoginParams) => {
      try {
        const response = await login(params);
        if (response.access_token) {
          const user = await me(response.access_token);

          if (user !== undefined) {
            const authUser = { ...user, access_token: response.access_token };
            localStorage.setItem('activeTigger.auth', JSON.stringify(authUser));
            setAuthenticatedUser(authUser);
            notify({ type: 'success', message: `Logged in as ${user.username}` });
          } else {
            throw new Error('Good token but no user?');
          }
        }
      } catch (error) {
        notify({ type: 'warning', message: 'could not authenticate. Please retry.' });
        localStorage.removeItem('activeTigger.auth');
        setAuthenticatedUser(undefined);
      }
    },
    // the method code will change if setAuthenticatedUser changes which happens only at init
    [setAuthenticatedUser, notify],
  );

  const _logout = useCallback(
    async () => {
      if (authenticatedUser && authenticatedUser.access_token) {
        try {
          const success = await logout(authenticatedUser.access_token);
          if (success) {
            localStorage.removeItem('activeTigger.auth');
            setAuthenticatedUser(undefined);
            return success;
          }
          return false;
        } catch (error) {
          console.log(error);
          if (error instanceof HttpError) {
            // TODO: create a nice message depending on error.status
          }
          notify({ type: 'error', message: error + '' });
          return false;
        }
      } else {
        notify({ type: 'warning', message: 'You must be logged-in to be able to log out!' });
        return false;
      }
    },
    // the method code will change if setAuthenticatedUser changes which happens only at init
    [setAuthenticatedUser, notify, authenticatedUser],
  );

  // returns the AuthContext new value each time the authenticatedUser changes
  return {
    authenticatedUser,
    login: _login,
    logout: _logout,
  };
};

/**
 * AuthProvider a react context component which will propagate the auth context value to all its children
 */
export const AuthProvider: FC<PropsWithChildren> = ({ children }) => {
  const auth = _useAuth();
  return <authContext.Provider value={auth}>{children}</authContext.Provider>;
};

/**
 * useAuth
 * the main auth hook which provides the auth context current value
 * @returns
 */
export function useAuth() {
  // we could use useContext(authContext) in our components but we reright a more elegant useAuth for clarity
  return useContext(authContext);
}
/**
 * getAuthHeaders
 * utility functions which provides the HTTP headers to be included in API calls
 * we write this code here to centralize auth logics
 * @param user a Authenticated user
 * @returns Authorization and username HTTP headers to be included in API calls
 */
export function getAuthHeaders(user?: AuthenticatedUser) {
  return user
    ? {
        headers: {
          Authorization: `Bearer ${user.access_token}`,
          username: user.username,
        },
      }
    : undefined;
}
