import {
  FC,
  PropsWithChildren,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';

import { UserModel } from '../types';
import { LoginParams, login, me } from './api';

export type AuthenticatedUser = UserModel & {
  access_token?: string;
};
export type AuthContext = {
  authenticatedUser?: AuthenticatedUser;
  login: (params: LoginParams) => Promise<void>;
};

const authContext = createContext<AuthContext>({
  authenticatedUser: undefined,
  login: async (_: LoginParams) => {},
});

const _useAuth = (): AuthContext => {
  const storedAuth = localStorage.getItem('activeTigger.auth');

  const [authenticatedUser, setAuthenticatedUser] = useState<AuthenticatedUser | undefined>(
    storedAuth ? JSON.parse(storedAuth) : {},
  );

  useEffect(() => {
    localStorage.setItem('activeTigger.auth', JSON.stringify(authenticatedUser));
  }, [authenticatedUser]);

  // TODO check session validity
  const _login = useCallback(
    async (params: LoginParams) => {
      const response = await login(params);
      if (response.access_token) {
        const user = await me(response.access_token);

        if (user !== undefined) {
          setAuthenticatedUser({ ...user, access_token: response.access_token });
        } else setAuthenticatedUser(undefined);
      }
    },
    [setAuthenticatedUser],
  );

  return {
    authenticatedUser,
    login: _login,
    //TODO: logout
  };
};

export const AuthProvider: FC<PropsWithChildren> = ({ children }) => {
  const auth = _useAuth();
  return <authContext.Provider value={auth}>{children}</authContext.Provider>;
};

export function useAuth() {
  return useContext(authContext);
}

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
