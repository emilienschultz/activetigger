import { FC, PropsWithChildren } from 'react';
import { Navigate, useLocation } from 'react-router-dom';

import { useAuth } from '../../core/auth';

/**
 * AuthRequired
 * a component which protect private routes from access by unauthenticated user
 * @returns children or a redirection to the login form
 */
export const AuthRequired: FC<PropsWithChildren> = ({ children }) => {
  // first we get authenticated user state
  const { authenticatedUser } = useAuth();
  // location is provided by react-router library and contains the current page (i.e. in which URL path this component has been mounted)
  const location = useLocation();

  return authenticatedUser ? (
    // if the user is currently authenticated just let him go through its destination by mounting children (i.e. the requested page component)
    children
  ) : (
    // else ask him to first login and then redirects him back to where we wanted to go
    <Navigate to="/login" replace state={{ path: location.pathname }} />
  );
};
