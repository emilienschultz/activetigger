import { FC, PropsWithChildren } from 'react';
import { Navigate, useLocation } from 'react-router-dom';

import { useAuth } from '../../core/auth';

export const AuthRequired: FC<PropsWithChildren> = ({ children }) => {
  const { authenticatedUser } = useAuth();

  const location = useLocation();
  return authenticatedUser ? (
    children
  ) : (
    <Navigate to="/login" replace state={{ path: location.pathname }} />
  );
};
