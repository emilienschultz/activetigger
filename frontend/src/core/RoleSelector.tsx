import { FC } from 'react';
import { Navigate, useParams } from 'react-router-dom';
import { useAuth } from './auth';

export interface RoleSelectorProps {
  allowedRoles: string[];
}

export const RoleSelector: FC<RoleSelectorProps> = ({ allowedRoles }) => {
  const { authenticatedUser } = useAuth(); // Assuming `user` has a `role` property
  const { projectName } = useParams();

  if (!authenticatedUser || !allowedRoles.includes(authenticatedUser.status as string)) {
    return <Navigate to={`/projects/${projectName}`} replace />;
  }
  return null;
};
