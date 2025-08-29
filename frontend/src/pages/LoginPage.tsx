import { FC } from 'react';
import { useLocation } from 'react-router-dom';

import { LoginForm } from '../components/forms/LoginForm';
import { PageLayout } from '../components/layout/PageLayout';
import { useAuth } from '../core/auth';

export const LoginPage: FC = () => {
  const { authenticatedUser } = useAuth();
  const { state } = useLocation();

  return (
    <PageLayout currentPage="login">
      <div className="container-fluid">
        {authenticatedUser?.username && (
          <div className="row">
            <div className="col-1"></div>

            <div className="col-11 col-lg-6 ">
              <div className="user-info">
                You're logged in as <span>{authenticatedUser.username}</span> ( status :{' '}
                {authenticatedUser.status})
              </div>
            </div>
          </div>
        )}
        <div className="row">
          <div className="col-1"></div>

          <div className="col-11 col-lg-6">
            <div className="subsection m-2">Change account</div>

            <LoginForm redirectTo={state?.path || '/projects'} />
          </div>
        </div>
      </div>
    </PageLayout>
  );
};
