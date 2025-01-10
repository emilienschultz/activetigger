import { FC } from 'react';
import { useLocation } from 'react-router-dom';
import { useAuth } from '../../core/auth';
import { ChangePassword } from './../forms/ChangePassword';
import { LoginForm } from './../forms/LoginForm';
import { PageLayout } from './../layout/PageLayout';

export const AccountPage: FC = () => {
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
              <ChangePassword />
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
