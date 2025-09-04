import { FC } from 'react';
import { ChangePassword } from '../components/forms/ChangePassword';
import { PageLayout } from '../components/layout/PageLayout';
import { useAuth } from '../core/auth';

export const AccountPage: FC = () => {
  const { authenticatedUser } = useAuth();

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
      </div>
    </PageLayout>
  );
};
