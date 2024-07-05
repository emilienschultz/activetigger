import { FC } from 'react';

import { useAppContext } from '../core/context';
import { LoginForm } from './forms/LoginForm';
import { PageLayout } from './layout/PageLayout';

export const LoginPage: FC = () => {
  const { appContext } = useAppContext();
  return (
    <PageLayout currentPage="login">
      <div className="container-fluid">
        {appContext.user?.username && (
          <div className="row">
            <div className="col-12 col-lg-6">
              You're logged in as {appContext.user.username} ({appContext.user.status})
            </div>
          </div>
        )}
        <div className="row">
          <div className="col-12 col-lg-6">
            <LoginForm />
          </div>
        </div>
      </div>
    </PageLayout>
  );
};
