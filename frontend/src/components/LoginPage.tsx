import { FC, useEffect } from 'react';
import { useLocation } from 'react-router-dom';

import { useAuth } from '../core/auth';
import { LoginForm } from './forms/LoginForm';
import { PageLayout } from './layout/PageLayout';

export const LoginPage: FC = () => {
  const { authenticatedUser } = useAuth();
  const { state } = useLocation();

  const handleKeyboardEvents = (ev: KeyboardEvent) => {
    if (ev.code === 'Numpad1') {
      // apply tag #1
    }

    console.log(ev.code);
  };
  useEffect(() => {
    document.addEventListener('keydown', handleKeyboardEvents);

    return () => document.removeEventListener('keydown', handleKeyboardEvents);
  }, []);

  return (
    <PageLayout currentPage="login">
      <div className="container-fluid">
        {authenticatedUser?.username && (
          <div className="row">
            <div className="col-12 col-lg-6">
              You're logged in as {authenticatedUser.username} ({authenticatedUser.status})
            </div>
          </div>
        )}
        <div className="row">
          <div className="col-12 col-lg-6">
            <LoginForm redirectTo={state?.path || '/projects'} />
          </div>
        </div>
      </div>
    </PageLayout>
  );
};
