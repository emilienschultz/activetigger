import { FC } from 'react';
import { Link } from 'react-router-dom';

import { useAuth } from '../core/auth';
import { LoginForm } from './forms/LoginForm';
import Notifications from './layout/Notifications';

export const HomePage: FC = () => {
  const { authenticatedUser } = useAuth();
  return (
    <>
      <main className="container-fluid">
        <div className="row">
          <div className="col-0 col-lg-3" />
          <div className="col-12 col-lg-6">
            <h1>Active tigger</h1>
            <h3>Explore & Annotate textual data</h3>
            {!authenticatedUser ? (
              <LoginForm />
            ) : (
              <div>
                <p>
                  you're logged in as <b>{authenticatedUser.username}</b> (
                  {authenticatedUser.status})
                </p>
                <Link to="/projects">your projects</Link>
              </div>
            )}
            <div className="general-info">
              <div>Frontend v0.1</div>
              <div>For any information, please contact emilien.schultz [at] ensae.fr</div>
            </div>
          </div>
        </div>
      </main>
      <Notifications />
    </>
  );
};
