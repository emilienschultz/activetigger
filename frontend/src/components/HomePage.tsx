import { FC } from 'react';
import { Link } from 'react-router-dom';

import logo from '../assets/at.png';
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
            <center>
              <h1>Active Tigger</h1>
              <h3>Explore, Classify & Analyze Text</h3>
              <img
                src={logo}
                alt="ActiveTigger"
                className="me-2"
                style={{ width: '200px', height: '200px' }}
              />
              {!authenticatedUser ? (
                <LoginForm />
              ) : (
                <div>
                  <div className="user-info">
                    You're logged in as <span>{authenticatedUser.username}</span> ( status :{' '}
                    {authenticatedUser.status})
                  </div>
                  <Link
                    to="/projects"
                    className="btn btn-primary btn-lg shadow-sm rounded-pill m-3"
                  >
                    Go to your projects
                  </Link>
                </div>
              )}
              <div className="general-info">
                <div>Frontend v0.6</div>
                <div>For any information, please contact emilien.schultz [at] ensae.fr</div>
                <div className="text-muted">
                  Credits : Julien Boelaert & Étienne Ollion & Émilien Schultz & Ouestware
                </div>
              </div>
            </center>
          </div>
        </div>
        <footer className="footer mt-auto py-1 bg-primary text-white fixed-bottom">
          <div className="container text-center">
            <i className="fas fa-info-circle"></i>
            <span className="ml-2">
              CREST / CSS @ IPP © 2024 - <i>under development -</i>
            </span>
          </div>
        </footer>
      </main>
      <Notifications />
    </>
  );
};
