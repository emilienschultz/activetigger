import { FC } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import logo from '../assets/at.png';
import { LoginForm } from '../components/forms/LoginForm';
import Notifications from '../components/layout/Notifications';
import { useGetActiveUsers, useGetServer } from '../core/api';
import { useAuth } from '../core/auth';
import { LoginParams } from '../types';
export const HomePage: FC = () => {
  const { authenticatedUser } = useAuth();
  const { users } = useGetActiveUsers();

  // add a development mode switch
  // const {
  //   appContext: { developmentMode },
  //   setAppContext,
  // } = useAppContext();

  // function to change the status of the interface
  // const actionDevelopmentMode = () => {
  //   setAppContext((prev) => ({ ...prev, developmentMode: !prev.developmentMode }));
  // };

  // possibility to log directly from the URL
  const navigate = useNavigate();
  const params = new URLSearchParams(window.location.search);
  const { login } = useAuth();
  const { version, messages } = useGetServer(null);
  if (params.get('username') && params.get('password')) {
    login({
      username: params.get('username'),
      password: params.get('password'),
    } as LoginParams).then(() => {
      navigate('/projects');
      console.log('Connect');
    });
  }

  return (
    <>
      <main className="container-fluid">
        <div className="row">
          <center>
            <div className="alert alert-warning mt-3">
              ⚠️ This interface is in beta testing. Please save your data.{' '}
              <a href="https://github.com/emilienschultz/activetigger/issues">
                Please open a issue for any bug or problem
              </a>
            </div>
          </center>
          <div className="col-0 col-lg-3" />
          <div className="col-12 col-lg-6">
            <center>
              <div className="text-center">
                <h1
                  className="mb-1 fs-2 activetigger"
                  style={{
                    color: '#ff9a3c',
                  }}
                >
                  Active Tigger
                </h1>
                <h3 className="m-0 fs-5 text-muted fw-normal">Explore & Annotate Text</h3>
                <img
                  src={logo}
                  alt="ActiveTigger"
                  className="me-2"
                  style={{ width: '200px', height: '200px' }}
                />
              </div>

              {!authenticatedUser ? (
                <LoginForm />
              ) : (
                <div>
                  <div>
                    Welcome <span className="fw-bold">{authenticatedUser.username}</span>
                    <div className="justify-content-center">
                      <Link
                        to="/projects"
                        className="btn btn-lg text-white fw-bold shadow-sm px-4 py-2"
                        style={{
                          background: 'linear-gradient(90deg, #ff9a3c, #ff6f3c, #ffb347)',
                          border: 'none',
                          borderRadius: '2rem',
                          transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                        }}
                        onMouseOver={(e) => {
                          e.currentTarget.style.transform = 'scale(1.05)';
                          e.currentTarget.style.boxShadow = '0 8px 16px rgba(0,0,0,0.2)';
                        }}
                        onMouseOut={(e) => {
                          e.currentTarget.style.transform = 'scale(1)';
                          e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.1)';
                        }}
                      >
                        🐯 Go to your projects
                      </Link>
                    </div>
                    <div style={{ maxWidth: '600px', margin: '1rem auto' }}>
                      {(messages || []).map((msg) => (
                        <div
                          key={msg.id}
                          style={{
                            fontSize: '0.9rem',
                            color: '#333',
                          }}
                        >
                          <div style={{ color: '#777' }}>
                            {msg.content} •{' '}
                            <span style={{ fontSize: '0.5rem' }}>
                              {new Date(msg.time || Date.now()).toLocaleString()}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {!users && (
                <div className="alert alert-alert mt-3">Problem connecting to the server</div>
              )}

              <div className="general-info mt-3">
                <div>
                  Current users : {users?.length} <br />
                  Backend version <b>{version}</b> - Last update of the frontend
                  <b> {__BUILD_DATE__}</b>
                </div>
              </div>
            </center>
            <div style={{ height: '50px' }}></div>
          </div>
        </div>
        <footer className="footer mt-auto py-1 bg-primary text-white fixed-bottom">
          <div className="container text-center">
            <i className="fas fa-info-circle"></i>
            <span className="ml-2">
              CREST / CSS @ IPP © 2025 - Julien Boelaert & Étienne Ollion &{' '}
              <a href="https://www.ouestware.com/" style={{ all: 'unset', cursor: 'pointer' }}>
                Ouestware
              </a>{' '}
              &{' '}
              <a href="http://eschultz.fr" style={{ all: 'unset', cursor: 'pointer' }}>
                Émilien Schultz
              </a>{' '}
            </span>
          </div>
        </footer>
      </main>
      <Notifications />
    </>
  );
};
