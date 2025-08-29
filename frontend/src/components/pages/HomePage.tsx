import { FC } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import logo from '../../assets/at.png';
import { useGetActiveUsers, useGetVersion } from '../../core/api';
import { useAuth } from '../../core/auth';
import { LoginParams } from '../../types';
import { LoginForm } from './../forms/LoginForm';
import Notifications from './../layout/Notifications';
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
  const { version } = useGetVersion();
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
          <div className="col-0 col-lg-3" />
          <div className="col-12 col-lg-6">
            <center>
              <h1>Active Tigger</h1>
              <h3>Explore, Annotate & Classify Text</h3>

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
                    You are logged in as <span>{authenticatedUser.username}</span> ( status :{' '}
                    {authenticatedUser.status}){' '}
                    {/* <div className="form-check form-switch">
                      <label className="form-check-label" htmlFor="devMode">
                        <input
                          className="form-check-input mx-2"
                          type="checkbox"
                          role="switch"
                          id="devMode"
                          checked={developmentMode}
                          onChange={actionDevelopmentMode}
                        />
                        Dev mode
                        <a className="batchsize mx-2">
                          <HiOutlineQuestionMarkCircle />
                        </a>
                        <Tooltip anchorSelect=".batchsize" place="top">
                          Dev mode will display experimental features that have not been tested
                          extensively.
                        </Tooltip>
                      </label>
                    </div> */}
                    <div>
                      {users ? (
                        <div className="explanations">Active users : {users?.length}</div>
                      ) : (
                        <div className="text-danger">Problem connecting to the server</div>
                      )}
                    </div>
                    <div className="justify-content-center">
                      <Link
                        to="/projects"
                        className="btn btn-primary btn-lg shadow-sm rounded-pill m-3"
                      >
                        Go to your projects
                      </Link>
                    </div>
                  </div>
                </div>
              )}
              <div className="alert alert-warning fw-bold mt-3">
                ⚠️ Warning: This interface is in beta testing.
                <br></br>
                Continuity of service is not guaranteed, please save your data. <br></br>
                <a href="https://github.com/emilienschultz/activetigger/issues">
                  Please report any bug or problem on the Github of the project
                </a>
                .
              </div>

              <div className="general-info mt-3">
                <div>
                  Backend version <b>{version}</b> - Last update of the frontend
                  <b> {__BUILD_DATE__}</b>
                </div>
                <div>For any information, please contact emilien.schultz [at] ensae.fr</div>
              </div>
            </center>
            <div style={{ height: '50px' }}></div>
          </div>
        </div>
        <footer className="footer mt-auto py-1 bg-primary text-white fixed-bottom">
          <div className="container text-center">
            <i className="fas fa-info-circle"></i>
            <span className="ml-2">
              CREST / CSS @ IPP © 2025 -{' '}
              <i>
                credits : Julien Boelaert & Étienne Ollion &{' '}
                <a href="https://www.ouestware.com/" style={{ all: 'unset', cursor: 'pointer' }}>
                  Ouestware
                </a>{' '}
                &{' '}
                <a href="http://eschultz.fr" style={{ all: 'unset', cursor: 'pointer' }}>
                  Émilien Schultz
                </a>{' '}
              </i>
            </span>
          </div>
        </footer>
      </main>
      <Notifications />
    </>
  );
};
