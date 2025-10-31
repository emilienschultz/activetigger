import cx from 'classnames';
import { FC, useState } from 'react';
import { IoMdLogIn, IoMdLogOut } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import logo from '../../assets/at.png';
import { useAuth } from '../../core/auth';
import { useAppContext } from '../../core/context';

const DOCUMENTATION_LINK = 'https://emilienschultz.github.io/activetigger/docs';

interface NavBarPropsType {
  currentPage?: string;
  projectName?: string | null;
}

const NavBar: FC<NavBarPropsType> = ({ currentPage }) => {
  const { authenticatedUser, logout } = useAuth();
  const navigate = useNavigate();

  const [expanded, setExpanded] = useState<boolean>(false);

  // function to clear history
  const {
    appContext: { displayConfig },
  } = useAppContext();

  const PAGES: { id: string; label: string; href: string }[] =
    displayConfig.interfaceType === 'default'
      ? [
          { id: 'projects', label: 'Projects', href: '/projects' },
          { id: 'account', label: 'Account', href: '/account' },
          { id: 'users', label: 'Users', href: '/users' },
        ]
      : [
          { id: 'projects', label: 'Projects', href: '/projects' },
          { id: 'account', label: 'Account', href: '/account' },
        ];

  return (
    <div className="bg-primary" id="header">
      <div id="logo-container">
        <Link className="navbar-brand" to="/">
          <img
            src={logo}
            alt="ActiveTigger"
            className="me-2"
            style={{ width: '50px', height: '50px' }}
          />
          Active Tigger
        </Link>
      </div>
      <nav className="navbar navbar-dark navbar-expand-lg bg-primary">
        <div className="container">
          <button
            className="navbar-toggler"
            type="button"
            aria-controls="navbarSupportedContent"
            aria-expanded={expanded}
            aria-label="Toggle navigation"
            onClick={() => setExpanded((e) => !e)}
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div
            className={cx('navbar-collapse ', expanded ? 'expanded' : 'collapse')}
            id="navbarSupportedContent"
          >
            <ul className="navbar-nav me-auto mb-2 mb-lg-0 d-flex">
              {PAGES.map(({ id, label, href }) => (
                <li key={id} className="nav-item">
                  <Link
                    className={cx('nav-link', currentPage === id && 'active')}
                    aria-current={currentPage === id ? 'page' : undefined}
                    to={href}
                  >
                    {label}
                  </Link>
                </li>
              ))}
              <li className="nav-item" key="docs">
                <a
                  className={cx('nav-link', currentPage === 'docs' && 'active')}
                  href={DOCUMENTATION_LINK}
                  target="_blank"
                  rel="noreferrer"
                  aria-current={currentPage === 'docs' ? 'page' : undefined}
                >
                  Documentation
                </a>
              </li>
            </ul>
          </div>
        </div>
      </nav>

      {authenticatedUser ? (
        <div id="log-container">
          <div className="d-flex nav-item">
            {/* <button className="btn btn-primary clearhistory mx-1" onClick={actionClearHistory}>
                <FiRefreshCcw />
                <span className="badge badge-warning">{history.length}</span>
              </button> */}
            <Tooltip anchorSelect=".clearhistory" place="top">
              Clear the history
            </Tooltip>
          </div>
          <div className="nav-item">
            <span>Logged as {authenticatedUser.username}</span>
          </div>
          <div className="nav-item">
            <button
              className="btn btn-primary mx-2 logout"
              onClick={async () => {
                const success = await logout();
                if (success) navigate('/');
              }}
            >
              {' '}
              <IoMdLogOut title="Logout" />
            </button>
            <Tooltip anchorSelect=".logout" place="top">
              Log out
            </Tooltip>
          </div>
        </div>
      ) : (
        <Link to="/login">
          <IoMdLogIn title="login" />
        </Link>
      )}
    </div>
  );
};

export default NavBar;
