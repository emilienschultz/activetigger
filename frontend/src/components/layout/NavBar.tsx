import cx from 'classnames';
import { FC, useState } from 'react';
import { FiRefreshCcw } from 'react-icons/fi';
import { IoMdLogIn, IoMdLogOut } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import logo from '../../assets/at.png';
import { useGetServer } from '../../core/api';
import { useAuth } from '../../core/auth';
import { useAppContext } from '../../core/context';

const PAGES: { id: string; label: string; href: string }[] = [
  { id: 'projects', label: 'Projects', href: '/projects' },
  // { id: 'monitor', label: 'Monitor', href: '/monitor' },
  // { id: 'help', label: 'Help', href: '/help' },
  { id: 'account', label: 'Account', href: '/account' },
  { id: 'users', label: 'Users', href: '/users' },
  {
    id: 'docs',
    label: 'Documentation',
    href: '/documentation',
  },
];

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
    appContext: { history, currentProject },
    setAppContext,
  } = useAppContext();
  const actionClearHistory = () => {
    setAppContext((prev) => ({ ...prev, history: [] }));
  };

  // display the number of current processes on the server
  const { queueState, gpu } = useGetServer(currentProject || null);

  return (
    <div className="bg-primary">
      <nav className="navbar navbar-dark navbar-expand-lg bg-primary">
        <div className="container">
          <Link className="navbar-brand" to="/">
            <img
              src={logo}
              alt="ActiveTigger"
              className="me-2"
              style={{ width: '50px', height: '50px' }}
            />
            Active Tigger
          </Link>
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
            </ul>
            <ul className="navbar-nav me-auto mb-2 mb-lg-0 d-flex">
              <li className="m-1">
                <div
                  className="nav-item badge text-bg-secondary"
                  title="Number of processes running"
                >
                  <span className="d-none d-md-inline">Process: </span>
                  {Object.values(queueState || []).length}
                </div>
              </li>
              <li className="m-1">
                <div className="badge text-bg-secondary" title="Used/Total">
                  <span className="d-none d-md-inline">GPU:</span>
                  {gpu
                    ? `${(gpu['total_memory'] - gpu['available_memory']).toFixed(1)} / ${gpu['total_memory']} Go`
                    : 'No'}
                </div>
              </li>
            </ul>

            {authenticatedUser ? (
              <ul className="d-flex navbar-nav me-auto mb-2 mb-lg-0 navbar-text navbar-text-margins align-items-center">
                <li className="d-flex nav-item">
                  <button className="btn btn-primary clearhistory" onClick={actionClearHistory}>
                    <FiRefreshCcw />
                    <span className="badge badge-warning">{history.length}</span>
                  </button>
                </li>
                <li className="nav-item">
                  <span>Logged as {authenticatedUser.username}</span>
                </li>
                <li className="nav-item">
                  <Tooltip anchorSelect=".clearhistory" place="top">
                    Clear the current session (you can only annotate once each element by session)
                  </Tooltip>
                  <button
                    className="btn btn-primary"
                    onClick={async () => {
                      const success = await logout();
                      if (success) navigate('/');
                    }}
                  >
                    {' '}
                    <IoMdLogOut title="Logout" />
                  </button>
                </li>
              </ul>
            ) : (
              <Link to="/login">
                <IoMdLogIn title="login" />
              </Link>
            )}
          </div>
        </div>
      </nav>
    </div>
  );
};

export default NavBar;
