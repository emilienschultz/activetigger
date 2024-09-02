import cx from 'classnames';
import { FC, useState } from 'react';
import { IoMdLogIn, IoMdLogOut } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom';
import logo from '../../assets/at.png';
import { useAuth } from '../../core/auth';

const PAGES: { id: string; label: string; href: string }[] = [
  { id: 'projects', label: 'Projects', href: '/projects' },
  { id: 'users', label: 'Users', href: '/users' },
  { id: 'help', label: 'Help', href: '/help' },
  { id: 'login', label: 'Login', href: '/login' },
];

interface NavBarPropsType {
  currentPage?: string;
  projectName?: string | null;
}

const NavBar: FC<NavBarPropsType> = ({ currentPage }) => {
  const { authenticatedUser, logout } = useAuth();
  const navigate = useNavigate();

  const [expanded, setExpanded] = useState<boolean>(false);

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
            className={cx('navbar-collapse', expanded ? 'expanded' : 'collapse')}
            id="navbarSupportedContent"
          >
            <ul className="navbar-nav me-auto mb-2 mb-lg-0">
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
            {authenticatedUser ? (
              <span className="d-flex align-items-center navbar-text navbar-text-margins">
                Logged as {authenticatedUser.username}
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
              </span>
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
