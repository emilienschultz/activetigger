import classNames from 'classnames';
import { FC } from 'react';
import { FaCogs } from 'react-icons/fa';
import { PiTagDuotone } from 'react-icons/pi';
import { Link } from 'react-router-dom';

import { PossibleProjectActions } from './ProjectPageLayout';

/* define a component for project action bar 
with the project & the current action*/
export const ProjectActionsSidebar: FC<{
  projectName: string;
  currentProjectAction?: PossibleProjectActions;
}> = ({ currentProjectAction, projectName }) => {
  return (
    <div className="project-sidebar d-flex flex-column flex-shrink-0 p-3 bg-light">
      <ul className="nav nav-pills flex-column mb-auto">
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}`}
            className={classNames('nav-link', !currentProjectAction && 'active')}
            aria-current="page"
          >
            Project{' '}
            <span>
              <b>{projectName}</b>
            </span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/annotate`}
            className={classNames('nav-link', currentProjectAction === 'annotate' && 'active')}
            aria-current="page"
          >
            <PiTagDuotone /> Annotate
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/parameters`}
            className={classNames('nav-link', currentProjectAction === 'parameters' && 'active')}
            aria-current="page"
          >
            <span className="parameters">
              <FaCogs /> Parameters
            </span>
          </Link>
        </li>
      </ul>
    </div>
  );
};
