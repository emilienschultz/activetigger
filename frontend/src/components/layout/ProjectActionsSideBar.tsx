import classNames from 'classnames';
import { FC } from 'react';
import { PiTagDuotone } from 'react-icons/pi';
import { Link } from 'react-router-dom';

import { PossibleProjectActions } from './ProjectPageLayout';

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
            {projectName}
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
      </ul>
    </div>
  );
};
